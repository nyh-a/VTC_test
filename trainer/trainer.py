import numpy as np
import torch
from torch import autograd, nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle as pkl
import random
from tqdm import tqdm
from torchvision.utils import make_grid
from base import BaseTrainer
from utils import inf_loop, MetricTracker, calc_metrics
from model.loss import *
from utils import print_memory_usage
from sentence_transformers import SentenceTransformer, util
import time
from torch.amp import GradScaler, autocast
import gc 
from data_loader.dataset import QueryEvalDataset, CandidateEvalDataset
from data_loader.data_loaders import eval_collate_fn


class Trainer_S2(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, config, device,
                 data_loader, # Training loader
                 lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.data_loader = data_loader
        self.device = device
        self.len_epoch = len(self.data_loader) if len_epoch is None else len_epoch
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer'].get('log_step', int(np.sqrt(data_loader.batch_size))) * 5
        self.use_amp = config['trainer'].get('mixed_precision', True)
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        # self.margin = config['trainer'].get('margin', 0.5)
        self.train_metrics = MetricTracker('loss', writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.scale = 20.0
        
        # Guide Model Related
        self.guide_model = config['trainer'].get('guide_model', None)
        self.use_text_guide = config['trainer'].get('use_text_guide', True)
        self.use_visual_guide = config['trainer'].get('use_visual_guide', True)
        self.dirty_pos_threshold = config['trainer'].get('dirty_pos_threshold', 0.45)

        self.validation_steps = config['trainer'].get('validation_steps', 50) 

        print_memory_usage(note="Trainer Initialized")
        self._setup_eval_infrastructure()
        print_memory_usage(note="_setup_eval_infrastructure")


    def _setup_eval_infrastructure(self):
        self.logger.info("Setting up evaluation DataLoaders and pre-processing unique images...")
        self.eval_batch_size = self.config['trainer'].get('eval_batch_size', 128)
        self.ref_loader = self.data_loader
        g_data = self.ref_loader.dataset
        
        self.max_len = g_data.max_len
        
        
        # 2. Create Query and Candidate Datasets & DataLoaders for evaluation
        query_dataset = QueryEvalDataset(g_data.test_node_list, g_data.feature_dir, g_data.tokenizer, g_data.num_tokens, self.max_len)
        self.query_test_loader = DataLoader(
            query_dataset, batch_size=self.eval_batch_size, num_workers=self.ref_loader.num_workers,
            collate_fn=eval_collate_fn, pin_memory=True, shuffle=False)

        candidate_dataset = CandidateEvalDataset(g_data.all_edges, g_data, g_data.feature_dir, self.ref_loader.tokenizer, g_data.num_tokens, self.max_len)
        self.candidate_test_loader = DataLoader(
            candidate_dataset, batch_size=self.eval_batch_size, num_workers=self.ref_loader.num_workers,
            collate_fn=eval_collate_fn, pin_memory=True, shuffle=False)
        
        self.eval_candidate_info = {
            "node_pair_to_idx": {pair: i for i, pair in enumerate(candidate_dataset.candidate_position_nodes)},
            "count": len(candidate_dataset)
        }
        self.logger.info(f"Test loaders created. Queries: {len(query_dataset)}, Candidates: {self.eval_candidate_info['count']}")


        if self.criterion.use_text:
            self.logger.info("Initializing Text Guide (One-off Pre-computation)...")
            self._precompute_guide_embeddings()
        else:
            self.guide_q_emb = None
            self.guide_pos_map = None

    def _precompute_guide_embeddings(self):
        """
        [One-off] 使用 SentenceTransformer (Qwen3) 预计算所有 Guide Embeddings。
        启用 Flash Attention 2 加速。
        """
        model_name = self.guide_model.split("/")[-1] if self.guide_model else "Qwen3-Embedding-0.6B"
        self.logger.info(f">>> [Pre-compute] Initializing Guide Model ({model_name}) with FA2...")
        
        # 加载模型: 启用 Flash Attention 2 和 FP16
        # 注意: 需要 GPU 支持 (Ampere架构如 3090/4090/A100)
        try:
            model = SentenceTransformer(
                self.guide_model,
                trust_remote_code=True,
                device=self.device,
                model_kwargs={"dtype": "bfloat16"}
            )
        except Exception as e:
            self.logger.warning(f"Failed to init with FA2: {e}. Fallback to default settings.")
            model = SentenceTransformer(guide_model_name, trust_remote_code=True, device=self.device)
            model.half() # 尝试转半精度
            
        # Evaluation mode (SentenceTransformer 默认 eval)
        
        dataset = self.data_loader.dataset
        taxon2id = dataset.taxon2id
        
        all_nodes = dataset.node_list 
        num_nodes = len(all_nodes)
        
        # 1. 建立映射: Real ID -> Dense Index (0..N-1)
        self.q_id_to_dense_idx = {}
        
        # 2. 预分配紧凑的 Tensor
        self.guide_q_emb = torch.zeros((num_nodes, 1024), dtype=torch.float16, device=self.device)
        
        self.logger.info(f"Encoding {num_nodes} Queries into compact tensor...")
        
        query_texts = []
        dense_indices = []
        
        for i, node in enumerate(all_nodes):
            term = node.display_name # 假设 Node 对象有这个属性
            real_id = taxon2id[node]
            
            # 记录映射
            self.q_id_to_dense_idx[real_id] = i
            
            # 准备文本
            defn = dataset.id2desc[real_id]
            query_texts.append(f"Query Node: {term}. Definition: {defn}")
            dense_indices.append(i)
            
        # 批量编码
        q_embs = model.encode(
            query_texts, 
            batch_size=256,
            convert_to_tensor=True, 
            show_progress_bar=True, 
            normalize_embeddings=True
        )
        
        # 存入 Tensor (按 dense index 顺序)
        # 如果 q_embs 顺序和 query_texts 一致，直接赋值即可，不用切片
        self.guide_q_emb = q_embs.to(torch.float16)
        
        del q_embs, query_texts
        torch.cuda.empty_cache()

        # ==========================================
        # B. Encode Positions (Dictionary)
        # Key: (p_id, c_id) -> Value: Tensor
        # ==========================================
            
        self.logger.info(f"Encoding {len(dataset.all_edges)} unique Candidate Positions...")
        
        self.guide_pos_map = {} # 存在 GPU 上
        
        pos_list = dataset.all_edges
        chunk_size = 256
        
        for i in tqdm(range(0, len(pos_list), chunk_size), desc="Guide Positions"):
            batch_pairs = pos_list[i : i + chunk_size]
            batch_texts = []
            batch_keys = []
            
            for p_node, c_node in batch_pairs:
                # 获取原始 ID 作为 Key
                p_id = taxon2id[p_node]
                c_id = taxon2id[c_node]
                batch_keys.append((p_id, c_id))
                
                # 确定性 Sibling
                s_node = dataset._get_sibling(p_node, c_node)
                s_id = taxon2id[s_node]
                
                
                # 构造文本
                p_def = dataset.id2desc[p_id]
                c_def = dataset.id2desc[c_id]
                s_def = dataset.id2desc[s_id]
                
                text = (
                    f"Parent Node: {p_node.display_name}. Definition: {p_def}; "
                    f"Child Node: {c_node.display_name}. Definition: {c_def}; "
                    f"Sibling Node: {s_node.display_name}. Definition: {s_def}"
                )
                batch_texts.append(text)
            
            embs = model.encode(
                batch_texts,
                batch_size=chunk_size, 
                convert_to_tensor=True,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            for key, emb in zip(batch_keys, embs):
                self.guide_pos_map[key] = emb.to(torch.float16)
        
        self.logger.info("Injecting Guide Data into Loss Criterion...")
        self.criterion.set_guide_data(
            guide_q_emb=self.guide_q_emb,       # Tensor
            q_id_to_dense=self.q_id_to_dense_idx, # Dict
            guide_pos_map=self.guide_pos_map    # Dict
        )
                
        del model
        del self.guide_q_emb
        del self.guide_pos_map
        del self.q_id_to_dense_idx
        torch.cuda.empty_cache()
        self.logger.info("Guide Embeddings computed and cached.")

    def _train_epoch(self, epoch):
        self.model.train()
        self.train_metrics.reset()
        pbar = tqdm(self.data_loader, desc=f"Train Epoch: {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            batch_dev = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            batch_dev['pos_id_pair'] = batch['pos_id_pair'] 

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp, device_type='cuda'):
                q_emb, c_emb = self.model(batch_dev)
                loss = self.criterion(q_emb, c_emb, batch_dev)

            
            self.scaler.scale(loss).backward()          
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # log
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            
            if batch_idx % self.log_step == 0: 
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                
            if epoch > 5 and (batch_idx + 1) % self.validation_steps == 0 and batch_idx > 0:
                self.logger.info(f"--- Running validation at step {batch_idx + 1} ---")
                
                # 直接调用你已经写好的验证函数
                val, leaf_val, nonleaf_val = self._valid_epoch(epoch, "test")
                log = {**self.train_metrics.result(), **val, **leaf_val, **nonleaf_val}
                self.logger.info(f"Validation results at step {batch_idx + 1}: {log}")
                for key, value in log.items():
                    self.logger.info('    {:15s}: {}'.format(str(key), value))

            # For debug valid_epoch
            # break
            
            # For test both train and valid
            # if batch_idx == 5: break
            
            # For normal train
            if batch_idx == self.len_epoch: break
        
        log = self.train_metrics.result()
        if self.config['trainer']['do_validation']:
            val, leaf_val, nonleaf_val = self._valid_epoch(epoch, "test")
            log = {**log, **val, **leaf_val, **nonleaf_val}

        return log

    def _valid_epoch(self, epoch, mode="test"):
        self.logger.info(f">>> Starting Global Ranking Validation ({mode})...")
        self.model.eval()
        self.valid_metrics.reset() # Assuming one metrics tracker for simplicity
        print_memory_usage(f"Start Validation {mode}")
        
        dataset = self.data_loader.dataset
        taxon2id = self.data_loader.dataset.taxon2id
        node2pos = dataset.test_node2pos if mode == "test" else dataset.valid_node2pos
        
        
        
        self.logger.info("Classifying queries into leaf and non-leaf...")
        
        # 获取当前验证集所有的 Query Nodes
        # 注意：query_test_loader.dataset.nodes 存储的是 Query Node 对象或字符串
        queries_to_classify = self.query_test_loader.dataset.query_nodes
        leaf_queries_set = set()
        
        for query_node in queries_to_classify:
            # 获取该 Query 的所有正确位置
            poses = node2pos.get(query_node, set())
            if not poses: 
                continue
            
            # 核心判断逻辑：如果所有位置的 Child 都是 Pseudo Leaf，则该 Query 是 Leaf Node
            # 注意：这里需要确保 pos[1] 对象能和 pseudo_leaf_node 正确比较
            is_leaf_query = all(pos[1] == dataset.pseudo_leaf_node for pos in poses)
            
            if is_leaf_query:
                leaf_queries_set.add(query_node)
                
        self.logger.info(f"Found {len(leaf_queries_set)} leaf queries out of {len(queries_to_classify)} total.")
        
        # --- 1. Encode Candidates (All Positions) ---
        cand_embs = []
        cand_id_pair_to_emb_row = {}
        processed_cand_count = 0
        
        with torch.no_grad():
            for batch in tqdm(self.candidate_test_loader, desc="Enc Candidates"):
                # Encode
                batch_dev = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                embs = self.model.encode_candidate(batch_dev)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                cand_embs.append(embs.cpu())
                
                # Record Metadata
                # batch['pos_id_pair'] 是 [(p1, c1), (p2, c2)...] 的列表
                for p_id, c_id in batch['pos_id_pair']:
                    # 确保是 tuple 以便作为 dict key
                    pair_key = (int(p_id), int(c_id))
                    cand_id_pair_to_emb_row[pair_key] = processed_cand_count
                    processed_cand_count += 1
        
        C = torch.cat(cand_embs, dim=0) # [N_Cand, Dim]
        
        # --- 2. Encode Queries & Rank ---
        
        
        all_ranks, leaf_ranks, nonleaf_ranks = [], [], []
        
        with torch.no_grad():
            for batch in tqdm(self.query_test_loader, desc="Eval Queries"):
                # Encode Query
                batch_dev = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                q_embs = self.model.encode_query(batch_dev)
                q_embs = torch.nn.functional.normalize(q_embs, p=2, dim=1)
                
                # Similarity: [B, N_Cand]
                # 注意：如果 N_Cand 很大，建议分块计算，这里假设能存下
                sims = torch.mm(q_embs, C.to(self.device).t())
                
                # Sort: [B, N_Cand] (Descending)
                _, indices = torch.sort(sims, dim=1, descending=True)
                indices = indices.cpu().numpy()
                
                # Calculate Metrics per Query
                current_q_nodes = batch['query_node']
                
                for i, q_node in enumerate(current_q_nodes):
                   
                    true_pos_pairs = node2pos.get(q_node, set())
                    if not true_pos_pairs: continue
                    
                    ranks_for_this_query = []
                    for p_node, c_node in true_pos_pairs:
                    # a. 将真实正例的节点对象转换为 ID 对 (p_id, c_id)
                        true_id_pair = (taxon2id[p_node], taxon2id[c_node])
                        true_pos_column_idx = cand_id_pair_to_emb_row.get(true_id_pair)
                            
                        if true_pos_column_idx is not None:
                            # 3. 在预测排序 indices[i] 中查找该行号的位置
                            # np.where 返回元组，取 [0] 得到 array
                            rank_loc = np.where(indices[i] == true_pos_column_idx)[0]
                            
                            if len(rank_loc) > 0:
                                # 0-based index -> 1-based rank
                                rank = rank_loc[0] + 1
                                ranks_for_this_query.append(rank)
                            else:
                                self.logger.error(f"Logic Error: Candidate index {true_pos_column_idx} not found in sorted indices.")

                    

                    if ranks_for_this_query:
                        # Append to the correct list based on pre-classification
                        all_ranks.append(ranks_for_this_query)
                        if q_node in leaf_queries_set:
                            leaf_ranks.append(ranks_for_this_query)
                        else:
                            nonleaf_ranks.append(ranks_for_this_query)
        
        self.logger.info(f"Finished evaluation. Total rank sets: {len(all_ranks)}, Leaf: {len(leaf_ranks)}, Non-Leaf: {len(nonleaf_ranks)}")

        if not all_ranks:
            self.logger.error("Metrics calculation failed: all_ranks list is empty.")
            return {m.__name__: 0 for m in self.metric_ftns}, {}, {}
    
        return calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks)

    
    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)