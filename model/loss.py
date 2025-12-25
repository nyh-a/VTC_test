import re
from itertools import product

import more_itertools as mit
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

EPS = 1e-9

def ce_loss(output, target):
    return F.cross_entropy(output, target)

def bce_loss(output, target):
    return F.binary_cross_entropy(output, target)

def bceWlogit_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

    

COSINE_DISTANCE = lambda x,y: 1 - F.cosine_similarity(x,y)

def OnlineContrastiveLoss(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Online Contrastive loss. Similar to ConstrativeLoss, but it selects hard positive (positives that are far apart)
    and hard negative pairs (negatives that are close) and computes the loss only for these pairs. Often yields
    better performances than ConstrativeLoss.
    """
    distance_matrix = distance_metric(emb_a, emb_b)
    negs = distance_matrix[labels == 0]
    poss = distance_matrix[labels == 1]

    # select hard positive and hard negative pairs
    negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
    positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

    positive_loss = positive_pairs.pow(2).sum()
    negative_loss = F.relu(margin - negative_pairs).pow(2).sum()
    
    # sum
    loss = positive_loss + negative_loss
    # mean1 actual calculated sample 
    # loss = (positive_loss + negative_loss) / (len(positive_pairs) + len(negative_pairs) + 1e-8)
    # mean2 all batch
    # loss = (positive_loss + negative_loss) / len(emb_a)
    return loss

def ContrastiveLoss(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    """
    size_average = False
    distances = distance_metric(emb_a, emb_b)
    losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2))
    return losses.mean() if size_average else losses.sum()

def ContrastiveLoss2(emb_a, emb_b, labels, margin=0.5, distance_metric=COSINE_DISTANCE):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    """
    distances = distance_metric(emb_a, emb_b)
    losses = labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(margin - distances).pow(2)
    return losses.sum()

import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedHardGISTLoss_Final(nn.Module):
    def __init__(self,
                 margin=0.5,
                 mining_alpha=0.2,  # 新增：挖掘缓冲区，建议 0.1~0.2
                 use_text_guide=True,
                 use_visual_guide=True,
                 false_neg_ratio=0.05, 
                 dirty_pos_threshold=0.45,
                 device=None):
        super().__init__()
        self.margin = margin
        self.mining_alpha = mining_alpha
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device
        self.false_neg_ratio = false_neg_ratio

        # 数据容器 (由 set_guide_data 注入)
        self.guide_q_emb = None
        self.q_id_to_dense = None
        self.guide_pos_map = None

    def set_guide_data(self, guide_q_emb, q_id_to_dense, guide_pos_map):
        """
        注入离线计算好的 Guide Embeddings。
        """
        self.guide_q_emb = guide_q_emb
        self.q_id_to_dense = q_id_to_dense
        self.guide_pos_map = guide_pos_map

    def _lookup_text_guide(self, batch):
        """
        严格模式 ID 查找，找不到直接报错
        """
        real_q_ids = batch['query_id'].tolist()
        
        dense_ids = []
        for rid in real_q_ids:
            # 尝试多种 Key 类型，确保兼容性
            if rid in self.q_id_to_dense:
                dense_ids.append(self.q_id_to_dense[rid])
            elif str(rid) in self.q_id_to_dense:
                dense_ids.append(self.q_id_to_dense[str(rid)])
            elif int(rid) in self.q_id_to_dense:
                dense_ids.append(self.q_id_to_dense[int(rid)])
            else:
                # 遇到数据问题直接抛出异常，不再静默失败
                raise KeyError(f"Fatal Error: Query ID '{rid}' (Type: {type(rid)}) not found in `q_id_to_dense` map. "
                               f"Please check your offline guide data processing.")

        dense_ids = torch.tensor(dense_ids, device=self.device)
        g_q = self.guide_q_emb[dense_ids].float()

        p_emb = []
        for p_id, c_id in batch['pos_id_pair']:
            key_int = (int(p_id), int(c_id))
            
            if self.guide_pos_map is not None and key_int in self.guide_pos_map:
                p_emb.append(self.guide_pos_map[key_int].float())
            else:
                raise KeyError(f"Fatal Error: Position Pair {key_int} not found in `guide_pos_map`. "
                               f"Please check your offline guide data processing.")
        
        g_p = torch.stack(p_emb)
        return g_q, g_p

    def forward(self, student_q_emb, student_p_emb, batch):
        labels = batch['label']
        
        # =====================================================================
        # 1. 计算 Student 距离 (梯度源头)
        # =====================================================================
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_p_emb)

        # 【核心修复】构建带有梯度的 Dummy Zero
        # 即使后续 Loss 全为 0，这个变量也能保证 student_dist 参与计算图，防止 DDP 崩溃
        dummy_zero = student_dist.sum() * 0.0

        # =====================================================================
        # 2. 计算 Guide 距离 (No Grad)
        # =====================================================================
        d_text = None
        if self.use_text and self.guide_q_emb is not None:
            with torch.no_grad():
                g_q, g_p = self._lookup_text_guide(batch)
                d_text = 1 - F.cosine_similarity(g_q, g_p)
                # 安全检查：防止 Guide 输出全一样
                if d_text.shape[0] > 1 and d_text.std() < 1e-6:
                     print("Warning: Text Guide distances variance is 0. Possible mapping error.")

        d_visual = None
        if self.use_visual and 'vis_q' in batch:
            with torch.no_grad():
                d_qp = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_p'])
                d_qc = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_c'])
                d_qs = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_s'])
                d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # =====================================================================
        # 3. GIST Mask 生成逻辑
        # =====================================================================
        def get_thresholds(dist, current_labels, base_thresh):
            temp_dirty = (current_labels == 1) & (dist > base_thresh)
            valid_pos = dist[(current_labels == 1) & (~temp_dirty)]
            if len(valid_pos) > 0:
                pos_mean = valid_pos.mean()
                # 这样能保留更多“长得有一点像但其实是负例”的 Hard Negatives
                strict_mean = pos_mean * 0.8 
                dyn_thresh = torch.min(strict_mean, torch.tensor(base_thresh, device=self.device))
            else:
                dyn_thresh = torch.tensor(base_thresh, device=self.device)
            return dyn_thresh

        # 默认为 False
        dirty_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        
        monitor = {
            "total_pos": (labels == 1).sum().item(),
            "total_neg": (labels == 0).sum().item(),
            "drop_pos_count": 0,    # 多少正例被当成脏数据扔了
            "drop_neg_count": 0,    # 多少负例被当成假负例扔了
            "drop_neg_text": 0,     # 文本 Guide 认为是假负例的数量
            "drop_neg_vis": 0,      # 视觉 Guide 认为是假负例的数量
        }

        if d_text is not None and d_visual is not None:
            t_thr = get_thresholds(d_text, labels, self.dirty_thresh)
            v_thr = get_thresholds(d_visual, labels, self.dirty_thresh)
            dirty_mask = (labels == 1) & (d_text > self.dirty_thresh) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & ((d_text < t_thr) | (d_visual < v_thr))
            
            monitor["drop_neg_text"] = mask_text_fn.sum().item()
            monitor["drop_neg_vis"] = mask_vis_fn.sum().item()
        elif d_text is not None:
            t_thr = get_thresholds(d_text, labels, self.dirty_thresh)
            dirty_mask = (labels == 1) & (d_text > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_text < t_thr)
            monitor["drop_neg_text"] = false_neg_mask.sum().item()
        elif d_visual is not None:
            v_thr = get_thresholds(d_visual, labels, self.dirty_thresh)
            dirty_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_visual < v_thr)
            monitor["drop_neg_vis"] = false_neg_mask.sum().item()
            
        monitor["drop_pos_count"] = dirty_mask.sum().item()
        monitor["drop_neg_count"] = false_neg_mask.sum().item()
        
        # 计算比例 (避免除以0)
        monitor["pos_drop_rate"] = monitor["drop_pos_count"] / max(monitor["total_pos"], 1)
        monitor["neg_drop_rate"] = monitor["drop_neg_count"] / max(monitor["total_neg"], 1)

        
        final_fn_thresh = torch.min(t_thr, torch.tensor(self.false_neg_ratio, device=self.device))
        
        false_neg_mask = (labels == 0) & (d_text < final_fn_thresh)
        
        # =====================================================================
        # 4. 构建 Clean Pool (解耦挖掘)
        # =====================================================================
        # 仅保留 Guide 认为“干净”的数据参与阈值计算和挖掘
        valid_pos_mask = (labels == 1) & (~dirty_mask)
        valid_neg_mask = (labels == 0) & (~false_neg_mask)
        
        valid_pos_dists = student_dist[valid_pos_mask]
        valid_neg_dists = student_dist[valid_neg_mask]

        # 【空集保护】如果全被过滤了，返回 dummy_zero
        if len(valid_pos_dists) == 0 or len(valid_neg_dists) == 0:
            return dummy_zero, monitor

        # =====================================================================
        # 5. Buffered Hard Mining (带缓冲的在线挖掘)
        # =====================================================================
        # 确定边界：使用 detach 截断梯度，防止逻辑混淆
        # 难负例必须比 valid_pos_dists.max() 还要近
        # 难正例必须比 valid_neg_dists.min() 还要远
        base_neg_min = valid_neg_dists.min().detach()
        base_pos_max = valid_pos_dists.max().detach()

        # 引入 Alpha 缓冲区：不仅要分开，还要分开一段距离
        # Hard Positive: dist > (Negative Boundary - alpha)
        hard_pos_threshold = base_neg_min - self.mining_alpha
        
        # Hard Negative: dist < (Positive Boundary + alpha)
        hard_neg_threshold = base_pos_max + self.mining_alpha

        # 筛选
        hard_pos = valid_pos_dists[valid_pos_dists > hard_pos_threshold]
        
        # 加 1e-6 确保数值比较稳定性
        hard_neg = valid_neg_dists[valid_neg_dists < hard_neg_threshold + 1e-6]
        
        # 记录参与最终计算的样本数
        monitor["active_hard_pos"] = len(hard_pos)
        monitor["active_hard_neg"] = len(hard_neg)

        # =====================================================================
        # 6. Loss 计算
        # =====================================================================
        # 使用 dummy_zero 替代 0.0，确保梯度链条完整
        pos_loss = hard_pos.pow(2).sum() if len(hard_pos) > 0 else dummy_zero
        neg_loss = F.relu(self.margin - hard_neg).pow(2).sum() if len(hard_neg) > 0 else dummy_zero
        
        # 返回总 loss
        return pos_loss + neg_loss, monitor




class FusedGISTContrastiveLoss(nn.Module):
    def __init__(self, 
                 margin=0.5, 
                 use_text_guide=True, 
                 use_visual_guide=True, 
                 dirty_pos_threshold=0.45,
                 device=None):
        super().__init__()
        self.margin = margin
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device
        
        # --- 缓存容器 ---
        self.guide_q_emb = None      # Tensor [N_nodes, D]
        self.q_id_to_dense = None    # Dict: real_id -> dense_idx
        self.guide_pos_map = None    # Dict: (p, c) -> Tensor
        self.default_pos_emb = None  # Fallback

    def set_guide_data(self, guide_q_emb, q_id_to_dense, guide_pos_map):
        """
        由 Trainer 在预计算完成后调用，注入数据
        """
        self.guide_q_emb = guide_q_emb
        self.q_id_to_dense = q_id_to_dense
        self.guide_pos_map = guide_pos_map
        
        # 创建默认 embedding (全0) 用于处理缺失键
        if self.guide_q_emb is not None:
            dim = self.guide_q_emb.shape[1]
            self.default_pos_emb = torch.zeros(dim, device=self.device, dtype=torch.float16)

    def _lookup_text_guide(self, batch):
        """内部查表函数"""
        # A. Query Lookup
        real_q_ids = batch['query_id'].cpu().tolist()
        dense_indices = [self.q_id_to_dense[rid] for rid in real_q_ids]
        dense_indices_tensor = torch.tensor(dense_indices, device=self.device, dtype=torch.long)
        
        g_q = self.guide_q_emb[dense_indices_tensor].float() # FP16 -> FP32
        
        # B. Position Lookup
        pos_pairs = batch['pos_id_pair'] # List of tuples
        p_emb_list = []
        
        for p_id, c_id in pos_pairs:
            key = (int(p_id), int(c_id))
            emb = self.guide_pos_map.get(key, self.default_pos_emb)
            p_emb_list.append(emb)
            
        g_p = torch.stack(p_emb_list).float()
        
        return g_q, g_p

    def forward(self, student_q_emb, student_c_emb, batch):
        """
        Args:
            student_q_emb: [B, D]
            student_c_emb: [B, D]
            batch: 包含 labels, query_id, pos_id_pair, vis_* 的字典
        """
        labels = batch['label']
        
        # 1. Student Distance
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_c_emb)
        
        # 2. Guide Distances
        d_text = None
        if self.use_text and self.guide_q_emb is not None:
            g_q, g_p = self._lookup_text_guide(batch)
            d_text = 1 - F.cosine_similarity(g_q, g_p)
            
        d_visual = None
        if self.use_visual:
            vis_q, vis_p = batch['vis_q'], batch['vis_p']
            vis_c, vis_s = batch['vis_c'], batch['vis_s']
            
            d_qp = 1 - F.cosine_similarity(vis_q, vis_p)
            d_qc = 1 - F.cosine_similarity(vis_q, vis_c)
            d_qs = 1 - F.cosine_similarity(vis_q, vis_s)
            d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # 3. Mask Logic (复用之前逻辑)
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        def get_dynamic_thresh(dist_vec):
            valid_pos = dist_vec[labels == 1]
            if len(valid_pos) > 0:
                return torch.min(torch.tensor(self.dirty_thresh, device=self.device), valid_pos.mean())
            return torch.tensor(self.dirty_thresh, device=self.device)

        if self.use_text and self.use_visual:
            # Fused
            thresh_t = get_dynamic_thresh(d_text)
            thresh_v = get_dynamic_thresh(d_visual)
            
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & ((d_text < thresh_t) | (d_visual < thresh_v))
            
        elif self.use_text:
            thresh_t = get_dynamic_thresh(d_text)
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_text < thresh_t)
            
        elif self.use_visual:
            thresh_v = get_dynamic_thresh(d_visual)
            dirty_pos_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_visual < thresh_v)

        # 4. Loss Weighting
        weights = torch.ones_like(student_dist)
        weights[dirty_pos_mask] = 0.0
        weights[false_neg_mask] = 0.0
        
        raw_loss = labels * student_dist.pow(2) + (1 - labels) * F.relu(self.margin - student_dist).pow(2)
        
        return (raw_loss * weights).sum() / (weights.sum() + 1e-9)


class FusedGISTContrastiveLoss_1(nn.Module):
    def __init__(self, 
                 margin=0.5, 
                 guide_model_name="Qwen/Qwen3-Embedding-0.6B",
                 use_text_guide=True, 
                 use_visual_guide=True,
                 dirty_pos_threshold=0.45,
                 device=None):
        super().__init__()
        self.margin = margin
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device
        
        # --- Initialize Text Guide Model ---
        if self.use_text:
            print(f"Loading Guide Model: {guide_model_name}...")
            
            # 配置 Flash Attention 2
            model_kwargs = {
                "attn_implementation": "flash_attention_2", 
                "torch_dtype": torch.float16  # 建议配合 fp16 使用
            }
            
            try:
                self.guide_model = SentenceTransformer(
                    guide_model_name,
                    model_kwargs=model_kwargs,
                    trust_remote_code=True
                )
            except Exception as e:
                print(f"Warning: Failed to load with Flash Attention 2 ({e}). Fallback to default.")
                self.guide_model = SentenceTransformer(guide_model_name, trust_remote_code=True)
            
            # 移至设备并冻结
            if self.device:
                self.guide_model.to(self.device)
            
            self.guide_model.eval()
            for param in self.guide_model.parameters():
                param.requires_grad = False
        else:
            self.guide_model = None

    def forward(self, 
                student_q_emb, student_c_emb, labels, 
                q_text_raw_list=None, pos_text_raw_list=None, # Raw Text for Guide
                vis_q=None, vis_p=None, vis_c=None, vis_s=None): # Raw Visual for Guide
        
        """
        Args:
            student_q/c_emb: [B, D]
            labels: [B]
            q_text_raw_list: List[str] of length B
            pos_text_raw_list: List[str] of length B
            vis_*: [B, D_vis] tensors
        """
        # 1. 计算 Student 基础距离
        # distance = 1 - sim
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_c_emb)
        
        # 2. 计算 Guide Distances
        d_text = None
        d_visual = None
        
        with torch.no_grad():
            # --- Text Guide ---
            if self.use_text and self.guide_model:
                # Qwen Embedding 对 Query 需要 prompt_name="query"
                # 对 Candidate (Position) 不需要 prompt (或默认)
                
                # Encode returns numpy or tensor, convert to tensor
                # convert_to_tensor=True 直接返回 GPU tensor
                emb_text_q = self.guide_model.encode(
                    q_text_raw_list, 
                    prompt_name="query", 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                )
                emb_text_c = self.guide_model.encode(
                    pos_text_raw_list, 
                    convert_to_tensor=True, 
                    show_progress_bar=False
                )
                
                # SentenceTransformer 默认已经 normalize 了，但保险起见
                # cosine_similarity handle batch logic automatically
                d_text = 1 - F.cosine_similarity(emb_text_q, emb_text_c) # [B]

            # --- Visual Guide ---
            if self.use_visual:
                # 只要 Q 跟 P, C, S 任意一个像，就算 Visual Sim 高 (Dist 低)
                d_qp = 1 - F.cosine_similarity(vis_q, vis_p)
                d_qc = 1 - F.cosine_similarity(vis_q, vis_c)
                d_qs = 1 - F.cosine_similarity(vis_q, vis_s)
                
                d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # 3. 生成 Mask (Fused Strategy)
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # 动态阈值计算
        def get_dynamic_thresh(dist_vec):
            # 取当前Batch中认为是正例的样本的平均距离
            valid_pos_dists = dist_vec[labels == 1]
            if len(valid_pos_dists) > 0:
                return min(valid_pos_dists.mean(), torch.tensor(self.dirty_thresh, device=self.device))
            return torch.tensor(self.dirty_thresh, device=self.device)

        if self.use_text and self.use_visual:
            # AND for Dirty Pos, OR for False Neg
            is_dirty_text = d_text > self.dirty_thresh
            is_dirty_vis = d_visual > self.dirty_thresh
            dirty_pos_mask = (labels == 1) & is_dirty_text & is_dirty_vis
            
            thresh_text = get_dynamic_thresh(d_text)
            thresh_vis = get_dynamic_thresh(d_visual)
            
            is_sim_text = d_text < thresh_text
            is_sim_vis = d_visual < thresh_vis
            false_neg_mask = (labels == 0) & (is_sim_text | is_sim_vis)
            
        elif self.use_text:
            # Only Text
            thresh_text = get_dynamic_thresh(d_text)
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_text < thresh_text)
            
        elif self.use_visual:
            # Only Visual
            thresh_vis = get_dynamic_thresh(d_visual)
            dirty_pos_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_visual < thresh_vis)

        # 4. 计算最终 Loss
        # 生成权重: 正常=1, 过滤=0
        weights = torch.ones_like(student_dist)
        weights[dirty_pos_mask] = 0.0
        weights[false_neg_mask] = 0.0
        
        # 基础对比损失
        pos_loss = labels * student_dist.pow(2)
        neg_loss = (1 - labels) * F.relu(self.margin - student_dist).pow(2)
        
        weighted_loss = (pos_loss + neg_loss) * weights
        final_loss = weighted_loss.sum() / (weights.sum() + 1e-9)
        
        return final_loss