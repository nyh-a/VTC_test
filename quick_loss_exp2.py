import os
import logging
from tqdm.auto import tqdm
from datasets import Dataset  # ★★★ 我们现在需要这个库 ★★★
import csv
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

# ★★★ 关键导入：现代训练器、参数和模型类 ★★★
from sentence_transformers import SentenceTransformer, InputExample, losses, models, util
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers import SentenceTransformerTrainer

# 确保你的工具函数可用
from data_loader.data_loaders import DataLoader_Base
from utils.util import calc_metrics
from enum import Enum


class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

os.environ["HF_SAFE_WEIGHTS_SAVE"] = "0"
# os.environ["WANDB_DISABLED"] = "true"

# --- 1. 配置实验 ---
# ==========================================
# MODEL_NAME = '/home/u2120230636/work/huggingface_models/bert_base_uncased'
STUDENT_MODEL_NAME = "/home/u2120230631/codes/VTC/all-mpnet-base-v2"
# STUDENT_MODEL_NAME = "/home/u2120230631/codes/VTC/bert-base-uncased"
GUIDE_MODEL_NAME = "/home/u2120230631/codes/VTC/Qwen3-Embedding-0.6B"
BATCH_SIZE = 128
# 设置一个相对充足的 Epoch 数量，让模型有机会过拟合
# 我们将依赖验证集来找到最佳点
NUM_EPOCHS = 10
EXPERIMENT_NAME = 'GISTEmbed_Comparison_v2' # 给这次实验起个名字，用于保存模型

# --- ★★★ 切换损失函数 ★★★ ---
# 可选值: "MNRL", "CachedMNRL", "SymmetricMNRL", "OnlineContrastiveLoss" 
# "GISTEmbedLoss", "CachedGISTEmbedLoss", "GISTOnlineContrastiveLoss"
LOSS_CHOICE = "GISTOnlineContrastiveLoss2"

# --- Loss-specific parameters ---
CONTRASTIVE_MARGIN = 0.5 # Margin for OnlineContrastiveLoss
DIRTY_POSITIVE_THRESHOLD = 0.45 # For GISTOnlineContrastiveLoss2
# For CachedGISTEmbedLoss
GIST_MARGIN_STRATEGY = "absolute" # or "relative"
GIST_MARGIN = 0.1
# ==========================================

MODEL_SAVE_PATH = f'output/{EXPERIMENT_NAME}_{LOSS_CHOICE}_{GUIDE_MODEL_NAME.split("/")[-1]}_to_{STUDENT_MODEL_NAME.split("/")[-1]}'

NAME = "food"
if NAME == "food":
    DATA_PATH = "data/SemEval-Food/semeval_food.pickle.bin"
    TAXONOMY_NAME = "semeval_food"
elif NAME == "verb":
    DATA_PATH = "data/SemEval-Verb/wordnet_verb.pickle.bin"
    TAXONOMY_NAME = "wordnet_verb"
elif NAME == "mesh":
    DATA_PATH = "data/mesh/mesh.pickle.bin"
    TAXONOMY_NAME = "mesh"
else:
    raise FileNotFoundError("No such dataset " + NAME)
# ==========================================



# ★★★ 数据准备函数，现在输出 Hugging Face `datasets.Dataset` 对象 ★★★
def prepare_contrastive_hf_dataset(
    data_provider, 
    node_list, 
    negatives_per_positive=31, 
    use_prompts=False, 
    query_instruction="", 
    corpus_instruction=""
):
    """
    Prepares data in the (anchor, candidate, label) format for OnlineContrastiveLoss
    and wraps it in a Hugging Face `datasets.Dataset` object.

    Args:
        data_provider: Your MAGDataset object that provides access to all data.
        node_list: The list of query nodes to process (e.g., train_node_list).
        negatives_per_positive (int): The number of negative samples to generate for each positive sample.
        use_prompts (bool): If True, will not use the descriptive templates.
        query_instruction (str): The instruction prompt for queries. (Used by the Trainer, not here)
        corpus_instruction (str): The instruction prompt for corpus items. (Used by the Trainer, not here)
    """
    logging.info("Preparing Hugging Face dataset for OnlineContrastiveLoss...")
    
    # This dictionary will be used to build the Dataset object
    data = {"texts_a": [], "texts_b": [], "label": []}
    
    # Direct access to data provider attributes
    node2pos = data_provider.node2pos
    id2desc = data_provider.id2desc
    taxon2id = data_provider.taxon2id
    all_edges = data_provider.all_edges

    # Define templates conditionally based on whether prompts are used
    if use_prompts:
        # For INSTRUCTOR models, we feed clean text. The instruction is handled by the Trainer.
        template_q = "{}"
        template_p = "{} ; {} ; {}" # Use a clear separator
    else:
        # For standard models, we embed the instruction/role in the template itself
        template_q = "Query Node: Definition: \"{}\""
        template_p = "Parent Node: Definition: \"{}\"; Child Node: Definition: \"{}\"; Sibling Node: Definition: \"{}\";"

    for query_node in tqdm(node_list, desc="Processing contrastive pairs"):
        if query_node not in node2pos or not node2pos[query_node]:
            continue
        
        try:
            query_text = template_q.format(id2desc[taxon2id[query_node]])
            if not query_text.strip().strip('"'): continue # Skip if definition is empty
        except KeyError:
            continue

        positive_positions = node2pos[query_node]
        
        # Get negatives for this query once
        num_negatives_to_sample = negatives_per_positive * len(positive_positions)
        negative_positions = data_provider._get_k_negatives(query_node, num_negatives_to_sample)
        neg_idx = 0

        for p_node, c_node in positive_positions:
            try:
                # --- Process 1 Positive Example ---
                s_node = data_provider._get_sibling(p_node, c_node)
                p_desc = id2desc[taxon2id[p_node]]
                c_desc = id2desc[taxon2id[c_node]]
                s_desc = id2desc[taxon2id[s_node]]
                positive_text = template_p.format(p_desc, c_desc, s_desc)
                
                # Add the positive pair to our dictionary
                data["texts_a"].append(query_text)
                data["texts_b"].append(positive_text)
                data["label"].append(1)
                
                # --- Process N Negative Examples ---
                for _ in range(negatives_per_positive):
                    if neg_idx >= len(negative_positions):
                        break # Stop if we run out of sampled negatives
                    
                    neg_p_node, neg_c_node = negative_positions[neg_idx]
                    neg_idx += 1
                    
                    neg_s_node = data_provider._get_sibling(neg_p_node, neg_c_node)
                    neg_p_desc = id2desc[taxon2id[neg_p_node]]
                    neg_c_desc = id2desc[taxon2id[neg_c_node]]
                    neg_s_desc = id2desc[taxon2id[neg_s_node]]
                    negative_text = template_p.format(neg_p_desc, neg_c_desc, neg_s_desc)

                    # Add the negative pair to our dictionary
                    data["texts_a"].append(query_text)
                    data["texts_b"].append(negative_text)
                    data["label"].append(0)

            except KeyError:
                # This can happen if a node in a pair is missing from the taxon2id map
                continue
                
    logging.info(f"Created a dataset with {len(data['texts_a'])} total pairs.")
    # Convert the dictionary of lists into a Hugging Face Dataset object
    return Dataset.from_dict(data)



def evaluate_model(model, dataset_provider):
    """
    使用你的评估逻辑来测试模型性能。
    """
    print("\n--- Starting Evaluation ---")
    
    # 1. 准备数据
    queries = dataset_provider.test_node_list
    node2pos = dataset_provider.test_node2pos
    candidate_positions_nodes = dataset_provider.all_edges
    id2desc = dataset_provider.id2desc
    taxon2id = dataset_provider.taxon2id

    # 定义文本模板
    template_q = "Query Node: Definition: \"{}\""
    template_p = "Parent Node: Definition: \"{}\"; Child Node: Definition: \"{}\"; Sibling Node: Definition: \"{}\";"
    
    # 2. 计算所有 candidate positions 的 embedding
    print("Encoding candidate positions (corpus)...")
    corpus = []
    for p, c in tqdm(candidate_positions_nodes, desc="Formatting Corpus"):
        s = dataset_provider._get_sibling(p, c)
        try:
            p_desc = id2desc[taxon2id[p]]
            c_desc = id2desc[taxon2id[c]]
            s_desc = id2desc[taxon2id[s]]
            corpus.append(template_p.format(p_desc, c_desc, s_desc))
        except KeyError:
            # 如果某个节点信息缺失，用空字符串代替，保持索引对齐
            corpus.append("") 
            
    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, show_progress_bar=True)

    # 3. 计算所有 queries 的 embedding
    print("Encoding queries...")
    queries_text = [template_q.format(id2desc[taxon2id[q]]) for q in queries]
    query_embeddings = model.encode(queries_text, convert_to_tensor=True, show_progress_bar=True)

    # 4. 执行语义搜索
    print("Performing semantic search...")
    # 使用 dot_score 是推荐的，因为它与 MultipleNegativesRankingLoss 的内部计算一致
    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=len(candidate_positions_nodes))
    
    # 5. 计算指标
    print("Calculating metrics...")
    
    # 创建从 node pair 到语料库索引的映射
    node_pair_to_corpus_idx = {pair: i for i, pair in enumerate(candidate_positions_nodes)}
    
    leaf_queries = set()
    for query in queries:
        poses = node2pos[query]
        if poses and all(pos[1] == dataset_provider.pseudo_leaf_node for pos in poses):
            leaf_queries.add(query)

    all_ranks, leaf_ranks, nonleaf_ranks = [], [], []
    
    for i, hit_list in enumerate(hits):
        query = queries[i]
        true_pos_pairs = node2pos[query]
        if not true_pos_pairs:
            continue
            
        # 获取真实正例在语料库中的索引
        true_pos_indices = {node_pair_to_corpus_idx.get(pos) for pos in true_pos_pairs}
        # 过滤掉可能因KeyError而找不到的索引
        true_pos_indices = {idx for idx in true_pos_indices if idx is not None}
        
        if not true_pos_indices:
            continue

        # 获取预测的排名列表 (corpus_id 列表)
        pred_rank_indices = [hit['corpus_id'] for hit in hit_list]
        
        # 找到每个真实正例的排名
        ranks_for_this_query = []
        for pos_idx in true_pos_indices:
            try:
                rank = pred_rank_indices.index(pos_idx) + 1
                ranks_for_this_query.append(rank)
            except ValueError:
                # 如果一个正例不在预测列表中，说明搜索出了问题，或者语料库不匹配
                # 理论上不应发生，但为了鲁棒性，可以赋一个最大排名
                ranks_for_this_query.append(len(candidate_positions_nodes))

        if ranks_for_this_query:
            all_ranks.append(ranks_for_this_query)
            if query in leaf_queries:
                leaf_ranks.append(ranks_for_this_query)
            else:
                nonleaf_ranks.append(ranks_for_this_query)

    print("--- Evaluation Results ---")
    calc_metrics(all_ranks, leaf_ranks, nonleaf_ranks)
    print("--- End of Evaluation ---")


class RankingEvaluator(SentenceEvaluator):
    """
    Evaluates model based on a comprehensive set of ranking metrics (MRR, Hit@K, Recall@K, etc.).
    It distinguishes between leaf and non-leaf queries and logs results to a CSV file.
    """
    def __init__(self, queries: dict[str, str], corpus: dict[str, str], 
                 positive_relations: dict[str, set[str]], leaf_query_ids: set[str],
                 name: str = '', batch_size: int = 32, show_progress_bar: bool = False):
        super().__init__()

        self.primary_metric = "total_mrr_scaled_10"
        # `greater_is_better` 告诉 fit() 函数，这个指标是越高越好
        self.greater_is_better = True
        
        self.queries = queries
        self.corpus = corpus
        self.positive_relations = positive_relations
        self.leaf_query_ids = leaf_query_ids
        self.name = name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        self.corpus_ids = list(self.corpus.keys())
        self.query_ids = list(self.queries.keys())

    def _compute_metrics(self, ranks_list: list[list[int]], total_positives: int) -> dict[str, float]:
        """
        Computes a dictionary of metrics from a list of ranks.
        """
        if not ranks_list or not any(ranks_list):
            return defaultdict(float)

        # Helper to calculate Hit@K and Recall@K
        def hits_and_recall_at_k(k):
            # Hits: How many queries have at least one positive in the top K
            hits_count = sum(1 for ranks in ranks_list if min(ranks) <= k)
            # Recall: How many positive items (in total) were found in the top K
            recall_count = sum(sum(1 for r in ranks if r <= k) for ranks in ranks_list)
            
            return hits_count / len(ranks_list), recall_count / total_positives

        # --- Standard Metrics (based on the first hit for each query) ---
        first_hit_ranks = [min(ranks) for ranks in ranks_list if ranks]
        
        mrr_standard = np.mean([1.0 / r for r in first_hit_ranks]) if first_hit_ranks else 0
        mr_first_hit = np.mean(first_hit_ranks) if first_hit_ranks else 0

        # --- User's Custom Metrics (based on all hits) ---
        all_ranks_flat = [r for ranks in ranks_list for r in ranks]
        
        if all_ranks_flat:
            rank_positions = np.array(all_ranks_flat)
            mrr_all_hits = (1.0 / rank_positions).mean()
            scaled_rank_positions = np.ceil(rank_positions / 10)
            mrr_scaled_10 = (1.0 / scaled_rank_positions).mean()
        else:
            mrr_all_hits, mrr_scaled_10 = 0, 0

        # --- Initialize the metrics dictionary with all MRR variants ---
        metrics = {
            "mrr_standard": mrr_standard,
            "mr_first_hit": mr_first_hit,
            "mrr_all_hits": mrr_all_hits,
            "mrr_scaled_10": mrr_scaled_10,
        }
        
        for k in [1, 5, 10, 50, 100]:
            hits_k, recall_k = hits_and_recall_at_k(k)
            metrics[f"hit_at_{k}"] = hits_k
            metrics[f"recall_at_{k}"] = recall_k
            # Precision@K: Averaged over all queries
            precision_k = np.mean([len([r for r in ranks if r <= k]) / k for ranks in ranks_list])
            if k <= 10:
                metrics[f"precision_at_{k}"] = precision_k

        return metrics

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logging.info(f"ComprehensiveRankingEvaluator: Evaluating model on {self.name} {out_txt}")
        
        # 1. Encode corpus and queries
        corpus_embeddings = model.encode(list(self.corpus.values()), convert_to_tensor=True, 
                                         show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)
        query_embeddings = model.encode(list(self.queries.values()), convert_to_tensor=True, 
                                        show_progress_bar=self.show_progress_bar, batch_size=self.batch_size)

        # 2. Perform search
        hits = util.semantic_search(query_embeddings, corpus_embeddings, 
                                    score_function=util.dot_score, top_k=len(self.corpus))

        # 3. Collect ranks for all, leaf, and non-leaf queries
        all_ranks, leaf_ranks, nonleaf_ranks = [], [], []
        
        for i, query_id in enumerate(self.query_ids):
            true_positives = self.positive_relations.get(query_id, set())
            if not true_positives:
                continue

            hit_list = hits[i]
            # `h['corpus_id']` 是整数索引
            # `self.corpus_ids` 是一个按顺序保存了所有字符串ID的列表
            pred_rank_ids = [self.corpus_ids[h['corpus_id']] for h in hit_list]

            # pred_rank_ids = [h['corpus_id'] for h in hits[i]]
            ranks_for_this_query = []
            for pos_id in true_positives:
                try:
                    rank = pred_rank_ids.index(pos_id) + 1
                    ranks_for_this_query.append(rank)
                except ValueError:
                    # Positive not found in predictions, assign max rank
                    ranks_for_this_query.append(len(self.corpus) + 1)
            
            if ranks_for_this_query:
                all_ranks.append(ranks_for_this_query)
                if query_id in self.leaf_query_ids:
                    leaf_ranks.append(ranks_for_this_query)
                else:
                    nonleaf_ranks.append(ranks_for_this_query)

        # 4. Calculate metrics for each group
        total_pos_all = sum(len(self.positive_relations.get(qid, set())) for qid in self.query_ids)
        total_pos_leaf = sum(len(self.positive_relations.get(qid, set())) for qid in self.leaf_query_ids)
        total_pos_nonleaf = total_pos_all - total_pos_leaf
        
        total_metrics = self._compute_metrics(all_ranks, total_pos_all)
        leaf_metrics = self._compute_metrics(leaf_ranks, total_pos_leaf)
        nonleaf_metrics = self._compute_metrics(nonleaf_ranks, total_pos_nonleaf)
    
        # Define the desired order for presentation
        metric_order = [
            "mrr_standard", "mr_first_hit", "mrr_all_hits", "mrr_scaled_10",
            "hit_at_1", "hit_at_5", "hit_at_10", "hit_at_50", "hit_at_100",
            "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_50", "recall_at_100",
            "precision_at_1", "precision_at_5", "precision_at_10"
        ]

        logging.info("\n--- Evaluation Results ---")
        
        # --- Print Total Metrics in Order ---
        logging.info("--- Total Metrics ---")
        for key in metric_order:
            if key in total_metrics:
                logging.info(f"    {key:<20}: {total_metrics[key]:.4f}")
        
        # --- Print Leaf Metrics in Order ---
        logging.info("--- Leaf Metrics ---")
        for key in metric_order:
            if key in leaf_metrics:
                logging.info(f"    {key:<20}: {leaf_metrics[key]:.4f}")

        # --- Print Non-Leaf Metrics in Order ---
        logging.info("--- Non-Leaf Metrics ---")
        for key in metric_order:
            if key in nonleaf_metrics:
                logging.info(f"    {key:<20}: {nonleaf_metrics[key]:.4f}")
        logging.info("--------------------------")
        
        # --- Build the flat dictionary for CSV in the same desired order ---
        all_metrics_dict = {}
        
        # Add total, then leaf, then non-leaf metrics, each block following the metric_order
        for prefix, metrics_dict in [("total", total_metrics), ("leaf", leaf_metrics), ("nonleaf", nonleaf_metrics)]:
            for key in metric_order:
                if key in metrics_dict:
                    all_metrics_dict[f"{prefix}_{key}"] = metrics_dict[key]
        
        # Write to CSV
        if output_path is not None:
            csv_path = os.path.join(output_path, "evaluation_results.csv")
            self._write_to_csv(csv_path, all_metrics_dict)
        
        # Return the flat dictionary for `fit()` to use
        return all_metrics_dict


    def _write_to_csv(self, path: str, metrics_dict: dict):
        """
        Writes a single row of metrics to a CSV file.
        The header is written only if the file does not exist.
        """
        file_exists = os.path.isfile(path)
        
        with open(path, mode="a", newline="", encoding="utf-8") as f:
            # The fieldnames are derived directly from the ordered dictionary keys
            writer = csv.DictWriter(f, fieldnames=metrics_dict.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics_dict)



# ★★★ 我们自己为 GIST+OnlineContrastive 编写的损失类 ★★★
class GISTOnlineContrastiveLoss(nn.Module):
    """
    Combines GISTEmbed's guided negative filtering with OnlineContrastiveLoss's
    hard negative mining.
    """
    def __init__(self, model, guide_model_name: str, 
                 margin: float = 0.5, 
                 distance_metric = SiameseDistanceMetric.COSINE_DISTANCE):
        super(GISTOnlineContrastiveLoss, self).__init__()
        self.student_model = model
        self.margin = margin
        self.distance_metric = distance_metric
        
        print(f"Initializing GISTOnlineContrastiveLoss with guide model: {guide_model_name}")
        # ★★★ 1. 加载指导模型 ★★★
        # 我们假设指导模型和学生模型在同一个设备上
        self.guide_model = SentenceTransformer(guide_model_name)
        # 冻结指导模型的所有参数，并设为评估模式
        self.guide_model.eval()
        for param in self.guide_model.parameters():
            param.requires_grad = False
            
    def forward(self, sentence_features: list[dict], labels: torch.Tensor):
        # sentence_features: a list of {'input_ids': ..., 'attention_mask': ...}
        # It contains [anchor_1, positive_1, anchor_2, negative_2, ...]
        # For OnlineContrastiveLoss, sentence_features[0] are anchors, sentence_features[1] are candidates
        
        # ★★★ 2. 分别获取学生和专家的 Embeddings ★★★
        # `reps` 是一个包含两个张量的列表：[embeddings_a, embeddings_b]
        reps = [self.student_model(feat)['sentence_embedding'] for feat in sentence_features]
        
        with torch.no_grad():
            # 使用指导模型获取“专家”的意见
            guide_reps = [self.guide_model(feat)['sentence_embedding'] for feat in sentence_features]

        # ★★★ 3. 使用专家意见，创建 GISTEmbed 的“屏蔽”矩阵 (Mask) ★★★
        # a. 计算专家的距离矩阵
        guide_distance_matrix = self.distance_metric(guide_reps[0], guide_reps[1])
        
        # b. 找到每个正样本对的“黄金标准”距离
        # labels.diag() == 1 会得到一个布尔掩码，标记出正样本对
        # 注意：在 OnlineContrastiveLoss 的标准输入中，labels 是一个 0/1 矩阵，正例在对角线上
        # 我们需要根据实际的 labels 构造
        is_positive_pair = (labels == 1)
        # 获取所有正样本对的专家距离
        positive_guide_distances = guide_distance_matrix[is_positive_pair]

        # c. 创建屏蔽矩阵：如果一个负样本与 query 的专家距离 < 对应正样本的专家距离，就屏蔽它
        # 这里需要一些 broadcast 操作，我们用一个简化的、更鲁棒的阈值法
        # 使用正样本专家距离的平均值或中位数作为阈值
        if len(positive_guide_distances) > 0:
            # GISTEmbed 核心思想：如果一个负样本对的专家距离，比一个合理的正样本对的距离还小
            # （说明专家认为它们很相关），那它就是“假负例”，应该被屏蔽
            threshold = torch.mean(positive_guide_distances)
            
            # is_negative_pair 是 label==0 的地方
            is_negative_pair = (labels == 0)
            
            # guide_distance_matrix[is_negative_pair] 得到所有负样本对的专家距离
            # dangerous_negatives 是那些专家认为距离太近的负样本
            dangerous_negatives_mask = (guide_distance_matrix < threshold) & is_negative_pair
        else:
            dangerous_negatives_mask = torch.zeros_like(guide_distance_matrix, dtype=torch.bool)
            
        # ★★★ 4. 计算学生模型的距离矩阵，并应用屏蔽 ★★★
        student_distance_matrix = self.distance_metric(reps[0], reps[1])
        
        # 屏蔽掉“危险”的负样本，让它们在后续计算中不起作用
        # 我们可以给它们一个非常大的距离，这样它们就不会被选为“硬负例”
        student_distance_matrix[dangerous_negatives_mask] = self.margin + 1 # 确保它大于 margin

        # ★★★ 5. 在“干净”的距离矩阵上，执行 OnlineContrastiveLoss 的逻辑 ★★★
        negs = student_distance_matrix[labels == 0]
        poss = student_distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        
        loss = positive_loss + negative_loss
        return loss


class GISTOnlineContrastiveLoss2(nn.Module):
    def __init__(self, model, guide_model_name: str, 
                 margin: float = 0.5, 
                 dirty_positive_threshold: float = 0.45): # 新增：脏数据阈值
        super(GISTOnlineContrastiveLoss2, self).__init__()
        self.student_model = model
        self.margin = margin
        # Cosine distance: 0 (same) -> 2 (opposite)
        self.distance_metric = lambda x, y: 1 - F.cosine_similarity(x, y)
        self.dirty_positive_threshold = dirty_positive_threshold
        
        print(f"Loading Guide Model: {guide_model_name}...")
        self.guide_model = SentenceTransformer(guide_model_name)
        self.guide_model.eval()
        for param in self.guide_model.parameters():
            param.requires_grad = False
            
    def forward(self, sentence_features, labels: torch.Tensor):
        # 1. 计算 Student Embeddings
        reps = [self.student_model(feat)['sentence_embedding'] for feat in sentence_features]
        
        # 2. 计算 Guide Embeddings (No Grad)
        with torch.no_grad():
            guide_reps = [self.guide_model(feat)['sentence_embedding'] for feat in sentence_features]
            
        # 3. 计算距离矩阵
        student_distance = self.distance_metric(reps[0], reps[1])
        guide_distance = self.distance_metric(guide_reps[0], guide_reps[1])
        
        # 4. GISTEmbed 核心逻辑：生成 Mask
        
        # --- A. 清洗脏正例 (你的痛点) ---
        # 如果 Label是1，但专家认为距离很远(>0.45)，则视为脏数据
        # 策略：将其在 Student 矩阵中修改为 0 (距离为0)，这样它就不会被选为 Hard Positive
        dirty_pos_mask = (labels == 1) & (guide_distance > self.dirty_positive_threshold)
        
        # --- B. 清洗假负例 (GISTEmbed 原生逻辑) ---
        # 计算动态阈值 (参考正例的平均难度)
        # 注意：计算平均值时，要排除掉刚才发现的脏正例，否则阈值会被拉高
        valid_pos_guide_dist = guide_distance[(labels == 1) & (~dirty_pos_mask)]
        
        
        if len(valid_pos_guide_dist) > 0:
            # 动态阈值：比平均正例更像正例的负例，就是假负例
            dynamic_threshold = torch.mean(valid_pos_guide_dist)
        else:
            # Fallback: 如果没有有效正例，设为一个保守值
            dynamic_threshold = self.dirty_positive_threshold # Fallback to a reasonable value

        # 确保动态阈值不会因为异常值而变得过大
        # 我们不希望动态阈值比我们设定的“脏正例”阈值还宽松
        final_dynamic_threshold = torch.min(
            torch.as_tensor(dynamic_threshold), 
            torch.as_tensor(self.dirty_positive_threshold)
        )

        # 如果 Label是0，但专家认为距离很近 (<动态阈值)，则视为假负例
        # 策略：将其在 Student 矩阵中修改为 margin + 0.1 (距离很大)，使其在 Negative Loss 中为 0
        false_neg_mask = (labels == 0) & (guide_distance < final_dynamic_threshold)

        # 5. 应用 Mask 到 Student 距离矩阵
        # 这一步必须 clone，否则会修改反向传播图中的叶子节点导致报错
        modified_student_dist = student_distance.clone()
        
        # 处理脏正例：把距离设为 0，这样它既不会产生 Positive Loss (0^2=0)，
        # 也不会被选为 Hard Positive (因为它比任何正常正例的距离都小)
        modified_student_dist[dirty_pos_mask] = 0.0
        
        # 处理假负例：把距离设为 > margin，这样 ReLU(margin - dist) = 0，
        # 且在 Hard Negative Mining 时，它看起来距离很远，不会被选中
        modified_student_dist[false_neg_mask] = self.margin + 0.1

        # 6. Online Contrastive Loss 逻辑
        negs = modified_student_dist[labels == 0]
        poss = modified_student_dist[labels == 1]

        # Hard Mining
        # 同样需要注意：如果poss全被过滤成0了，或者negs全被过滤大了，需要处理空的情况
        if len(poss) == 0 or len(negs) == 0:
            return torch.tensor(0.0, device=reps[0].device, requires_grad=True)

        # 选取 Hard Positive: 距离 > 负例最小值 (或正例均值)
        # 注意：这里已经被我们把脏数据置为0了，所以脏数据绝对不会被选上
        hard_pos_limit = negs.min() if len(negs) > 0 else poss.mean()
        hard_positives = poss[poss > hard_pos_limit]

        # 选取 Hard Negative: 距离 < 正例最大值 (或负例均值)
        # 注意：假负例已经被我们置为 margin+0.1了，绝对不会小于正例最大值
        hard_neg_limit = poss.max() if len(poss) > 0 else negs.mean()
        hard_negatives = negs[negs < hard_neg_limit + 1e-6]

        positive_loss = hard_positives.pow(2).sum()
        negative_loss = F.relu(self.margin - hard_negatives).pow(2).sum()
        
        loss = positive_loss + negative_loss
        return loss



if __name__ == '__main__':
    
    
    # 1. 加载学生模型和指导模型
    logging.info(f"Loading student model: {STUDENT_MODEL_NAME}")
    student_model = SentenceTransformer(STUDENT_MODEL_NAME)
    
    guide_model = None
    if LOSS_CHOICE in ["GISTEmbedLoss", "CachedGISTEmbedLoss"]:
        logging.info(f"Loading guide model: {GUIDE_MODEL_NAME}")
        guide_model = SentenceTransformer(GUIDE_MODEL_NAME)

    # 2. 加载数据提供器
    logging.info("Loading MAGDataset provider...")
    dataloader = DataLoader_Base(data_path=DATA_PATH, taxonomy_name=TAXONOMY_NAME, processor_path="/home/u2120230631/codes/VTC/blip-itm-large-coco",tokenizer_path=STUDENT_MODEL_NAME)
    dataset = dataloader.dataset
    
    # test evaluation function
    # evaluate guide model zero shot
    # evaluate_model(guide_model, dataset)
    

    # 3. 准备 Hugging Face 格式的训练数据集
    logging.info("Preparing data for the chosen loss function...")
    # 在这个最终版本中，我们只关注你已验证最有效的 OnlineContrastiveLoss 及其 GIST 变体
    train_dataset = prepare_contrastive_hf_dataset(dataset, dataset.node_list)

    # 4. 实例化损失函数
    if LOSS_CHOICE == "OnlineContrastiveLoss":
        train_loss = losses.OnlineContrastiveLoss(model=student_model, margin=CONTRASTIVE_MARGIN)
    elif LOSS_CHOICE == "GISTOnlineContrastiveLoss":
        train_loss = GISTOnlineContrastiveLoss(model=student_model, guide_model_name=GUIDE_MODEL_NAME, margin=CONTRASTIVE_MARGIN)
    elif LOSS_CHOICE == "GISTOnlineContrastiveLoss2":
        train_loss = GISTOnlineContrastiveLoss2(model=student_model, guide_model_name=GUIDE_MODEL_NAME, margin=CONTRASTIVE_MARGIN, dirty_positive_threshold=DIRTY_POSITIVE_THRESHOLD)
    else:
        raise ValueError(f"Loss choice '{LOSS_CHOICE}' is not configured in this final script.")

    # 5. ★★★ 定义现代化的训练参数 ★★★
    args = SentenceTransformerTrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        # fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
        # ★★★ 核心：精确控制评估和保存的频率 ★★★
        eval_strategy="steps",
        eval_steps=50,  # 每 50 步进行一次评估
        save_strategy="steps",
        save_steps=5000,  # 每 50 步保存一次 checkpoint
        save_total_limit=1, # 只保留最好的一个 checkpoint
        logging_strategy="steps",
        logging_steps=10,
        run_name=MODEL_SAVE_PATH.split('/')[-1],
        # ★★★ 自动加载在所有评估中表现最好的模型 ★★★
        load_best_model_at_end=True, 
        metric_for_best_model="eval_total_mrr_standard",
        greater_is_better=True, 
        report_to="none",  # 禁用 WandB 报告
    )

    # 6. 创建你的定制评估器 (不变)
    logging.info("Preparing evaluator using the fixed test set...")
    test_queries, test_corpus, test_positive_relations = {}, {}, {}
    leaf_query_ids = set()

    template_q = "Query Node: Definition: \"{}\""
    template_p = "Parent Node: Definition: \"{}\"; Child Node: Definition: \"{}\"; Sibling Node: Definition: \"{}\";"

    # a. 填充语料库 (Corpus)
    for p_node, c_node in tqdm(dataset.all_edges, desc="Building Corpus"):
        corpus_id = f"{dataset.taxon2id[p_node]}-{dataset.taxon2id[c_node]}"
        s_node = dataset._get_sibling(p_node, c_node)
        p_desc = dataset.id2desc[dataset.taxon2id[p_node]]
        c_desc = dataset.id2desc[dataset.taxon2id[c_node]]
        s_desc = dataset.id2desc[dataset.taxon2id[s_node]]
        test_corpus[corpus_id] = template_p.format(p_desc, c_desc, s_desc)
    
    # b. 填充查询 (Queries) 和正例关系 (Positive Relations)
    for q_node in tqdm(dataset.test_node_list, desc="Building Queries & Relations"):
        query_id = str(dataset.taxon2id[q_node])
        test_queries[query_id] = template_q.format(dataset.id2desc[dataset.taxon2id[q_node]])
        
        positive_corpus_ids = set()
        is_leaf = True
        for p_node, c_node in dataset.test_node2pos[q_node]:
            if c_node != dataset.pseudo_leaf_node:
                is_leaf = False
            pos_corpus_id = f"{dataset.taxon2id[p_node]}-{dataset.taxon2id[c_node]}"
            if pos_corpus_id in test_corpus:
                positive_corpus_ids.add(pos_corpus_id)
        test_positive_relations[query_id] = positive_corpus_ids
        if is_leaf and positive_corpus_ids:
            leaf_query_ids.add(query_id)

    # c. 实例化我们自己的评估器
    evaluator = RankingEvaluator(
        queries=test_queries,
        corpus=test_corpus,
        positive_relations=test_positive_relations,
        leaf_query_ids=leaf_query_ids,
        name='taxonomy-test-ranking',
        batch_size=BATCH_SIZE * 2, # 在评估时可以使用更大的 batch size
    )

    # 7. ★★★ 创建并启动现代化的训练器 ★★★
    trainer = SentenceTransformerTrainer(
        model=student_model,
        args=args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=evaluator,
    )
    trainer.train()

    # 8. 保存最终的最佳模型
    #    因为设置了 load_best_model_at_end=True, trainer.model 现在就是最佳模型
    final_save_path = f"{MODEL_SAVE_PATH}_final"
    logging.info(f"Training finished. Saving final best model to {final_save_path}")
    trainer.save_model(final_save_path)