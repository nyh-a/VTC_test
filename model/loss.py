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


class FusedHardGISTLoss(nn.Module):
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

        # --- text guide cache ---
        self.guide_q_emb = None
        self.q_id_to_dense = None
        self.guide_pos_map = None
        self.default_pos_emb = None

    # ============================================================
    # Guide injection (called by trainer)
    # ============================================================
    def set_guide_data(self, guide_q_emb, q_id_to_dense, guide_pos_map):
        self.guide_q_emb = guide_q_emb
        self.q_id_to_dense = q_id_to_dense
        self.guide_pos_map = guide_pos_map

        if guide_q_emb is not None:
            dim = guide_q_emb.shape[1]
            self.default_pos_emb = torch.zeros(dim, device=self.device)

    # ============================================================
    # Text guide lookup
    # ============================================================
    def _lookup_text_guide(self, batch):
        real_q_ids = batch['query_id'].tolist()
        dense_ids = [self.q_id_to_dense[rid] for rid in real_q_ids]
        dense_ids = torch.tensor(dense_ids, device=self.device)

        g_q = self.guide_q_emb[dense_ids].float()

        p_emb = []
        for p_id, c_id in batch['pos_id_pair']:
            key = (int(p_id), int(c_id))
            p_emb.append(self.guide_pos_map.get(key, self.default_pos_emb))
        g_p = torch.stack(p_emb).float()

        return g_q, g_p

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, student_q_emb, student_p_emb, batch):
        """
        student_q_emb: [B, D]
        student_p_emb: [B, D]
        batch: {
            label,
            query_id,
            pos_id_pair,
            vis_q, vis_p, vis_c, vis_s (optional)
        }
        """
        labels = batch['label']  # [B]

        # ------------------------------------------------------------
        # 1. Student distance
        # ------------------------------------------------------------
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_p_emb)

        # ------------------------------------------------------------
        # 2. Guide distances (NO grad)
        # ------------------------------------------------------------
        d_text = None
        if self.use_text and self.guide_q_emb is not None:
            with torch.no_grad():
                g_q, g_p = self._lookup_text_guide(batch)
                d_text = 1 - F.cosine_similarity(g_q, g_p)

        d_visual = None
        if self.use_visual:
            with torch.no_grad():
                d_qp = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_p'])
                d_qc = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_c'])
                d_qs = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_s'])
                d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # ------------------------------------------------------------
        # 3. Dirty / False masks (guide-only)
        # ------------------------------------------------------------
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)

        def dynamic_thresh(dist):
            pos = dist[labels == 1]
            if len(pos) > 0:
                return torch.min(
                    torch.tensor(self.dirty_thresh, device=dist.device),
                    pos.mean()
                )
            return torch.tensor(self.dirty_thresh, device=dist.device)

        if self.use_text and self.use_visual:
            t_thr = dynamic_thresh(d_text)
            v_thr = dynamic_thresh(d_visual)

            dirty_pos_mask = (
                (labels == 1) &
                (d_text > self.dirty_thresh) &
                (d_visual > self.dirty_thresh)
            )

            false_neg_mask = (
                (labels == 0) &
                ((d_text < t_thr) | (d_visual < v_thr))
            )

        elif self.use_text:
            t_thr = dynamic_thresh(d_text)
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_text < t_thr)

        elif self.use_visual:
            v_thr = dynamic_thresh(d_visual)
            dirty_pos_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_visual < v_thr)

        # ------------------------------------------------------------
        # 4. Distance rewriting (core trick)
        # ------------------------------------------------------------
        mod_dist = student_dist.clone()

        # dirty positive → impossible hard positive
        mod_dist[dirty_pos_mask] = 0.0

        # false negative → impossible hard negative
        mod_dist[false_neg_mask] = self.margin + 0.1

        # ------------------------------------------------------------
        # 5. Hard mining (GIST-style)
        # ------------------------------------------------------------
        pos = mod_dist[labels == 1]
        neg = mod_dist[labels == 0]

        # AMP-safe zero (keeps graph)
        zero = student_dist.sum() * 0.0

        if len(pos) == 0 or len(neg) == 0:
            return zero

        hard_pos_limit = neg.min()
        hard_neg_limit = pos.max()

        hard_pos = pos[pos > hard_pos_limit]
        hard_neg = neg[neg < hard_neg_limit + 1e-6]

        if len(hard_pos) == 0 and len(hard_neg) == 0:
            return zero

        pos_loss = hard_pos.pow(2).sum() if len(hard_pos) > 0 else zero
        neg_loss = (
            F.relu(self.margin - hard_neg).pow(2).sum()
            if len(hard_neg) > 0 else zero
        )

        return pos_loss + neg_loss



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