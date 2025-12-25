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
                 dirty_pos_threshold=0.45,
                 device=None):
        super().__init__()
        self.margin = margin
        self.mining_alpha = mining_alpha
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device

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

        
        final_fn_thresh = torch.min(t_thr, torch.tensor(0.05, device=self.device))
        
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






class FusedHardGISTLoss_1(nn.Module):
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

        # --- Cache (由 set_guide_data 注入) ---
        self.guide_q_emb = None    # Tensor: [All_Nodes, Dim]
        self.q_id_to_dense = None  # Dict: Real_ID -> Dense_Index
        self.guide_pos_map = None  # Dict: (p_id, c_id) -> Tensor
        
        # Fallback 向量：用于处理 guide_pos_map 中查不到的 Key
        self.default_pos_emb = None 

    def set_guide_data(self, guide_q_emb, q_id_to_dense, guide_pos_map):
        """
        注入离线计算好的 Guide Embeddings。
        务必确保 guide_pos_map 的 Key 是 python 的 int tuple: (int(p_id), int(c_id))
        """
        self.guide_q_emb = guide_q_emb
        self.q_id_to_dense = q_id_to_dense
        self.guide_pos_map = guide_pos_map

        if guide_q_emb is not None:
            dim = guide_q_emb.shape[1]
            # Fix 1: 初始化为随机单位向量，而不是全 0
            # 这样即使缺失 Embedding，距离也是随机的 (约等于 1.0)，而不是 0.0 (被误判为完美匹配)
            rand_vec = torch.randn(dim, device=self.device, dtype=torch.float32)
            self.default_pos_emb = F.normalize(rand_vec, p=2, dim=0)

    def _lookup_text_guide(self, batch):
        """
        根据 Batch 信息查找 Guide Embeddings
        """
        real_q_ids = batch['query_id'].tolist()
        
        # 1. 查找 Query Embeddings
        # 使用 get(id, 0) 防止极少数情况下的 key error，虽然理论上不该发生
        dense_ids = [self.q_id_to_dense.get(rid, 0) for rid in real_q_ids]
        dense_ids = torch.tensor(dense_ids, device=self.device)
        
        # Fix 2: 强制转换为 float32 进行后续计算
        g_q = self.guide_q_emb[dense_ids].float()

        # 2. 查找 Candidate Position Embeddings
        p_emb = []
        # pos_id_pair 是 list of tuples
        for p_id, c_id in batch['pos_id_pair']:
            # Fix 3: 强制转 int，确保与字典 Key 类型一致
            key = (int(p_id), int(c_id))
            
            if self.guide_pos_map is not None and key in self.guide_pos_map:
                p_emb.append(self.guide_pos_map[key].float())
            else:
                p_emb.append(self.default_pos_emb)
        
        g_p = torch.stack(p_emb)

        return g_q, g_p

    # ============================================================
    # Forward
    # ============================================================
    def forward(self, student_q_emb, student_p_emb, batch):
        """
        student_q_emb: [B, D] (Normalized)
        student_p_emb: [B, D] (Normalized)
        batch: Dict containing 'label', 'query_id', 'pos_id_pair', 'vis_q', 'vis_p', etc.
        """
        labels = batch['label']  # [B]
        device = student_q_emb.device

        # =====================================================================
        # 1. 计算 Student 距离
        # =====================================================================
        # Cosine Distance: 0.0 (Same) -> 2.0 (Opposite)
        # 假设输入已经是归一化的，否则需要先 F.normalize
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_p_emb)

        # =====================================================================
        # 2. 计算 Guide 距离 (No Grad)
        # =====================================================================
        d_text = None
        if self.use_text and self.guide_q_emb is not None:
            with torch.no_grad():
                g_q, g_p = self._lookup_text_guide(batch)
                d_text = 1 - F.cosine_similarity(g_q, g_p)

        d_visual = None
        if self.use_visual:
            with torch.no_grad():
                # 计算 Query 与 Parent, Child, Sibling 的视觉距离
                # 取三者中的最小值作为 "Query 与该 Candidate Group" 的视觉距离
                d_qp = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_p'])
                d_qc = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_c'])
                d_qs = 1 - F.cosine_similarity(batch['vis_q'], batch['vis_s'])
                d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # =====================================================================
        # 3. 生成 Mask (Joint Logic - 联合判定版)
        # =====================================================================
        final_dirty_mask = torch.zeros_like(labels, dtype=torch.bool)
        final_false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)

        # 辅助函数：计算单模态阈值
        def get_thresholds(dist, current_labels, base_thresh):
            # 1. 临时计算该模态下的脏 Mask (仅用于计算动态阈值，不作为最终 Mask)
            temp_dirty = (current_labels == 1) & (dist > base_thresh)
            
            # 2. 计算动态阈值 (排除脏数据后)
            valid_pos = dist[(current_labels == 1) & (~temp_dirty)]
            if len(valid_pos) > 0:
                pos_mean = valid_pos.mean()
                dyn_thresh = torch.min(pos_mean, torch.tensor(base_thresh, device=device))
            else:
                dyn_thresh = torch.tensor(base_thresh, device=device)
            
            return dyn_thresh

        # --- 分情况处理 ---

        # 情况 1: 同时有 Text 和 Visual (执行 Joint 逻辑)
        if d_text is not None and d_visual is not None:
            # 分别计算动态阈值
            t_fn_thresh = get_thresholds(d_text, labels, self.dirty_thresh)
            v_fn_thresh = get_thresholds(d_visual, labels, self.dirty_thresh)

            # A. 脏正例 (Joint/AND): 两个模态都认为是脏的，才算脏
            # 这是你觉得可能丢失的逻辑
            final_dirty_mask = (
                (labels == 1) & 
                (d_text > self.dirty_thresh) & 
                (d_visual > self.dirty_thresh)
            )

            # B. 假负例 (Union/OR): 只要有一个模态认为是正的，就算假负例
            final_false_neg_mask = (
                (labels == 0) & 
                ((d_text < t_fn_thresh) | (d_visual < v_fn_thresh))
            )

        # 情况 2: 只有 Text
        elif d_text is not None:
            t_fn_thresh = get_thresholds(d_text, labels, self.dirty_thresh)
            final_dirty_mask = (labels == 1) & (d_text > self.dirty_thresh)
            final_false_neg_mask = (labels == 0) & (d_text < t_fn_thresh)

        # 情况 3: 只有 Visual
        elif d_visual is not None:
            v_fn_thresh = get_thresholds(d_visual, labels, self.dirty_thresh)
            final_dirty_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            final_false_neg_mask = (labels == 0) & (d_visual < v_fn_thresh)

        # =====================================================================
        # 4. 修改 Student 距离矩阵 (Distance Rewriting)
        # =====================================================================
        # Clone 距离矩阵，避免 In-place 修改影响梯度计算
        mod_dist = student_dist.clone()

        # 处理脏正例 -> 距离设为 0.0
        # 1. 在 Positive Loss (x^2) 中变为 0
        # 2. 在 Hard Mining 时，距离最小，不会被选为 "Hard Positive"
        if final_dirty_mask.any():
            mod_dist[final_dirty_mask] = 0.0

        # 处理假负例 -> 距离设为 Margin + Buffer
        # 1. 在 Negative Loss (ReLU(m-x)^2) 中，因 x > m，Loss 变为 0
        # 2. 在 Hard Mining 时，距离很大，不会被选为 "Hard Negative"
        if final_false_neg_mask.any():
            mod_dist[final_false_neg_mask] = self.margin + 0.1

        # =====================================================================
        # 5. Hard Mining (在线难例挖掘)
        # =====================================================================
        # 分离正负例
        pos_dists = mod_dist[labels == 1]
        neg_dists = mod_dist[labels == 0]

        # 异常检查：如果 Batch全是正例或全是负例
        # AMP-safe zero (keeps graph)
        zero = student_dist.sum() * 0.0

        if len(pos_dists) == 0 or len(neg_dists) == 0:
            return zero
        
        # 确定 Hard 界限
        # Hard Positive Limit: 至少要比 "最容易区分的负例" 更难 (距离更大)
        # 注意：neg_dists 中的假负例已经是极大值，min() 会取到真正的负例
        hard_pos_limit = neg_dists.min().detach()

        # Hard Negative Limit: 至少要比 "最难区分的正例" 更难 (距离更小)
        # 注意：pos_dists 中的脏正例已经是 0，max() 会取到真正的正例
        hard_neg_limit = pos_dists.max().detach()

        # 筛选 Hard Samples
        # 加入 1e-6 防止浮点数相等判断失误
        hard_pos = pos_dists[pos_dists > hard_pos_limit]
        hard_neg = neg_dists[neg_dists < hard_neg_limit + 1e-6]

        # =====================================================================
        # 6. 计算最终 Loss
        # =====================================================================
        # Contrastive Loss:
        # Pos: d^2
        # Neg: max(0, margin - d)^2
        
        zero = student_dist.sum() * 0.0
        
        if len(hard_pos) > 0:
            pos_loss = hard_pos.pow(2).sum() if len(hard_pos) > 0 else torch.tensor(0.0, device=device)
        else:
            pos_loss = zero

        if len(hard_neg) > 0:
            neg_loss = F.relu(self.margin - hard_neg).pow(2).sum() if len(hard_neg) > 0 else torch.tensor(0.0, device=device)
        else:
            neg_loss = zero
        
        # pos_loss = hard_pos.pow(2).sum() if len(hard_pos) > 0 else torch.tensor(0.0, device=device)
        # neg_loss = F.relu(self.margin - hard_neg).pow(2).sum() if len(hard_neg) > 0 else torch.tensor(0.0, device=device)

        # 这里的归一化方式可以选 sum 或 mean。
        # 如果使用 sum，建议 Learning Rate 调小；如果使用 mean，除以的是 hard samples 的数量。
        # 为了与之前的代码保持一致，这里使用 sum。
        total_loss = pos_loss + neg_loss

        return total_loss


class FusedGISTContrastiveLoss_2(nn.Module):
    def __init__(self, 
                 margin=0.5, 
                 use_text_guide=True, 
                 use_visual_guide=True, 
                 dirty_pos_threshold=0.6, # 建议设为 0.6 或更高，先宽松一点
                 guide_model_name="Qwen/Qwen3-Embedding-0.6B",
                 device=None):
        super().__init__()
        self.margin = margin
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device
        
        # --- Online Guide Model ---
        self.guide_model = None
        if self.use_text:
            print(f"Loading Guide Model: {guide_model_name}...")
            try:
                self.guide_model = SentenceTransformer(
                    guide_model_name, 
                    trust_remote_code=True,
                    device=device,
                    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16}
                )
            except:
                print("FA2 failed, falling back...")
                self.guide_model = SentenceTransformer(guide_model_name, trust_remote_code=True, device=device)
                # self.guide_model.half()
            
            self.guide_model.eval()
            for p in self.guide_model.parameters(): p.requires_grad = False

    def forward(self, student_q_emb, student_c_emb, batch):
        labels = batch['label']
        batch_size = labels.size(0)
        
        # 1. 计算 Student 距离 (Cosine Distance: 0~2)
        # 必须确保 student_q_emb, student_c_emb 已经 normalized
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_c_emb)
        
        # 2. 计算 Guide Distances (在线推理)
        d_text = None
        if self.use_text:
            with torch.no_grad():
                g_q = self.guide_model.encode(
                    batch['q_text_raw'], prompt_name="query", 
                    convert_to_tensor=True, batch_size=batch_size, 
                    normalize_embeddings=True, show_progress_bar=False
                )
                g_p = self.guide_model.encode(
                    batch['pos_text_raw'], 
                    convert_to_tensor=True, batch_size=batch_size, 
                    normalize_embeddings=True, show_progress_bar=False
                )
                d_text = 1 - F.cosine_similarity(g_q, g_p)

        d_visual = None
        if self.use_visual:
            # Visual Guide Logic (Min distance strategy)
            vis_q, vis_p = batch['vis_q'], batch['vis_p']
            vis_c, vis_s = batch['vis_c'], batch['vis_s']
            
            d_qp = 1 - F.cosine_similarity(vis_q, vis_p)
            d_qc = 1 - F.cosine_similarity(vis_q, vis_c)
            d_qs = 1 - F.cosine_similarity(vis_q, vis_s)
            d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # ==================================================================
        # 3. 生成 Mask (Strictly following GIST Logic)
        # ==================================================================
        
        # 初始化 Mask
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # 辅助函数：正确计算动态阈值 (排除脏数据后)
        def get_dynamic_thresh(dist_vec, is_dirty_mask):
            # 关键修正：只用 Clean Positives 计算均值
            clean_pos_dist = dist_vec[(labels == 1) & (~is_dirty_mask)]
            
            if len(clean_pos_dist) > 0:
                # 均值作为基准
                avg_dist = clean_pos_dist.mean()
                # 动态阈值不能超过硬阈值
                return torch.min(torch.tensor(self.dirty_thresh, device=self.device), avg_dist)
            
            # Fallback
            return torch.tensor(self.dirty_thresh, device=self.device)

        # --- 分支处理 ---
        if self.use_text and self.use_visual:
            # Fused Strategy
            # A. 脏正例 (AND): 两个都说不像
            is_dirty_text = d_text > self.dirty_thresh
            is_dirty_vis = d_visual > self.dirty_thresh
            dirty_pos_mask = (labels == 1) & is_dirty_text & is_dirty_vis
            
            # B. 假负例 (OR): 只要有一个说像
            # 关键：先算 Mask 再算 Thresh，或者分别计算
            # 这里为了严谨，分别计算各自的 clean thresh
            # 注意：这里的 dirty_mask 并不完全准确 (text dirty 不代表 vis dirty)，
            # 但为了简化，我们假设 dirty_pos_mask 是全局的 dirty
            
            thresh_t = get_dynamic_thresh(d_text, dirty_pos_mask)
            thresh_v = get_dynamic_thresh(d_visual, dirty_pos_mask)
            
            false_neg_mask = (labels == 0) & ((d_text < thresh_t) | (d_visual < thresh_v))
            
        elif self.use_text:
            # Text Only Strategy
            # 1. 先定脏正例
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)
            
            # 2. 再算动态阈值 (排除脏正例)
            thresh_t = get_dynamic_thresh(d_text, dirty_pos_mask)
            
            # 3. 定假负例
            false_neg_mask = (labels == 0) & (d_text < thresh_t)
            
        elif self.use_visual:
            # Visual Only Strategy
            dirty_pos_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            thresh_v = get_dynamic_thresh(d_visual, dirty_pos_mask)
            false_neg_mask = (labels == 0) & (d_visual < thresh_v)

        # ==================================================================
        # 4. 修改距离矩阵 (Modify Distance) - 核心修正
        # ==================================================================
        # 必须 Clone，否则破坏梯度
        modified_student_dist = student_dist.clone()
        
        # A. 处理脏正例：距离设为 0
        # 这样它比任何正常的 Hard Positive (距离大) 都小，绝对不会被选中
        modified_student_dist[dirty_pos_mask] = 0.0
        
        # B. 处理假负例：距离设为 Margin + 0.1
        # 这样它比任何正常的 Hard Negative (距离小) 都大，绝对不会被选中
        # 且 ReLU(Margin - Dist) = 0，Loss 也是 0
        modified_student_dist[false_neg_mask] = self.margin + 0.1

        # ==================================================================
        # 5. Hard Mining (基于修改后的距离)
        # ==================================================================
        
        # 分离正负例距离
        poss = modified_student_dist[labels == 1]
        negs = modified_student_dist[labels == 0]

        # 边界检查
        if len(poss) == 0 or len(negs) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # --- Hard Positive Mining ---
        # 选出距离大的正例。
        # 阈值：负例的最小值 (最难区分的负例)
        # 如果所有负例都很远 (negs.min 很大)，那么所有正例都值得学
        # 如果有很难的负例 (negs.min 很小)，那么只有比它还差的正例才值得学
        
        # GIST 逻辑：hard_pos_limit = negs.min() (如果存在负例)
        hard_pos_limit = negs.min() if len(negs) > 0 else poss.mean()
        
        # 筛选：距离 > limit
        # 注意：脏正例已被设为 0，不可能 > limit (除非 limit < 0，不可能)
        hard_positives = poss[poss > hard_pos_limit]

        # --- Hard Negative Mining ---
        # 选出距离小的负例。
        # 阈值：正例的最大值 (最难区分的正例)
        
        # GIST 逻辑：hard_neg_limit = poss.max()
        hard_neg_limit = poss.max() if len(poss) > 0 else negs.mean()
        
        # 筛选：距离 < limit
        # 注意：假负例已被设为 margin+0.1，如果 margin+0.1 > poss.max()，它就不会被选。
        # 如果 poss.max() 很大 (比如模型很烂，正例距离都很大 > margin)，假负例可能会被选中。
        # 但没关系，Loss 计算有 ReLU 保护。
        hard_negatives = negs[negs < hard_neg_limit]

        # ==================================================================
        # 6. Loss Calculation
        # ==================================================================
        
        # GIST 原版是 Sum，这里我们用 Mean 保持数值稳定
        pos_loss = hard_positives.pow(2).sum()
            
        neg_loss = F.relu(self.margin - hard_negatives).pow(2).sum()
            
        loss = pos_loss + neg_loss
        
        return loss

class FusedGISTContrastiveLoss_Online(nn.Module):
    def __init__(self, 
                 margin=0.5, 
                 use_text_guide=True, 
                 use_visual_guide=True, 
                 dirty_pos_threshold=0.45, # ★★★ 建议先设宽松点，比如 0.6，防止误杀
                 guide_model_name="Qwen/Qwen3-Embedding-0.6B",
                 device=None):
        super().__init__()
        self.margin = margin
        self.use_text = use_text_guide
        self.use_visual = use_visual_guide
        self.dirty_thresh = dirty_pos_threshold
        self.device = device
        
        # --- 在线加载 Guide Model ---
        self.guide_model = None
        if self.use_text:
            print(f"Loading Guide Model (Online): {guide_model_name}...")
            try:
                self.guide_model = SentenceTransformer(
                    guide_model_name, 
                    trust_remote_code=True,
                    device=device,
                    model_kwargs={"attn_implementation": "flash_attention_2", "torch_dtype": torch.float16}
                )
            except:
                print("FA2 failed, falling back to standard...")
                self.guide_model = SentenceTransformer(
                    guide_model_name, 
                    trust_remote_code=True,
                    device=device
                )
                # self.guide_model.half()
            
            self.guide_model.eval()
            for p in self.guide_model.parameters(): p.requires_grad = False

    def forward(self, student_q_emb, student_c_emb, batch):
        """
        Batch: {
            'q_text_raw': List[str], 
            'pos_text_raw': List[str], 
            'vis_*': Tensor, 
            'label': Tensor
        }
        """
        labels = batch['label']
        
        # 1. 计算 Student 距离 (Cosine Distance: 0=Same, 2=Diff)
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_c_emb)
        
        # 2. 计算 Guide Distances (在线推理)
        d_text = None
        d_visual = None
        
        # --- Text Guide ---
        if self.use_text:
            with torch.no_grad():
                g_q = self.guide_model.encode(
                    batch['q_text_raw'], prompt_name="query", 
                    convert_to_tensor=True, batch_size=len(labels), 
                    normalize_embeddings=True, show_progress_bar=False
                )
                g_p = self.guide_model.encode(
                    batch['pos_text_raw'], 
                    convert_to_tensor=True, batch_size=len(labels), 
                    normalize_embeddings=True, show_progress_bar=False
                )
                d_text = 1 - F.cosine_similarity(g_q, g_p)

        # --- Visual Guide ---
        if self.use_visual:
            vis_q, vis_p = batch['vis_q'], batch['vis_p']
            vis_c, vis_s = batch['vis_c'], batch['vis_s']
            
            d_qp = 1 - F.cosine_similarity(vis_q, vis_p)
            d_qc = 1 - F.cosine_similarity(vis_q, vis_c)
            d_qs = 1 - F.cosine_similarity(vis_q, vis_s)
            
            # 逻辑：只要像 Position 里的任何一个元素，就算像
            d_visual = torch.min(d_qp, torch.min(d_qc, d_qs))

        # 3. 生成 Mask (融合策略)
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)
        
        # 动态阈值辅助函数
        def get_dynamic_thresh(dist_vec):
            valid_pos_dists = dist_vec[(labels == 1) & (dist_vec < self.dirty_thresh)]
            if len(valid_pos_dists) > 0:
                return min(valid_pos_dists.mean(), torch.tensor(self.dirty_thresh, device=self.device))
            return torch.tensor(self.dirty_thresh, device=self.device)

        # 应用策略
        if self.use_text and self.use_visual:
            # Dirty Positive (AND): 两个老师都说不像
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh) & (d_visual > self.dirty_thresh)
            
            # False Negative (OR): 只要有一个老师说像
            thresh_t = get_dynamic_thresh(d_text)
            thresh_v = get_dynamic_thresh(d_visual)
            false_neg_mask = (labels == 0) & ((d_text < thresh_t) | (d_visual < thresh_v))
            
        elif self.use_text:
            # Text Only
            thresh_t = get_dynamic_thresh(d_text)
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_text < thresh_t)
            
        elif self.use_visual:
            # Visual Only
            thresh_v = get_dynamic_thresh(d_visual)
            dirty_pos_mask = (labels == 1) & (d_visual > self.dirty_thresh)
            false_neg_mask = (labels == 0) & (d_visual < thresh_v)

        # 4. 应用 Mask 到 Student 距离矩阵 (准备 Hard Mining)
        # 必须 clone，避免 inplace 操作破坏计算图
        modified_student_dist = student_dist.clone()
        
        # 处理脏正例：距离设为 0
        # 效果：1. Positive Loss = 0^2 = 0 (忽略)
        #       2. Hard Positive Mining 时，它最小，不会被选为 Hard (因为我们要选距离大的正例)
        modified_student_dist[dirty_pos_mask] = 0.0
        
        # 处理假负例：距离设为 > margin (例如 margin + 0.1)
        # 效果：1. Negative Loss = ReLU(margin - (margin+0.1)) = 0 (忽略)
        #       2. Hard Negative Mining 时，它很大，不会被选为 Hard (因为我们要选距离小的负例)
        modified_student_dist[false_neg_mask] = self.margin + 0.1

        # 5. ★★★ Hard Mining 逻辑 (完全还原) ★★★
        poss = modified_student_dist[labels == 1]
        negs = modified_student_dist[labels == 0]

        # 异常处理：如果没有正例或没有负例 (全被 Mask 掉了或者 Batch 太小)
        if len(poss) == 0 or len(negs) == 0:
            return student_dist.sum() * 0.0

        # 选取 Hard Positive: 
        # 定义：那些 student 觉得很不相似 (dist 大)，但实际上是正例的样本。
        # 阈值：负例的最小值 (最难区分的负例) 或 正例均值。
        # 只有比“最像正例的负例”还要不相似的正例，才值得重点学。
        hard_pos_limit = negs.min() if len(negs) > 0 else poss.mean()
        hard_positives = poss[poss > hard_pos_limit]

        # 选取 Hard Negative:
        # 定义：那些 student 觉得很相似 (dist 小)，但实际上是负例的样本。
        # 阈值：正例的最大值 (最难区分的正例) 或 负例均值。
        # 只有比“最不像正例的正例”还要相似的负例，才值得重点学。
        hard_neg_limit = poss.max() if len(poss) > 0 else negs.mean()
        # +1e-6 防止浮点误差
        hard_negatives = negs[negs < hard_neg_limit + 1e-6]

        # 6. 计算 Loss
        # 如果没有筛选出 Hard Sample，为了数值稳定，可以 fallback 到 all samples 或者 return 0
        # GISTEmbed 原版是直接 sum，如果没有这就为 0
        positive_loss = hard_positives.pow(2).sum()
        negative_loss = F.relu(self.margin - hard_negatives).pow(2).sum()
        
        # Normalize by Batch Size (或者有效样本数，这里按 Batch Size 算比较稳)
        # loss = (positive_loss + negative_loss) / len(labels)
        
        # 调试打印 (建议训练初期开启)
        # if torch.rand(1).item() < 0.01:
        #     print(f"Num Hard Pos: {len(hard_positives)}/{len(poss)}, Num Hard Neg: {len(hard_negatives)}/{len(negs)}")
        
        return positive_loss + negative_loss


class FusedHardGISTLoss_2(nn.Module):
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

        # Cache
        self.guide_q_emb = None
        self.q_id_to_dense = None
        self.guide_pos_map = None
        self.default_pos_emb = None

    def set_guide_data(self, guide_q_emb, q_id_to_dense, guide_pos_map):
        """加载离线计算的 Embeddings"""
        self.guide_q_emb = guide_q_emb
        self.q_id_to_dense = q_id_to_dense
        self.guide_pos_map = guide_pos_map

        if guide_q_emb is not None:
            # 预先生成一个随机向量作为 fallback，防止 None 导致的崩溃
            # 使用 float32 保证计算稳定性
            dim = guide_q_emb.shape[1]
            self.default_pos_emb = torch.randn(dim, device=self.device, dtype=torch.float32)
            self.default_pos_emb = F.normalize(self.default_pos_emb, p=2, dim=0)

    def _lookup_text_guide(self, batch):
        """查找 Guide Embedding，并强制转为 float32"""
        real_q_ids = batch['query_id'].tolist()
        
        # 1. Look up Query
        dense_ids = [self.q_id_to_dense.get(rid, 0) for rid in real_q_ids]
        dense_ids = torch.tensor(dense_ids, device=self.device)
        g_q = self.guide_q_emb[dense_ids].float() # Force float32

        # 2. Look up Position (Candidate)
        p_emb = []
        # 统计 Miss 情况，用于 Debug
        misses = 0 
        for p_id, c_id in batch['pos_id_pair']:
            # 确保 key 类型一致 (int, int)
            key = (int(p_id), int(c_id))
            if key in self.guide_pos_map:
                # 取出时转 float32
                p_emb.append(self.guide_pos_map[key].float())
            else:
                p_emb.append(self.default_pos_emb)
                misses += 1
        if misses > 0:
            print(f"Warning: {misses} position embeddings missed in guide lookup.")
        
        g_p = torch.stack(p_emb)
        return g_q, g_p

    def forward(self, student_q_emb, student_p_emb, batch):
        labels = batch['label'] # [B]

        # 1. Student Distance (Cosine Distance: 0=Same, 2=Opposite)
        # 假设 student output 已经是 normalized 的
        student_dist = 1 - F.cosine_similarity(student_q_emb, student_p_emb)

        # 2. Guide Distances (No Grad)
        d_text = None
        if self.use_text and self.guide_q_emb is not None:
            with torch.no_grad():
                g_q, g_p = self._lookup_text_guide(batch)
                d_text = 1 - F.cosine_similarity(g_q, g_p)

        # (这里为了简洁省略 Visual Guide，逻辑一样，可以叠加)

        # 3. 生成 Masks (逻辑修正版)
        dirty_pos_mask = torch.zeros_like(labels, dtype=torch.bool)
        false_neg_mask = torch.zeros_like(labels, dtype=torch.bool)

        if d_text is not None:
            # --- A. 识别脏正例 ---
            # Label=1 但 Guide 认为不相似 (>阈值)
            dirty_pos_mask = (labels == 1) & (d_text > self.dirty_thresh)

            # --- B. 识别假负例 (修正逻辑) ---
            # 关键修正：计算平均值时，必须【排除】刚才识别出的脏正例
            valid_pos_dist = d_text[(labels == 1) & (~dirty_pos_mask)]
            
            if len(valid_pos_dist) > 0:
                # 动态阈值：有效正例的平均距离
                pos_mean = valid_pos_dist.mean()
                # 限制阈值上限，防止异常
                dynamic_thresh = torch.min(pos_mean, torch.tensor(self.dirty_thresh, device=self.device))
                
                # Label=0 但 Guide 认为很相似 (<动态阈值)
                false_neg_mask = (labels == 0) & (d_text < dynamic_thresh)
            else:
                # 如果本 Batch 没有有效正例，则不进行假负例筛选，或者使用保守阈值
                pass

        # 4. 修改 Student 距离矩阵 (Distance Rewriting)
        mod_dist = student_dist.clone()

        # 脏正例 -> 距离改为 0 (Loss=0, 且不会被选为 Hard Positive)
        if dirty_pos_mask.any():
            mod_dist[dirty_pos_mask] = 0.0

        # 假负例 -> 距离改为 > margin (Loss=0, 且不会被选为 Hard Negative)
        if false_neg_mask.any():
            # 设为一个肯定大于 margin 的值
            mod_dist[false_neg_mask] = self.margin + 0.5

        # 5. Hard Mining (GIST 风格)
        # 获取修正后的正负例距离
        pos_dists = mod_dist[labels == 1]
        neg_dists = mod_dist[labels == 0]

        # 安全检查
        if len(pos_dists) == 0 or len(neg_dists) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Hard Mining 逻辑
        # Hard Positive: 距离比 "最容易的负例" 还大 (或者简单的取 batch max)
        # 参考你之前的逻辑：Hard Neg limit 是 Pos 的 max
        # Hard Pos limit 是 Neg 的 min
        
        # 注意：这里的 neg_dists 已经被处理过，假负例都在 margin+0.5，所以 min() 会取到真正的负例
        hard_pos_limit = neg_dists.min().detach() 
        
        # 注意：这里的 pos_dists 已经被处理过，脏正例都在 0，所以 max() 会取到真正的正例
        hard_neg_limit = pos_dists.max().detach()

        # 筛选
        hard_pos = pos_dists[pos_dists > hard_pos_limit]
        # 负例我们要选距离 < 正例最大值的 (即难以区分的)
        hard_neg = neg_dists[neg_dists < hard_neg_limit]

        # 计算 Loss
        pos_loss = hard_pos.pow(2).sum() if len(hard_pos) > 0 else 0.0
        neg_loss = F.relu(self.margin - hard_neg).pow(2).sum() if len(hard_neg) > 0 else 0.0

        # 平均化 (Optional, sum 也可以，取决于学习率)
        # 之前的代码是 sum，这里保持一致
        return pos_loss + neg_loss







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