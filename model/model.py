import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel, AutoProcessor, BlipModel, BlipForImageTextRetrieval
import numpy as np
from itertools import chain 
from sentence_transformers.cross_encoder import CrossEncoder
import scipy.sparse as smat
import random

from base import BaseModel
from utils import mean_pooling
from model.modules import *
from model.loss import *


class AutoModelForSentenceEmbedding(nn.Module):
    def __init__(self, model_name, normalize=True):
        super(AutoModelForSentenceEmbedding, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.normalize = normalize

    def forward(self, **kwargs):
        model_output = self.model(**kwargs)
        embeddings = self.mean_pooling(model_output, kwargs['attention_mask'])
        if self.normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)




import torch
import torch.nn as nn

class IM2TEXT_MLP(nn.Module):
    def __init__(self, visual_dim, text_dim, num_tokens=4, middle_dim=512, n_layer=2, dropout=0.1):
        super().__init__()
        
        self.num_tokens = num_tokens
        self.text_dim = text_dim
        
        layers = []
        input_dim = visual_dim
        
        # 1. 构建中间层 (Hidden Layers)
        for _ in range(n_layer):
            layers.append(nn.Linear(input_dim, middle_dim))
            layers.append(nn.LayerNorm(middle_dim)) # 关键：加入 LayerNorm
            layers.append(nn.GELU())                # 建议：使用 GELU
            layers.append(nn.Dropout(dropout))
            input_dim = middle_dim
            
        # 2. 构建输出层 (Output Layer)
        # 输出维度 = Token维度 * Token数量
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(middle_dim, text_dim * num_tokens)

    def forward(self, x):
        # x: [Batch, visual_dim]
        
        x = self.encoder(x)
        x = self.head(x) # [Batch, text_dim * num_tokens]
        
        # 自动 Reshape 为序列形式
        # Output: [Batch, num_tokens, text_dim]
        x = x.view(-1, self.num_tokens, self.text_dim)
        
        return x
  

class TaxonomyMappingModel_F(nn.Module):
    def __init__(self, mpnet_path, num_tokens=4):
        super().__init__()
        print("Loading MPNet (Frozen)...")
        self.mpnet = AutoModel.from_pretrained(mpnet_path)
        for p in self.mpnet.parameters():
            p.requires_grad = False # 冻结 MPNet
            
        self.text_dim = self.mpnet.config.hidden_size
        self.num_tokens = num_tokens
        
        # Mapping Network (Trainable)
        # BLIP proj output is 256
        self.mapping_net = IM2TEXT_MLP(visual_dim=256, text_dim=self.text_dim, num_tokens=num_tokens)

    def _get_emb(self, input_ids):
        return self.mpnet.embeddings.word_embeddings(input_ids)
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_text_embedding(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.mpnet(input_ids, attention_mask)
            return self.mean_pooling(out, attention_mask)

    def forward(self, batch, weights, strategy):
        """
        batch 包含:
        - visual_q, visual_p, visual_c, visual_s (Tensors)
        - 各种 input_ids, attention_mask
        """
        loss_dict = {}
        total_loss = torch.tensor(0.0, device=self.mpnet.device)
        batch_size = batch['visual_q'].size(0)
        device = self.mpnet.device
        
        # 1. 生成 Image Tokens (Query)
        tokens_q = self.mapping_net(batch['visual_q']) # [B, N, 768]

        # --- Task A: Richness ---
        if weights['a'] > 0:
            # Teacher: Long Caption Emb
            target_a = self.get_text_embedding(batch['task_a_ids'], batch['task_a_mask'])
            
            # Student: "A photo of " + [TOKENS]
            prefix = self._get_emb(batch['task_a_prefix_ids'])
            input_embeds = torch.cat([prefix, tokens_q], dim=1)
            
            # Mask
            token_mask = torch.ones((batch_size, self.num_tokens), device=device)
            att_mask = torch.cat([batch['task_a_prefix_mask'], token_mask], dim=1)
            
            out_a = self.mpnet(inputs_embeds=input_embeds, attention_mask=att_mask)
            pred_a = self.mean_pooling(out_a, att_mask)
            
            loss_a = 1 - F.cosine_similarity(pred_a, target_a).mean()
            loss_dict['loss_a'] = loss_a.item()
            total_loss += weights['a'] * loss_a

        # --- Task B: Alignment ---
        if weights['b'] > 0:
            # Teacher: Def + Concept
            target_b = self.get_text_embedding(batch['task_b_teacher_ids'], batch['task_b_teacher_mask'])
            
            # Student: Prefix(Def) + [TOKENS] + Suffix
            prefix = self._get_emb(batch['task_b_prefix_ids'])
            suffix = self._get_emb(batch['task_b_suffix_ids'])
            input_embeds = torch.cat([prefix, tokens_q, suffix], dim=1)
            
            token_mask = torch.ones((batch_size, self.num_tokens), device=device)
            att_mask = torch.cat([batch['task_b_prefix_mask'], token_mask, batch['task_b_suffix_mask']], dim=1)
            
            out_b = self.mpnet(inputs_embeds=input_embeds, attention_mask=att_mask)
            pred_b = self.mean_pooling(out_b, att_mask)
            
            loss_b = 1 - F.cosine_similarity(pred_b, target_b).mean()
            loss_dict['loss_b'] = loss_b.item()
            total_loss += weights['b'] * loss_b

        # --- Task C: Structure ---
        if weights['c'] > 0:
            # P/C/S Tokens
            tokens_p = self.mapping_net(batch['visual_p'])
            tokens_c = self.mapping_net(batch['visual_c'])
            tokens_s = self.mapping_net(batch['visual_s'])
            
            # 文本 Word Embeddings (P, C, S)
            word_p = self._get_emb(batch['txt_p_ids'])
            word_c = self._get_emb(batch['txt_c_ids'])
            word_s = self._get_emb(batch['txt_s_ids'])
            
            # 上下文 Segments
            seg_p = self._get_emb(batch['seg_p_def_ids'])
            seg_c = self._get_emb(batch['seg_c_def_ids'])
            seg_s = self._get_emb(batch['seg_s_def_ids'])
            seg_end = self._get_emb(batch['seg_end_ids'])
            
            # 2. 决定 Mask 策略 (决定每个位置用图还是用文)
            # mask_decisions: [mask_p, mask_c, mask_s] (True用图, False用文)
            
            if strategy == 'all':
                mask_decisions = [True, True, True]
            elif strategy == 'random_one':
                # 随机选一个位置替换为图
                idx = random.randint(0, 2) # 0:P, 1:C, 2:S
                mask_decisions = [False, False, False]
                mask_decisions[idx] = True
            elif strategy == 'child_only':
                mask_decisions = [False, True, False] # 只替换 Child
            else:
                mask_decisions = [False, True, False] # Default

            # 3. 组装输入序列 (Embeddings & Masks)
            # P 部分
            fill_p = tokens_p if mask_decisions[0] else word_p
            mask_p = torch.ones((batch_size, self.num_tokens), device=device) if mask_decisions[0] else batch['txt_p_mask']
            
            # C 部分
            fill_c = tokens_c if mask_decisions[1] else word_c
            mask_c = torch.ones((batch_size, self.num_tokens), device=device) if mask_decisions[1] else batch['txt_c_mask']
            
            # S 部分
            fill_s = tokens_s if mask_decisions[2] else word_s
            mask_s = torch.ones((batch_size, self.num_tokens), device=device) if mask_decisions[2] else batch['txt_s_mask']
            
            # 拼接 Embedding
            # Template: [SegP] [FillP] [SegC] [FillC] [SegS] [FillS] [SegEnd]
            input_embeds = torch.cat([
                seg_p, fill_p,
                seg_c, fill_c,
                seg_s, fill_s,
                seg_end
            ], dim=1)
            
            # 拼接 Attention Mask
            attention_mask = torch.cat([
                batch['seg_p_def_mask'], mask_p,
                batch['seg_c_def_mask'], mask_c,
                batch['seg_s_def_mask'], mask_s,
                batch['seg_end_mask']
            ], dim=1)
            
            
            
            # Teacher
            target_c = self.get_text_embedding(batch['task_c_teacher_ids'], batch['task_c_teacher_mask'])
            
            # Student Forward
            out_c = self.mpnet(inputs_embeds=input_embeds, attention_mask=attention_mask)
            pred_c = self.mean_pooling(out_c, attention_mask)
            
            loss_c = 1 - F.cosine_similarity(pred_c, target_c).mean()
            loss_dict['loss_c'] = loss_c.item()
            total_loss += weights['c'] * loss_c
            
        return total_loss, loss_dict



class Stage2Model(nn.Module):
    def __init__(self, mpnet_model_name, map_net="", num_tokens=4, visual_dim=256,
                 fusion_mode='gated', static_alpha=1.0, visual_boost=3.0):
        super().__init__()
        
        print(f"Stage 2: Loading MPNet from {mpnet_model_name}...")
        
        self.mpnet = AutoModel.from_pretrained(mpnet_model_name)
        self.text_dim = self.mpnet.config.hidden_size
        self.num_tokens = num_tokens
        self.fusion_mode = fusion_mode
        self.static_alpha = static_alpha
        print(f"Stage 2 Model Init | Mode: [{self.fusion_mode}] | Alpha: {self.static_alpha}")
        
        # Mapping Net (随机初始化)
        self.mapping_net = IM2TEXT_MLP(visual_dim, self.text_dim, num_tokens)
        if map_net:
            self.load_stage1_weights(map_net)
            
        # Gated Fusion
        if self.fusion_mode == 'gated':
            self.gate_fc = nn.Linear(self.text_dim * 2, 1)
            nn.init.constant_(self.gate_fc.bias, -5)
            nn.init.xavier_uniform_(self.gate_fc.weight)
        else:
            self.gate_fc = None
        
        self.visual_boost = nn.Parameter(torch.tensor(visual_boost)) 
        

    def load_stage1_weights(self, ckpt_path):
        """从 Stage 1 Checkpoint 加载 Mapping Network"""
        print(f"Loading Stage 1 weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        
        # 适配 key (去除 mapping_net. 前缀)
        model_dict = self.mapping_net.state_dict()
        load_dict = {}
        for k, v in state_dict.items():
            clean_k = k.replace('mapping_net.', '')
            if clean_k in model_dict and v.shape == model_dict[clean_k].shape:
                load_dict[clean_k] = v
        
        self.mapping_net.load_state_dict(load_dict, strict=True)
        print(f"Loaded {len(load_dict)} keys for Mapping Network.")

    def _get_emb(self, input_ids):
        return self.mpnet.embeddings.word_embeddings(input_ids)

    def _pooling(self, model_output, attention_mask):
        """
        Mean Pooling (Future: Modality-Aware Pooling)
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    
    def _naive_pooling(self, last_hidden_state, attention_mask):
        """
        Mode 2: Naive Mean Pooling
        不管是文本还是图片，一视同仁地平均 (导致 Semantic Drowning)
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _separated_pooling(self, last_hidden_state, text_mask, img_mask):
        """
        Helper: 分别对文本和图片区域进行 Pooling
        """
        text_mask_exp = text_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        img_mask_exp = img_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

        # Text Part
        text_emb = torch.sum(last_hidden_state * text_mask_exp, dim=1) / torch.clamp(text_mask_exp.sum(1), min=1e-9)
        
        # Image Part (如果 img_mask 全为0，这里会得到 0 向量)
        img_emb = torch.sum(last_hidden_state * img_mask_exp, dim=1) / torch.clamp(img_mask_exp.sum(1), min=1e-9)

        return text_emb, img_emb

    def _apply_fusion(self, last_hidden_state, att_mask, full_text_mask, full_img_mask):
        """
        核心调度函数：根据 fusion_mode 选择不同的处理路径
        """
        # --- Mode 2: Naive ---
        if self.fusion_mode == 'naive':
            return self._naive_pooling(last_hidden_state, att_mask)

        # 剩下三种模式都需要先分离特征
        raw_text_emb, raw_img_emb = self._separated_pooling(last_hidden_state, full_text_mask, full_img_mask)
        
        # norm
        text_emb = F.normalize(raw_text_emb, p=2, dim=1)
        img_emb = F.normalize(raw_img_emb, p=2, dim=1)
        

        # --- Mode 1: Text Only ---
        if self.fusion_mode == 'text_only':
            # 即使 Transformer 内部看见了图片，我们在最后强制丢弃图片特征
            return text_emb

        # --- Mode 3: Static Weighted ---
        elif self.fusion_mode == 'static':
            # 简单的加权残差: T + alpha * I
            return text_emb + self.static_alpha * img_emb

        # --- Mode 4: Gated (Ours) ---
        elif self.fusion_mode == 'gated':
            cnt_text = full_text_mask.sum(dim=1, keepdim=True) # [B, 1], e.g., 50
            cnt_img  = full_img_mask.sum(dim=1, keepdim=True)  # [B, 1], e.g., 4
            
            # 真正的 Baseline (Global Mean Pooling 的方向)
            # 公式: (Mean_Text * Count_Text + Mean_Img * Count_Img)
            # 这等价于 Sum_Text + Sum_Img，也就是所有 Token 的总和
            base_emb = raw_text_emb * cnt_text + raw_img_emb * cnt_img
            
            
            gate_input = torch.cat([text_emb, img_emb], dim=1)
            gate = torch.sigmoid(self.gate_fc(gate_input))
            # print(f"Gate Stats | Mean: {gate.mean().item():.4f} | Std: {gate.std().item():.4f}")
            return base_emb + gate * raw_img_emb * self.visual_boost
        
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion_mode}")

    def encode_query(self, batch):
        """
        Input: [SegStart] + [IMG_Q] + [SegEnd]
        """
        device = batch['vis_q'].device
        bs = batch['vis_q'].size(0)
        
        # 1. Tokens
        tok_q = self.mapping_net(batch['vis_q'])
        
        # 2. Concat
        emb_start = self._get_emb(batch['q_seg_start_ids'])
        emb_end = self._get_emb(batch['q_seg_end_ids'])
        
        input_embeds = torch.cat([emb_start, tok_q, emb_end], dim=1)
        
        # 3. Mask
        tok_mask = torch.ones((bs, self.num_tokens), device=device)
        att_mask = torch.cat([batch['q_seg_start_mask'], tok_mask, batch['q_seg_end_mask']], dim=1)
        
        # 4. Forward
        out = self.mpnet(inputs_embeds=input_embeds, attention_mask=att_mask)
        
        # 构建 Separated Mask (为了 mode != naive)
        zeros_start = torch.zeros_like(batch['q_seg_start_mask'])
        zeros_end = torch.zeros_like(batch['q_seg_end_mask'])
        full_img_mask = torch.cat([zeros_start, tok_mask, zeros_end], dim=1)
        full_text_mask = att_mask - full_img_mask
        
        final_emb = self._apply_fusion(out.last_hidden_state, att_mask, full_text_mask, full_img_mask)
        
        return final_emb

    def encode_candidate(self, batch):
        """
        Input: [SegP] [ImgP] [SegC] [ImgC] [SegS] [ImgS] [SegEnd]
        """
        device = batch['vis_p'].device
        bs = batch['vis_p'].size(0)
        
        # 1. Tokens
        tok_p = self.mapping_net(batch['vis_p'])
        tok_c = self.mapping_net(batch['vis_c'])
        tok_s = self.mapping_net(batch['vis_s'])
        
        # 2. Concat
        input_embeds = torch.cat([
            self._get_emb(batch['c_seg_p_ids']), tok_p,
            self._get_emb(batch['c_seg_c_ids']), tok_c,
            self._get_emb(batch['c_seg_s_ids']), tok_s,
            self._get_emb(batch['c_seg_end_ids'])
        ], dim=1)
        
        # 3. Mask
        tm = torch.ones((bs, self.num_tokens), device=device)
        att_mask = torch.cat([
            batch['c_seg_p_mask'], tm,
            batch['c_seg_c_mask'], tm,
            batch['c_seg_s_mask'], tm,
            batch['c_seg_end_mask']
        ], dim=1)
        
        # 4. Forward
        out = self.mpnet(inputs_embeds=input_embeds, attention_mask=att_mask)
        
        # 构建 Separated Mask
        zp = torch.zeros_like(batch['c_seg_p_mask'])
        zc = torch.zeros_like(batch['c_seg_c_mask'])
        zs = torch.zeros_like(batch['c_seg_s_mask'])
        ze = torch.zeros_like(batch['c_seg_end_mask'])
        
        full_img_mask = torch.cat([zp, tm, zc, tm, zs, tm, ze], dim=1)
        full_text_mask = att_mask - full_img_mask
        
        final_emb = self._apply_fusion(out.last_hidden_state, att_mask, full_text_mask, full_img_mask)
        return final_emb
        

    def forward(self, batch):
        # Dual Encoder Forward
        q_emb = self.encode_query(batch)
        c_emb = self.encode_candidate(batch)
        
        # Normalize
        q_emb = F.normalize(q_emb, p=2, dim=1)
        c_emb = F.normalize(c_emb, p=2, dim=1)
        
        return q_emb, c_emb