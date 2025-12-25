from functools import partial
from itertools import chain, product
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor
from torchvision import transforms
import gc
from torch.nn.utils.rnn import pad_sequence

from .dataset import *


def get_unique_items_and_inverse(items_list):
    unique_map = {}
    unique_items = []
    inverse_indices = []
    for item in items_list:
        if item not in unique_map:
            unique_map[item] = len(unique_items)
            unique_items.append(item)
        inverse_indices.append(unique_map[item])
    return unique_items, torch.tensor(inverse_indices)



def parse_and_tokenize_template(template_string, tokenizer, max_len):
    """
    Parses a template string with '{}' placeholders, tokenizes fixed parts,
    and calculates available space for the placeholders.

    Args:
        template_string (str): The template string (e.g., "Start {} middle {} end").
        tokenizer: The Hugging Face tokenizer instance.
        max_len (int): The maximum sequence length allowed by the model.

    Returns:
        tuple: A tuple containing:
            - available_space (int): Max tokens available for *all* placeholders combined.
            - tokenized_segments (list[list[int]]): A list where each element is the
            list of token IDs for a fixed segment of the template.
            - num_placeholders (int): The number of placeholders found (should be len(segments)-1).
    """
    if not template_string:
        return max_len, [], 0 # Or handle error as appropriate

    # Split the template by the placeholder '{}'
    # Example: "Start {} mid {} end" -> ["Start ", " mid ", " end"]
    fixed_segments = template_string.split('{}')
    num_placeholders = len(fixed_segments) - 1
    if num_placeholders < 0: # Should not happen if split works as expected
        print(f"Warning: Could not parse placeholders correctly in template: {template_string}")
        num_placeholders = 0 # Assume no placeholders if parsing fails strangely

    # Tokenize each fixed segment without adding special tokens
    tokenized_segments = []
    template_len = 0
    for segment in fixed_segments:
        # Important: Check if segment is empty, tokenizer might behave differently
        if segment:
            segment_tokens = tokenizer(segment, add_special_tokens=False)['input_ids']
            tokenized_segments.append(segment_tokens)
            template_len += len(segment_tokens)
        else:
            # Handle empty segments (e.g., if template starts/ends with '{}')
            tokenized_segments.append([])

    # Calculate space, considering special tokens the model adds ([CLS], [SEP], etc.)
    num_special_tokens = tokenizer.num_special_tokens_to_add(pair=False) # Usually 2
    available_space = max_len - template_len - num_special_tokens

    # Ensure available space isn't negative
    if available_space < 0:
        print(f"Warning: Template fixed parts ({template_len} tokens) + special tokens ({num_special_tokens}) "
            f"exceed max length ({max_len}) for template:\n'{template_string}'")
        available_space = 0 # Or handle as a hard error if desired

    return available_space, tokenized_segments, num_placeholders


    
    
class Stage2DataLoader(DataLoader):
    def __init__(self, 
                 data_path, 
                 taxonomy_name,
                 img_root_dir,
                 tokenizer_path,
                 img_feat_dir,
                 num_image_tokens=4,
                 batch_size=32,
                 negative_size=32, 
                 num_workers=8, 
                 shuffle=True,
                 
    ):
        """
        封装了 Tokenizer 加载、Transform 定义、Dataset 实例化以及 DataLoader 的初始化。
        """
        json_file = taxonomy_name + "_dataset_final.jsonl"
        json_data_path = os.path.join(img_root_dir, json_file)
        feat_file = taxonomy_name + "_blip_feat"
        img_feat_dir = os.path.join(img_feat_dir, feat_file)
        
        # 1. 加载 Tokenizer 和 Processor
        print(f"Loading Tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 3. 实例化 Dataset
        # 注意：这里需要传入你在外部定义的 Dataset_Stage2 类
        print("Instantiating Dataset_Stage2...")
        graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        self.dataset = Dataset_Stage2(
            graph_dataset=graph_dataset,
            json_data_path=json_data_path,
            img_feature_dir=img_feat_dir,
            negative_size=negative_size,
            tokenizer=self.tokenizer,
            num_tokens=num_image_tokens,
        )
        
        # 4. 初始化父类 DataLoader
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_stage2, # 绑定自定义整理函数
            pin_memory=True,
            drop_last=True
        )
        
    def get_tokenizer(self):
        """辅助方法：如果外部需要使用 tokenizer"""
        return self.tokenizer
    
    
def collate_fn_stage2(batch):
    out = {}
    pad_id = 1
    
    # Stack Tensors
    for k in ['vis_q', 'vis_p', 'vis_c', 'vis_s', 'label']:
        out[k] = torch.stack([x[k] for x in batch])
        
    # Pad Sequences
    keys = ['q_seg_start', 'q_seg_end', 'c_seg_p', 'c_seg_c', 'c_seg_s', 'c_seg_end']
    for k in keys:
        tensors = [x[k] for x in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        out[f"{k}_ids"] = padded
        
        lens = torch.tensor([len(t) for t in tensors])
        max_len = padded.size(1)
        mask = torch.arange(max_len).expand(len(lens), max_len) < lens.unsqueeze(1)
        out[f"{k}_mask"] = mask.long()
        
    out['pos_id_pair'] = [x['pos_id_pair'] for x in batch]
    out['query_id'] = torch.tensor([x['query_id'] for x in batch], dtype=torch.long)
    
    out['q_text_raw'] = [x['q_text_raw'] for x in batch]
    out['pos_text_raw'] = [x['pos_text_raw'] for x in batch]
    return out
    


def collate_fn_stage1(batch):
    batch_out = {}
    pad_id = 1  # MPNet pad_token_id
    
    # ==========================================
    # 1. Stack Images (定长 Tensor)
    # ==========================================
    # image_q: 用于 Task A & B
    # image_p, image_c, image_s: 用于 Task C (对应 P, C, S 三个位置)
    # for key in ['image_q', 'image_p', 'image_c', 'image_s']:
    #     if key in batch[0] and batch[0][key] is not None:
    #         batch_out[key] = torch.stack([x[key] for x in batch])
            
    for key in ['visual_q', 'visual_p', 'visual_c', 'visual_s']:
        if key in batch[0] and batch[0][key] is not None:
            batch_out[key] = torch.stack([x[key] for x in batch])
        
    # ==========================================
    # 2. Stack Fixed-length Task A Teacher
    # ==========================================
    # Task A Teacher 是预处理时长对齐的
    if 'task_a_ids' in batch[0]:
        batch_out['task_a_ids'] = torch.stack([x['task_a_ids'] for x in batch])
        batch_out['task_a_mask'] = torch.stack([x['task_a_mask'] for x in batch])
    
    # ==========================================
    # 3. Pad Variable-length items
    # ==========================================
    
    # 需要 Padding 的字段列表 (严格对应您的 __getitem__ 返回值)
    keys_to_pad = [
        # Task A Student
        'task_a_prefix',
        
        # Task B (Query Node)
        'task_b_teacher_ids', 'task_b_prefix', 'task_b_suffix',
        
        # Task C (Position Triplet: P, C, S)
        'task_c_teacher_ids',  # Full text teacher
        
        # Task C Segments (用于 Model 内部动态组装)
        'txt_p', 'txt_c', 'txt_s',                      # 概念词
        'seg_p_def', 'seg_c_def', 'seg_s_def', 'seg_end' # 定义+模板片段
    ]
    
    for key in keys_to_pad:
        if key not in batch[0]:
            continue
            
        # 提取 Tensor 列表
        tensors = [x[key] for x in batch]
        
        # Pad IDs
        padded_ids = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
        
        # 命名规范：如果 key 结尾不是 _ids，加上 _ids
        # 例如: txt_p -> txt_p_ids
        out_key_ids = key if key.endswith('_ids') else f"{key}_ids"
        batch_out[out_key_ids] = padded_ids
        
        # 生成 Attention Mask
        lengths = torch.tensor([len(t) for t in tensors])
        max_len = padded_ids.shape[1]
        # Mask: 1 for valid, 0 for pad
        mask = torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)
        
        # 命名规范：去掉 _ids 后缀(如果有)，加上 _mask
        # 例如: task_b_teacher_ids -> task_b_teacher_mask
        # 例如: txt_p -> txt_p_mask
        base_name = key[:-4] if key.endswith('_ids') else key
        out_key_mask = f"{base_name}_mask"
        batch_out[out_key_mask] = mask.long()
        
    return batch_out



def eval_collate_fn(batch):
    # 简单的通用 Collate
    out = {}
    pad_id = 1
    # Gather non-tensors
    for k in ['query_node', 'pos_id_pair']:
        if k in batch[0]: out[k] = [x[k] for x in batch]
    
    # Stack Tensors
    for k in ['vis_q', 'vis_p', 'vis_c', 'vis_s']:
        if k in batch[0]: out[k] = torch.stack([x[k] for x in batch])
        
    # Pad Sequences
    for k in ['q_seg_start', 'q_seg_end', 'c_seg_p', 'c_seg_c', 'c_seg_s', 'c_seg_end']:
        if k in batch[0]:
            tensors = [x[k] for x in batch]
            padded = pad_sequence(tensors, batch_first=True, padding_value=pad_id)
            out[f"{k}_ids"] = padded
            lens = torch.tensor([len(t) for t in tensors])
            mask = torch.arange(padded.size(1)).expand(len(lens), padded.size(1)) < lens.unsqueeze(1)
            out[f"{k}_mask"] = mask.long()
    return out

# ==========================================
# 2. 自定义 DataLoader Class
# ==========================================
class Stage1DataLoader(DataLoader):
    def __init__(self, 
                 data_path, 
                 taxonomy_name,
                 img_root_dir,
                 tokenizer_path,
                 img_feat_dir,
                 processor_path=None,
                 batch_size=32, 
                 num_workers=8, 
                 shuffle=True,
                 num_image_tokens=4,
                 max_seq_len=512):
        """
        封装了 Tokenizer 加载、Transform 定义、Dataset 实例化以及 DataLoader 的初始化。
        """
        
        json_file = taxonomy_name + "_dataset_final.jsonl"
        json_data_path = os.path.join(img_root_dir, json_file)
        
        # 1. 加载 Tokenizer 和 Processor
        print(f"Loading Tokenizer from {tokenizer_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # 3. 实例化 Dataset
        # 注意：这里需要传入你在外部定义的 Dataset_Stage1 类
        print("Instantiating Dataset_Stage1...")
        graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)
        self.dataset = Dataset_Stage1(
            graph_dataset=graph_dataset,
            json_data_path=json_data_path,
            img_root_dir=img_root_dir,
            tokenizer=self.tokenizer,
            feature_dir=img_feat_dir,
            # img_transforms=self.train_transforms,
            num_image_tokens=num_image_tokens,
            # test_transforms=self.test_transforms,
        )
        
        # 4. 初始化父类 DataLoader
        super().__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn_stage1, # 绑定自定义整理函数
            pin_memory=True,
            drop_last=True
        )
        
    def get_tokenizer(self):
        """辅助方法：如果外部需要使用 tokenizer"""
        return self.tokenizer