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
    
    
class DataLoader_Base(DataLoader):
    """
    Loads data using Dataset_MM2 where processing happens per item.
    Collate function handles batching of processed items.
    """
    def __init__(self, data_path, taxonomy_name, sampling_mode=0, batch_size=16, negative_size=32, shuffle=True, num_workers=8,
                 processor_path=None, # Renamed arg
                 tokenizer_path=None, img_processor_path=None,
                 train_transforms=None, test_transforms=None, eval_batch_size=128):

        print("Initializing DataLoader_Base (Standard Mode)...")
        self.batch_size = batch_size # This is the DataLoader batch size
        self.eval_batch_size = eval_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

        # --- Load Raw Data Provider ---
        # Keep reference if needed by _prep_test_data
        self.graph_dataset = MAGDataset(name=taxonomy_name, path=data_path, raw=False, existing_partition=True)

        # --- Processor and Tokenizer ---
        # self.processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        # if tokenizer_path and tokenizer_path != processor_path:
            
        # else:
        #     self.tokenizer = self.processor.tokenizer
        self.max_len = min(self.tokenizer.model_max_length, 512)

        # --- Define Transforms ---
        # Example transforms (customize as needed)
        # if img_processor_path and img_processor_path != processor_path:
        #     img_processor = AutoProcessor.from_pretrained(img_processor_path)
        # else:
        #     img_processor = self.processor.image_processor

        # target_size = img_processor.size["height"] # Or adjust key if needed (e.g., "height", "width")
        # image_mean = img_processor.image_mean
        # image_std = img_processor.image_std
        # self.train_transforms = train_transforms or transforms.Compose([
        #     transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])
        # self.test_transforms = test_transforms or transforms.Compose([
        #     transforms.Resize(target_size),
        #     transforms.CenterCrop(target_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=image_mean, std=image_std),
        # ])

        # --- Templates & Space Calculation ---
        self.template_q = "Query Node: Definition: \"{}\", Image: [MASK]"
        self.template_p = "Parent Node: Definition: \"{}\", Image: [MASK]; " \
                        "Child Node: Definition: \"{}\", Image: [MASK]; " \
                        "Sibling Node: Definition: \"{}\", Image: [MASK];"
        # Need template info for Dataset_MM
        (avail_q_space, seg_q, _) = parse_and_tokenize_template(self.template_q, self.tokenizer, self.max_len)
        (avail_p_space, seg_p, _) = parse_and_tokenize_template(self.template_p, self.tokenizer, self.max_len)
        self.template_q_info = {'space': avail_q_space, 'segments': seg_q}
        self.template_p_info = {'space': avail_p_space, 'segments': seg_p}
        print("Available space for query template: ", avail_q_space)
        print("Available space for position template: ", avail_p_space)

        # --- Instantiate the Dataset ---
        # Pass components needed by __getitem__
        self.dataset = Dataset_MM2(
            graph_dataset=self.graph_dataset,
            mode="train", # Always 'train' for the main training dataset?
            sampling_mode=sampling_mode,
            negative_size=negative_size,
            tokenizer=self.tokenizer,
            img_transforms=None, # Use training transforms
            template_q_info=self.template_q_info,
            template_p_info=self.template_p_info
        )

        # --- Initialize DataLoader ---
        super(DataLoader_Base, self).__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.collate_fn_simple, # Use simplified collate
            num_workers=self.num_workers,
            pin_memory=True # Usually good for GPU
        )
        print("DataLoader initialized.")

    def collate_fn_simple(self, batch_list_of_lists):
        """
        Collates batches where each item from dataset is a list of processed dicts.
        """
        # Flatten the list of lists into a single list of sample dicts
        flattened_batch = list(chain(*batch_list_of_lists))

        if not flattened_batch:
             return None # Handle empty batch

        query_texts_list = [sample.pop('query_text') for sample in flattened_batch] # Extract and remove
        pos_texts_list = [sample.pop('pos_text') for sample in flattened_batch]   # Extract and remove
        # Pad Text Seqs
        text_q = self.tokenizer.pad(
             {'input_ids': query_texts_list}, # input_ids expects list of lists
             padding='longest', max_length=self.max_len, return_tensors='pt', return_attention_mask=True
        )
        text_p = self.tokenizer.pad(
             {'input_ids': pos_texts_list},
             padding='longest', max_length=self.max_len, return_tensors='pt', return_attention_mask=True
        )

        # Use default collate to batch the dictionaries
        # It will stack tensors and create lists for other types
        # We need to handle the text padding separately
        try:
            batch_dict = torch.utils.data.default_collate(flattened_batch)
        except RuntimeError as e:
            print(f"Error during default_collate on remaining items: {e}")
            # Print keys and types in the remaining dicts to debug further
            if flattened_batch:
                print("Keys in remaining sample dict:", flattened_batch[0].keys())
                for key, value in flattened_batch[0].items():
                    print(f"  Key '{key}', Type: {type(value)}")
                    if isinstance(value, torch.Tensor):
                        print(f"     Shape: {value.shape}")
            raise e # Re-raise the error after printing info
        

        # Return in the expected order
        return (text_q, text_p,
                batch_dict['img_q'], batch_dict['img_p'], batch_dict['img_c'], batch_dict['img_s'],
                batch_dict['p_label'], batch_dict['c_label'], batch_dict['pc_label'])


    def _prep_test_data(self, mode):
        """
        Prepares test/validation data. Uses a separate Dataset instance
        and applies test-time transforms.
        Uses unique image optimization for efficiency.
        """
        print(f"Preparing data for mode: {mode} using standard loading...")
        g_data = self.dataset # Use raw graph data

        if mode == 'test':
            query_nodes = g_data.test_node_list
            node2pos_nodes = g_data.test_node2pos
            candidate_positions_nodes = g_data.all_edges
        elif mode == 'valid':
            query_nodes = g_data.valid_node_list
            node2pos_nodes = g_data.valid_node2pos
            candidate_positions_nodes = g_data.all_edges
        else:
            raise ValueError(f"Invalid mode: {mode}. Choose 'test' or 'valid'.")

        num_queries = len(query_nodes)
        num_candidates = len(candidate_positions_nodes)
        print(f'Number of queries: {num_queries}')
        print(f'Number of candidate positions: {num_candidates}')

        if num_queries == 0 or num_candidates == 0:
            # ... return empty structures ...
             empty_tensor = torch.empty(0)
             empty_dict = {'input_ids': empty_tensor, 'attention_mask': empty_tensor}
             return empty_dict, empty_dict, empty_tensor, empty_tensor, empty_tensor, empty_tensor, [], [], {}


        # --- Instantiate a temporary Dataset for test processing ---
        # Uses test transforms and doesn't need sampling info
        test_dataset = self.dataset

        # --- Process Queries ---
        print("Processing query data for test...")
        query_texts_list = []
        query_img_list = []
        valid_query_nodes = []
        valid_query_ids = []

        # for q_node in tqdm(query_nodes, desc="Processing Queries"):
        for q_node in query_nodes:
            if q_node not in test_dataset.taxon2id: continue
            q_id = test_dataset.taxon2id[q_node]
            img_q = test_dataset._load_and_transform_image(q_id)
            if img_q is None: continue # Skip if image failed

            text_q = test_dataset._process_query_text(q_id)

            query_texts_list.append(text_q)
            query_img_list.append(img_q)
            valid_query_nodes.append(q_node) # Keep track of original node
            valid_query_ids.append(q_id) # Keep track of ID

        # --- Process Candidate Positions (with unique image optimization) ---
        print("Processing candidate data for test...")
        position_texts_list = []
        candidate_p_ids = []
        candidate_c_ids = []
        candidate_s_ids = []
        valid_candidate_pos_nodes_test = [] # Track (p_node, c_node) for valid ones

        # 1. Collect all required IDs
        unique_candidate_node_ids = set()
        candidate_id_triplets = [] # Store (p_id, c_id, s_id) for valid positions
        original_node_pairs_for_valid = [] # Store (p_node, c_node) corresponding to triplets

        # for p_node, c_node in tqdm(candidate_positions_nodes, desc="Collecting Candidate IDs"):
        for p_node, c_node in candidate_positions_nodes:
            try:
                p_id = test_dataset.taxon2id[p_node]
                c_id = test_dataset.taxon2id[c_node]
                s_node = test_dataset._get_sibling(p_node, c_node)
                s_id = test_dataset.taxon2id[s_node]
                unique_candidate_node_ids.update([p_id, c_id, s_id])
                candidate_id_triplets.append((p_id, c_id, s_id)) # Store valid triplet
                original_node_pairs_for_valid.append((p_node, c_node)) # Store original pair
            except Exception as e:
                # print(f"Skipping candidate ({p_node}, {c_node}) during ID collection: {e}")
                pass # Skip if any ID or sibling fails

        # 2. Process unique images needed for candidates
        processed_candidate_images = {} # node_id -> processed_tensor
        print(f"Processing {len(unique_candidate_node_ids)} unique images for candidates...")
        # for node_id in tqdm(unique_candidate_node_ids, desc="Processing Unique Candidate Images"):
        for node_id in unique_candidate_node_ids:
             img_tensor = test_dataset._load_and_transform_image(node_id)
             # Store even if None, handle later
             processed_candidate_images[node_id] = img_tensor

        # Create placeholder if needed
        placeholder_img = None
        for img in processed_candidate_images.values():
             if img is not None:
                 placeholder_img = torch.zeros_like(img)
                 break
        if placeholder_img is None and unique_candidate_node_ids:
             print("Warning: Could not create placeholder for candidate images.")


        # 3. Process text and assemble image batches
        img_p_list = []
        img_c_list = []
        img_s_list = []
        print(f"Processing text for {len(candidate_id_triplets)} valid candidate positions...")
        # for idx, (p_id, c_id, s_id) in enumerate(tqdm(candidate_id_triplets, desc="Processing Candidate Text/Images")):
        for idx, (p_id, c_id, s_id) in enumerate(candidate_id_triplets):
            # Process text
            pos_text = test_dataset._process_position_text(p_id, c_id, s_id)

            # Get images (use placeholder if loading failed earlier or if ID missing)
            img_p = processed_candidate_images.get(p_id, placeholder_img)
            img_c = processed_candidate_images.get(c_id, placeholder_img)
            img_s = processed_candidate_images.get(s_id, placeholder_img)

            # Check if any essential image is missing (and no placeholder exists)
            if placeholder_img is None and None in [img_p, img_c, img_s]:
                print(f"Warning: Skipping candidate position ({p_id}, {c_id}) due to missing images and no placeholder.")
                # Need to remove the corresponding original_node_pairs_for_valid item if skipping!
                # This makes skipping complex. It's better to ensure placeholder exists.
                continue # Skip this triplet

            position_texts_list.append(pos_text)
            img_p_list.append(img_p if img_p is not None else placeholder_img)
            img_c_list.append(img_c if img_c is not None else placeholder_img)
            img_s_list.append(img_s if img_s is not None else placeholder_img)
            # Keep track of the ORIGINAL (p_node, c_node) that corresponds to this valid processed position
            valid_candidate_pos_nodes_test.append(original_node_pairs_for_valid[idx])


        # --- Stack Images & Pad Text ---
        print("Stacking images and padding text for test set...")
        img_q_batch = torch.stack(query_img_list) if query_img_list else torch.empty(0)
        img_p_batch = torch.stack(img_p_list) if img_p_list else torch.empty(0)
        img_c_batch = torch.stack(img_c_list) if img_c_list else torch.empty(0)
        img_s_batch = torch.stack(img_s_list) if img_s_list else torch.empty(0)

        text_q = self.tokenizer.pad(
            {"input_ids": query_texts_list}, padding='longest', max_length=self.max_len,
            return_tensors='pt', return_attention_mask=True
        ) if query_texts_list else {'input_ids': torch.empty(0), 'attention_mask': torch.empty(0)}

        text_p = self.tokenizer.pad(
            {"input_ids": position_texts_list}, padding='longest', max_length=self.max_len,
            return_tensors='pt', return_attention_mask=True
        ) if position_texts_list else {'input_ids': torch.empty(0), 'attention_mask': torch.empty(0)}


        # --- Prepare ground truth mapping (node2pos_indices) ---
        print("Building ground truth index mapping for test set...")
        # Use the final list of valid candidate node pairs that were actually processed
        candidate_node_pair_to_idx = {node_pair: i for i, node_pair in enumerate(valid_candidate_pos_nodes_test)}

        node2pos_indices = {}
        # Use original node2pos_nodes map (node -> set of (p_node, c_node))
        for query_node, positive_set_nodes in node2pos_nodes.items():
            if query_node not in valid_query_nodes: continue # Query was skipped
            query_id = test_dataset.taxon2id[query_node]
            indices = []
            for p_node, c_node in positive_set_nodes:
                 pos_idx = candidate_node_pair_to_idx.get((p_node, c_node), -1)
                 if pos_idx != -1:
                     indices.append(pos_idx)
            node2pos_indices[query_id] = indices

        print("Test/valid data preparation complete.")

        # Return data needed for evaluation
        # Note: valid_query_nodes list corresponds to rows in emb_q
        # valid_candidate_pos_nodes_test list corresponds to rows in emb_pos
        return (text_q, text_p, img_q_batch, img_p_batch, img_c_batch, img_s_batch,
                # valid_query_nodes, # Return the list of query nodes actually processed
                # valid_candidate_pos_nodes_test, # Return the list of candidate pairs actually processed
                node2pos_indices) # Return map from query_id -> list of valid candidate indices

    def __str__(self):
        return "\n\t".join([
            f"sampling_mode: {self.sampling_mode}",
            f"batch_size: {self.batch_size}",
            f"negative_size: {self.negative_size}",
        ])