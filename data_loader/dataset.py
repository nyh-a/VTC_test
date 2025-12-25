import os
import pickle 
import random
import time
from collections import deque
from itertools import chain, product
import numpy as np
from abc import abstractmethod
import scipy.sparse as smat

import networkx as nx
# import spacy
from networkx.algorithms import descendants
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import neighborhood
import glob
from PIL import Image
import json


MAX_TEST_SIZE = 1000
MAX_VALIDATION_SIZE = 1000

# nlp = spacy.load('en_core_web_md')

#LIU
import torch
import numpy as np
# from torch_geometric.utils.undirected import to_undirected
# from torch_geometric.utils import remove_self_loops
# def get_edges(edge_index_list):
#     undirected_edge_list = []
#     edge_index, _ = remove_self_loops(
#         torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
#     undirected_edge_list.append(to_undirected(edge_index))  # convert to undirected/bi-directed edge_index
#     return undirected_edge_list[0]

def get_edges(edge_index_list):
    undirected_edge_list = []
    edge_index_list = [[item[0], item[1]] for item in edge_index_list if item[0] != item[1]]
    for item in edge_index_list:
        if [item[1], item[0]] not in edge_index_list:
            edge_index_list.append([item[1], item[0]])
    edge_index = torch.from_numpy(np.array(edge_index_list)).transpose(1, 0)
    # edge_index, _ = remove_self_loops(
    #     torch.from_numpy(np.array(edge_index_list)).transpose(1, 0))  # remove self-loop
    undirected_edge_list.append(edge_index) # convert to undirected/bi-directed edge_index
    return undirected_edge_list[0]


class Taxon(object):
    def __init__(self, tx_id, rank=-1, norm_name="none", display_name="None", main_type="", level="-100", p_count=0,
                 c_count=0, create_date="None", use_wordnet=True, single_word=False):
        self.tx_id = tx_id
        self.rank = int(rank)
        self.norm_name = norm_name
        self.display_name = display_name
        self.main_type = main_type
        self.level = int(level)
        self.p_count = int(p_count)
        self.c_count = int(c_count)
        self.create_date = create_date
        self.use_wordnet = use_wordnet
        self.single_word = single_word
        self.description = ''
        # self.nlp_description = nlp('')

    def set_description(self, description):
        self.description = description
        # self.nlp_description = nlp(description)

    def __str__(self):
        return "Taxon {} (name: {}, level: {})".format(self.tx_id, self.norm_name, self.level)

    def __lt__(self, other):
        if self.display_name < other.display_name:
            return True
        else:
            return False

    def __eq__(self, other):
        if isinstance(other, Taxon):
            return self.display_name == other.display_name
        return False


    def __hash__(self):
        return hash(self.display_name)


class MAGDataset(object):
    def __init__(self, name, path, embed_suffix="", raw=True, existing_partition=True, partition_pattern='internal'):
        """ Raw dataset class for MAG dataset

        Parameters
        ----------
        name : str
            taxonomy name
        path : str
            path to dataset, if raw=True, this is the directory path to dataset, if raw=False, this is the pickle path
        embed_suffix : str
            suffix of embedding file name, by default ""
        raw : bool, optional
            load raw dataset from txt (True) files or load pickled dataset (False), by default True
        existing_partition : bool, optional_my
            whether to use the existing the train/validation/test partitions or randomly sample new ones, by default False
        """
        self.name = name  # taxonomy name
        self.embed_suffix = embed_suffix
        self.existing_partition = existing_partition
        self.partition_pattern = partition_pattern
        self.vocab = []  # from node_id to human-readable concept string
        self.train_node_ids = []  # a list of train node_ids
        self.validation_node_ids = []  # a list of validation node_ids
        self.test_node_ids = []  # a list of test node_ids
        self.data_path = path

        if raw:
            self._load_dataset_raw(path)
        else:
            self._load_dataset_pickled(path)

    def _load_dataset_pickled(self, pickle_path):
        print('loading pickled dataset')
        with open(pickle_path, "rb") as fin:
            data = pickle.load(fin)
        self.name = data["name"]
        self.taxonomy = data['taxonomy']
        self.node_id2taxon = data['id2taxon']
        self.taxon2node_id = data['taxon2id']
        self.vocab = data["vocab"]
        self.train_node_ids = data["train_node_ids"]
        self.validation_node_ids = data["validation_node_ids"]
        self.test_node_ids = data["test_node_ids"]

        # path = os.path.split(pickle_path)[0]
        #
        # with open(os.path.join(path, f'{self.name}.terms.train'), 'w') as f:
        #     for id in self.train_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.validation'), 'w') as f:
        #     for id in self.validation_node_ids:
        #         f.write(str(id) + '\n')
        # with open(os.path.join(path, f'{self.name}.terms.test'), 'w') as f:
        #     for id in self.test_node_ids:
        #         f.write(str(id) + '\n')
        # exit(0)
        print('dataset loaded')

    def _load_dataset_raw(self, dir_path):
        path = os.path.split(dir_path)[0]
        node_file_name = os.path.join(path, f"{self.name}.terms")
        edge_file_name = os.path.join(path, f"{self.name}.taxo")
        desc_file_name = os.path.join(path, f"{self.name}.desc")
        if self.embed_suffix == "":
            output_pickle_file_name = os.path.join(path, f"{self.name}.pickle.bin")
        else:
            output_pickle_file_name = os.path.join(path, f"{self.name}.{self.embed_suffix}.pickle.bin")

        tx_id2taxon = {}
        self.taxonomy = nx.DiGraph()

        # load nodes
        with open(desc_file_name, "r", encoding='utf-8') as fdesc:
            with open(node_file_name, "r", encoding='utf-8') as fin:
                for line, desc in tqdm(zip(fin, fdesc), desc="Loading terms"):
                    line = line.strip()
                    desc = desc.strip()
                    if line:
                        segs = line.split("\t")
                        segs_desc = desc.split("\t")
                        assert len(segs) == 2, f"Wrong number of segmentations {line}"
                        try:
                            assert segs[1] == segs_desc[0]
                            desc = segs_desc[1]
                        except AssertionError:
                            # assert len(segs_desc) == 1
                            desc = segs_desc[0]
                        taxon = Taxon(tx_id=segs[0], norm_name=segs[1], display_name=segs[1])
                        taxon.set_description(desc)
                        tx_id2taxon[segs[0]] = taxon
                        self.taxonomy.add_node(taxon)
        # load edges
        with open(edge_file_name, "r") as fin:
            for line in tqdm(fin, desc="Loading relations"):
                line = line.strip()
                if line:
                    segs = line.split("\t")
                    assert len(segs) == 2, f"Wrong number of segmentations {line}"
                    parent_taxon = tx_id2taxon[segs[0]]
                    child_taxon = tx_id2taxon[segs[1]]
                    self.taxonomy.add_edge(parent_taxon, child_taxon)

        # generate vocab
        # tx_id is the old taxon_id read from {self.name}.terms file, node_id is the new taxon_id from 0 to len(vocab)
        self.tx_id2node_id = {node.tx_id: idx for idx, node in enumerate(self.taxonomy.nodes())}
        self.node_id2tx_id = {v: k for k, v in self.tx_id2node_id.items()}
        self.node_id2taxon = {k: tx_id2taxon[v] for k, v in self.node_id2tx_id.items()}
        self.taxon2node_id = {v: k for k, v in self.node_id2taxon.items()}
        self.vocab = [tx_id2taxon[self.node_id2tx_id[node_id]].norm_name + "@@@" + str(node_id) for node_id in
                      self.node_id2tx_id]

        if self.existing_partition:
            # Use the pickled partitions
            # with open(os.path.join(path, 'split_node_ids.pickle'), 'rb') as f:
            #     data = pickle.load(f)
            # self.validation_node_ids = data["validation_node_ids"]
            # self.test_node_ids = data["test_node_ids"]
            # self.train_node_ids = data["train_node_ids"]

            # Or use the train/val/test files
            dir_path = os.path.dirname(dir_path)
            train_node_file_name = os.path.join(dir_path, f"{self.name}.terms.train")
            validation_node_file_name = os.path.join(dir_path, f"{self.name}.terms.validation")
            test_file_name = os.path.join(dir_path, f"{self.name}.terms.test")

            raw_train_node_list = self._load_node_list(train_node_file_name)
            raw_validation_node_list = self._load_node_list(validation_node_file_name)
            raw_test_node_list = self._load_node_list(test_file_name)

            self.train_node_ids = [int(n) for n in raw_train_node_list]
            self.validation_node_ids = [int(n) for n in raw_validation_node_list]
            self.test_node_ids = [int(n) for n in raw_test_node_list]

        else:
            print("Partition graph ...")
            if self.partition_pattern == 'leaf':
                sampled_node_ids = []
                for node in self.taxonomy.nodes():
                    if self.taxonomy.out_degree(node) == 0:
                        sampled_node_ids.append(self.tx_id2node_id[node.tx_id])
                random.seed(47)
                random.shuffle(sampled_node_ids)
            elif self.partition_pattern == 'internal':
                root_node = [node for node in self.taxonomy.nodes() if self.taxonomy.in_degree(node) == 0]
                sampled_node_ids = [self.tx_id2node_id[node.tx_id] for node in self.taxonomy.nodes() if
                                    node not in root_node]
                random.seed(47)
                random.shuffle(sampled_node_ids)
            else:
                raise ValueError('Unknown partition method!')

            validation_size = min(int(len(sampled_node_ids) * 0.1), MAX_VALIDATION_SIZE)
            test_size = min(int(len(sampled_node_ids) * 0.1), MAX_TEST_SIZE)
            self.validation_node_ids = sampled_node_ids[:validation_size]
            self.test_node_ids = sampled_node_ids[validation_size:(validation_size + test_size)]
            self.train_node_ids = [node_id for node_id in self.node_id2tx_id if
                                   node_id not in self.validation_node_ids and node_id not in self.test_node_ids]
            print("Finish partitioning graph ...")

        # save to pickle for faster loading next time
        print("start saving pickle data")
        with open(output_pickle_file_name, 'wb') as fout:
            data = {
                "name": self.name,
                "taxonomy": self.taxonomy,
                "id2taxon": self.node_id2taxon,
                "taxon2id": self.taxon2node_id,
                "vocab": self.vocab,
                "train_node_ids": self.train_node_ids,
                "validation_node_ids": self.validation_node_ids,
                "test_node_ids": self.test_node_ids,
            }
            pickle.dump(data, fout, pickle.HIGHEST_PROTOCOL)
        print(f"Save pickled dataset to {output_pickle_file_name}")

    def _load_node_list(self, file_path):
        node_list = []
        with open(file_path, "r") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    node_list.append(line)
        return node_list


class RawDataset(Dataset):
    def __init__(self, graph_dataset, mode="train", sampling_mode=1, negative_size=32):
        start = time.time()
        self.mode = mode
        self.sampling_mode = sampling_mode
        self.negative_size = negative_size

        self.taxon2id = graph_dataset.taxon2node_id
        self.id2taxon = graph_dataset.node_id2taxon
        train_nodes = [self.id2taxon[node_id] for node_id in graph_dataset.train_node_ids]

        # add pseudo root
        full_graph = graph_dataset.taxonomy     
        roots = [node for node in full_graph.nodes() if full_graph.in_degree(node) == 0]
        self.pseudo_root_node = Taxon(tx_id='', norm_name='pseudo root', display_name='pseudo root')
        full_graph.add_node(self.pseudo_root_node)
        for node in roots:
            full_graph.add_edge(self.pseudo_root_node, node)
        train_nodes.append(self.pseudo_root_node)
        self.full_graph = full_graph

        if mode == 'train':
            # add pseudo leaf node to core graph
            datapath = os.path.split(graph_dataset.data_path)[0]
            graph_pickle_path = os.path.join(datapath, 'subgraphs.pickle')
            graph_pickled = False
            if os.path.isfile(graph_pickle_path):
                graph_pickled = True
                with open(graph_pickle_path, 'rb') as f:
                    graphs = pickle.load(f)

            print('adding pseudo leaf')
            if graph_pickled:
                self.core_subgraph = graphs['core_subgraph']
                self.pseudo_leaf_node = graphs['pseudo_leaf_node']
            else:
                self.core_subgraph = self._get_holdout_subgraph(train_nodes)
                self.pseudo_leaf_node = Taxon(tx_id='', norm_name='pseudo leaf', display_name='pseudo leaf')
                self.core_subgraph.add_node(self.pseudo_leaf_node)
                for node in list(self.core_subgraph.nodes()):
                    self.core_subgraph.add_edge(node, self.pseudo_leaf_node)

            
            self.taxon2id[self.pseudo_leaf_node] = len(full_graph.nodes)
            self.taxon2id[self.pseudo_root_node] = len(full_graph.nodes) - 1
            self.id2taxon[len(full_graph.nodes)] = self.pseudo_leaf_node
            self.id2taxon[len(full_graph.nodes) - 1] = self.pseudo_root_node
            self.leaf_nodes = [node for node in self.core_subgraph.nodes() if self.core_subgraph.out_degree(node) == 1]

            self.id2desc = np.array([taxon.description for id, taxon in self.id2taxon.items()])

            # add interested node list and subgraph
            # remove supersource nodes (i.e., nodes without in-degree 0)
            self.node_list = [n for n in train_nodes if n != self.pseudo_root_node]

            # build node2pos, node2edge
            print('building node2pos, node2edge')
            self.node2pos, self.node2edge = {}, {}
            self.node2parents, self.node2children = {}, {}
            for node in self.node_list:
                parents = set(self.core_subgraph.predecessors(node))
                children = set(self.core_subgraph.successors(node))
                if len(children) > 1:
                    children = [i for i in children if i != self.pseudo_leaf_node]
                node_pos_edges = [(pre, suc) for pre in parents for suc in children if pre != suc]
                if len(node_pos_edges) == 0:
                    node_pos_edges = [(pre, suc) for pre in parents for suc in children]

                self.node2edge[node] = set(self.core_subgraph.in_edges(node)).union(
                    set(self.core_subgraph.out_edges(node)))
                self.node2pos[node] = node_pos_edges
                self.node2parents[node] = parents
                self.node2children[node] = children


            print('building valid and test node list')
            self.valid_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.validation_node_ids]
            if graph_pickled:
                self.valid_holdout_subgraph = graphs['valid_subgraph']
            else:
                self.valid_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.valid_node_list)
                self.valid_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.valid_holdout_subgraph.nodes() if
                             self.valid_holdout_subgraph.out_degree(node) == 0]:
                    self.valid_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.valid_id2taxon = {idx: taxon for idx, taxon in enumerate(self.valid_holdout_subgraph.nodes())}
            self.valid_taxon2id = {v: k for k, v in self.valid_id2taxon.items()}
            self.valid_node2pos = self._find_insert_position(self.valid_node_list, self.valid_holdout_subgraph)

            self.test_node_list = [self.id2taxon[node_id] for node_id in graph_dataset.test_node_ids]
            if graph_pickled:
                self.test_holdout_subgraph = graphs['test_subgraph']
            else:
                self.test_holdout_subgraph = self._get_holdout_subgraph(train_nodes + self.test_node_list)
                self.test_holdout_subgraph.add_node(self.pseudo_leaf_node)
                for node in [node for node in self.test_holdout_subgraph.nodes() if
                             self.test_holdout_subgraph.out_degree(node) == 0]:
                    self.test_holdout_subgraph.add_edge(node, self.pseudo_leaf_node)
            self.test_id2taxon = {idx: taxon for idx, taxon in enumerate(self.test_holdout_subgraph.nodes())}
            self.test_taxon2id = {v: k for k, v in self.test_id2taxon.items()}
            self.test_node2pos = self._find_insert_position(self.test_node_list, self.test_holdout_subgraph)

            if not graph_pickled:
                with open(graph_pickle_path, 'wb') as f:
                    pickle.dump({
                        'pseudo_leaf_node': self.pseudo_leaf_node,
                        'core_subgraph': self.core_subgraph,
                        'valid_subgraph': self.valid_holdout_subgraph,
                        'test_subgraph': self.test_holdout_subgraph
                    }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # used for sampling negative positions during train/validation stage
            self.pointer = 0
            self.all_edges = list(self._get_all_candidate_positions(self.core_subgraph))
            random.shuffle(self.all_edges)
            self.all_edges_id = [(self.taxon2id[edge[0]], self.taxon2id[edge[1]]) for edge in self.all_edges]
            # self.all_edge_id.sort()
            
            # TODO: for fixed sibling training
            self.fixed = False

            # self.all_pos = list(self._get_candidate_positions(self.core_subgraph, self.core_subgraph.nodes))
            self.all_pos = None
            

            self.node2pos_node = {}
            tot = 0
            for node, eles in self.node2pos.items():
                self.node2pos_node[node] = [set(), set()]
                # xu:正例只有一边时只加入前面
                if len(eles) == 1 and eles[0][1] is self.pseudo_leaf_node:
                    self.node2pos_node[node][0].add(eles[0][0])
                    tot += 1
                    continue
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])
            print(tot, len(self.node2pos))
            for node, eles in self.node2edge.items():
                for ele in eles:
                    self.node2pos_node[node][0].add(ele[0])
                    self.node2pos_node[node][1].add(ele[1])
            
        end = time.time()
        print(f"Finish loading dataset ({end - start} seconds)")

    def __str__(self):
        return f"{self.__class__.__name__} mode:{self.mode}"

    def __len__(self):
        return len(self.node_list)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Generate an data instance based on train/validation/test mode.
            
        """
        raise NotImplementedError



    def _get_holdout_subgraph(self, nodes):
        node_to_remove = [n for n in self.full_graph.nodes if n not in nodes]
        subgraph = self.full_graph.subgraph([node for node in nodes]).copy()
        for node in node_to_remove:
            parents = set()
            children = set()
            ps = deque(self.full_graph.predecessors(node))
            cs = deque(self.full_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_graph.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        # if subgraph.in_degree(s) > 1:
                        subgraph.remove_edge(node, s)
        return subgraph

    def _get_all_candidate_positions(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates
    
    def _get_candidate_positions(self, graph, care_nodes):
        # one-hop and two-hop candidates  return taxon (p, c)
        # node2descendants = {n: set(graph.neighbors(n)) for n in care_nodes}
        node2descendants = {}
        for n in care_nodes:
            neighbors = neighborhood(graph, n, 1) + neighborhood(graph, n, 2)
            node2descendants[n] = set(neighbors) 
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        return candidates
    
    def _get_candidate_positions_with_sibling(self, graph, care_nodes):
        # one-hop and two-hop candidates   return node_id (p,c,s)
        node2descendants = {}
        for n in care_nodes:
            neighbors = neighborhood(graph, n, 1) + neighborhood(graph, n, 2)
            node2descendants[n] = set(neighbors) 
        candidates = list(set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()])))
        candidates_id = [(self.taxon2id[p], self.taxon2id[c], self.taxon2id[self._get_sibling(p,c)]) for p,c in candidates]
        return candidates_id
    
    def _get_sibling(self, p, c):
        # return a taxon
        if p == self.pseudo_root_node:   # reasonable ?
            if c == self.pseudo_leaf_node:
                return self.pseudo_leaf_node
            if self.fixed:
                sibling = list(self.node2parents[c])[0]
            sibling = random.choice(list(self.node2parents[c]))
        else:
            if self.fixed:
                sibling = list(self.node2children[p])[0]
            sibling = random.choice(list(self.node2children[p])) if len(self.node2children[p])>1 else self.pseudo_leaf_node
        return sibling

    def _find_insert_position(self, node_ids, holdout_graph, ignore=[]):
        node2pos = {}
        subgraph = self.core_subgraph
        for node in node_ids:
            if node in ignore:
                continue
            parents = set()
            children = set()
            ps = deque(holdout_graph.predecessors(node))
            cs = deque(holdout_graph.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(holdout_graph.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(holdout_graph.successors(c))
            if not children:
                children.add(self.pseudo_leaf_node)
            position = [(p, c) for p in parents for c in children if p != c]
            node2pos[node] = position
        return node2pos

    


class Dataset_MM2(RawDataset):
    """
    Provides processed training samples (positive/negative).
    Loads/Transforms images and Tokenizes/Templates text in __getitem__.
    """
    def __init__(self, graph_dataset, mode="train", sampling_mode=0, negative_size=4,
                 tokenizer=None, img_transforms=None, template_q_info=None, template_p_info=None):
        # Initialize base RawDataset to get graph structure, mappings etc.
        super().__init__(graph_dataset, mode, sampling_mode, negative_size)

        # Store necessary components passed from DataLoader
        self.tokenizer = tokenizer
        self.img_transforms = img_transforms
        self.template_q_segments = template_q_info['segments']
        self.available_space_q_desc = template_q_info['space']
        self.template_p_segments = template_p_info['segments']
        self.available_space_p_desc_total = template_p_info['space']

        # Ensure necessary data from graph_dataset is available
        # These should be initialized by super().__init__(...)
        # if not all(hasattr(self, attr) for attr in ['id2desc', 'id2img', 'taxon2id', 'id2taxon',
        #                                             'node_list', 'node2pos', 'all_edges', 'node2parents',
        #                                             'node2children', '_get_sibling']):
        #      raise AttributeError("Dataset_MM2 requires graph_dataset to provide necessary attributes via RawDataset init.")

        # Pointer for negative sampling
        self.pointer = 0
        self._sampling_positions = []
        if self.sampling_mode == 0 and self.all_edges:
            self._sampling_positions = list(self.all_edges)
            random.shuffle(self._sampling_positions)
        elif self.sampling_mode == 1 and hasattr(self, 'all_pos') and self.all_pos: # Check if all_pos exists
            self._sampling_positions = list(self.all_pos)
            random.shuffle(self._sampling_positions)

        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id

        self.path = os.path.split(graph_dataset.data_path)[0]  # data/food
        self.__load_imgs__(self.path)

    def __load_imgs__(self, path):
        print("Loading images...")
        ###### file name version ######
        img_path = os.path.join(path, "imgs")
        image_paths = sorted(glob.glob(img_path + '/*.png'))
        names = [image_path.split('/')[-1].split('.')[0] for image_path in image_paths]
        self.id2img = {}
        # node_id name img
        for id, taxon in self.id2taxon.items():
            term = taxon.display_name
            idx = names.index(term)
            # self.id2img[id] = Image.open(image_paths[idx])
            self.id2img[id] = image_paths[idx]
        print("Loading images Done ")


    def _load_and_transform_image(self, node_id, transform=None):
        """Loads raw image by ID and applies transforms."""
        try:
            # Assume id2img provides PIL Image or path
            raw_img_data = self.id2img[node_id]
            if isinstance(raw_img_data, str): # If it's a path
                img = Image.open(raw_img_data).convert("RGB")
            elif isinstance(raw_img_data, Image.Image): # If it's already a PIL Image
                img = raw_img_data.convert("RGB")
            else:
                # Try to load from numpy array if id2img was loaded from .npy
                try:
                    img = Image.fromarray(raw_img_data).convert("RGB")
                except Exception:
                    raise TypeError(f"Unsupported image data type for ID {node_id}: {type(raw_img_data)}")
            if transform:
                img = transform(img)
            else:
                img = self.img_transforms(img)
            return img
        except (KeyError, IndexError, FileNotFoundError, TypeError, Exception) as e:
            print(f"Warning: Error loading/transforming image for node_id {node_id}: {e}. Returning None.")
            # Handling None should happen later, maybe return a placeholder tensor if possible
            # Placeholder creation is tricky without knowing the transform output size/dtype
            return None


    def _tokenize_and_truncate_balanced(self, desc_p, desc_c, desc_s):
        """Helper for balanced truncation used within __getitem__."""
        tok = self.tokenizer
        available_space = self.available_space_p_desc_total
        # Handle None or empty strings
        ids_p = tok(desc_p or "", add_special_tokens=False)['input_ids']
        ids_c = tok(desc_c or "", add_special_tokens=False)['input_ids']
        ids_s = tok(desc_s or "", add_special_tokens=False)['input_ids']

        current_total_len = len(ids_p) + len(ids_c) + len(ids_s)

        if current_total_len <= available_space:
            return ids_p, ids_c, ids_s

        overflow = current_total_len - available_space
        desc_tokens = [ids_p, ids_c, ids_s]
        desc_lengths = [len(ids_p), len(ids_c), len(ids_s)]

        for _ in range(overflow):
             if sum(desc_lengths) == 0: break
             longest_idx = np.argmax(desc_lengths)
             if desc_lengths[longest_idx] > 0:
                 desc_tokens[longest_idx].pop()
                 desc_lengths[longest_idx] -= 1
             else:
                 non_empty_indices = [i for i, length in enumerate(desc_lengths) if length > 0]
                 if not non_empty_indices: break
                 idx_to_truncate = non_empty_indices[0]
                 desc_tokens[idx_to_truncate].pop()
                 desc_lengths[idx_to_truncate] -= 1
        return desc_tokens[0], desc_tokens[1], desc_tokens[2]
    
    def _tokenize_and_truncate_right_priority(self, desc_p, desc_c, desc_s):
        """
        Tokenizes P, C, S and truncates from the right if needed,
        prioritizing removal from S, then C, then P.

        Returns:
            tuple: Tuple containing the final lists of token IDs for P, C, and S.
                   These lists might be empty if fully truncated.
        """
        tok = self.tokenizer
        available_space = self.available_space_p_desc_total # Total space for P, C, S tokens

        # 1. Tokenize individually without special tokens
        ids_p = tok(desc_p or "", add_special_tokens=False)['input_ids']
        ids_c = tok(desc_c or "", add_special_tokens=False)['input_ids']
        ids_s = tok(desc_s or "", add_special_tokens=False)['input_ids']

        # 2. Calculate initial total length
        current_total_len = len(ids_p) + len(ids_c) + len(ids_s)

        # 3. Check if truncation is needed
        if current_total_len <= available_space:
            return ids_p, ids_c, ids_s # No truncation needed

        # 4. Truncate from the right with priority
        overflow = current_total_len - available_space
        # Keep references to the lists for modification
        current_ids_p = ids_p
        current_ids_c = ids_c
        current_ids_s = ids_s

        for _ in range(overflow):
            # Prioritize truncating S from the end
            if len(current_ids_s) > 0:
                current_ids_s.pop() # Remove last token from S
            # Else, prioritize truncating C from the end
            elif len(current_ids_c) > 0:
                current_ids_c.pop() # Remove last token from C
            # Else, truncate P from the end
            elif len(current_ids_p) > 0:
                current_ids_p.pop() # Remove last token from P
            else:
                # Should not happen if overflow calculation is correct and available_space >= 0
                print("Warning: Could not truncate further, something went wrong.")
                break

        # Return the modified lists (which might now be empty)
        return current_ids_p, current_ids_c, current_ids_s

    def _process_query_text(self, node_id):
        """Tokenizes, templates, truncates query text."""
        desc = self.id2desc[node_id]
        ids_desc = self.tokenizer(desc or "", add_special_tokens=False,
                                  max_length=self.available_space_q_desc,
                                  truncation=True)['input_ids']
        # Combine with template segments
        # Assumes self.template_q_segments is like [seg0, seg1]
        return [self.cls_token_id] + self.template_q_segments[0] + ids_desc + self.template_q_segments[1] + [self.sep_token_id]

    def _process_position_text(self, p_id, c_id, s_id):
        """Tokenizes, templates, truncates position text."""
        desc_p = self.id2desc[p_id]
        desc_c = self.id2desc[c_id]
        desc_s = self.id2desc[s_id]

        # ids_p, ids_c, ids_s = self._tokenize_and_truncate_balanced(desc_p, desc_c, desc_s)
        ids_p, ids_c, ids_s = self._tokenize_and_truncate_right_priority(desc_p, desc_c, desc_s)

        # Combine with template segments
        # Assumes self.template_p_segments is like [seg0, seg1, seg2, seg3]
        return ([self.cls_token_id] + self.template_p_segments[0] + ids_p +
                self.template_p_segments[1] + ids_c +
                self.template_p_segments[2] + ids_s +
                self.template_p_segments[3] + [self.sep_token_id])

    def __getitem__(self, idx):
        # Get anchor query node and sample positive/negative positions
        query_node = self.node_list[idx]
        query_node_id = self.taxon2id[query_node]
        positive_positions = list(self.node2pos.get(query_node, []))

        # Sample positions (logic adapted from previous version)
        if self.negative_size > 0 and len(positive_positions) > self.negative_size:
             # Ensure at least one positive if possible, sample rest up to neg_size/2
             num_pos_to_keep = max(1, self.negative_size // 2)
             sampled_positives_nodes = random.sample(positive_positions, min(len(positive_positions), num_pos_to_keep))
        else:
             sampled_positives_nodes = positive_positions

        num_negatives_needed = max(0, self.negative_size - len(sampled_positives_nodes))
        negative_positions_nodes = self._get_k_negatives(query_node, num_negatives_needed)
        selected_positions_nodes = sampled_positives_nodes + negative_positions_nodes

        # Process each selected position (anchor query + candidate position)
        processed_samples = []
        for p_node, c_node in selected_positions_nodes:
            s_node = self._get_sibling(p_node, c_node)
            try:
                p_id = self.taxon2id[p_node]
                c_id = self.taxon2id[c_node]
                s_id = self.taxon2id[s_node]
            except KeyError as e:
                 print(f"Warning: Node ID not found for {e} in pos ({p_node},{c_node}). Skipping.")
                 continue

            # Load and transform images
            img_q = self._load_and_transform_image(query_node_id)
            img_p = self._load_and_transform_image(p_id)
            img_c = self._load_and_transform_image(c_id)
            img_s = self._load_and_transform_image(s_id)

            # Skip sample if any image failed to load (and returned None)
            if None in [img_q, img_p, img_c, img_s]:
                 print(f"Warning: Skipping sample for query {query_node_id} and pos ({p_id},{c_id}) due to image loading error.")
                 continue

            # Process text
            query_text_ids = self._process_query_text(query_node_id)
            pos_text_ids = self._process_position_text(p_id, c_id, s_id)

            # Calculate labels
            is_parent = p_node in self.node2parents.get(query_node, set())
            is_child = c_node in self.node2children.get(query_node, set())
            is_true_pair = (p_node, c_node) in self.node2pos.get(query_node, set())

            processed_samples.append({
                "query_text": query_text_ids,
                "pos_text": pos_text_ids,
                "img_q": img_q,
                "img_p": img_p,
                "img_c": img_c,
                "img_s": img_s,
                "p_label": int(is_parent),
                "c_label": int(is_child),
                "pc_label": int(is_true_pair),
                # Optional: Include IDs if needed for debugging/analysis later
                # "query_id": query_node_id,
                # "p_id": p_id,
                # "c_id": c_id,
                # "s_id": s_id,
            })

        # __getitem__ should ideally return one logical sample,
        # but here it returns a list representing the query + its pos/neg candidates.
        # The default collate_fn might handle list of dicts okay,
        # but a custom one is safer.
        return processed_samples # Return list of dicts

    def _get_k_negatives(self, query_node, negative_size, ignore=[]):
        # (Keep the implementation from the previous version - it returns node pairs)
        # ...
        if negative_size <= 0:
             return []
        true_positive_nodes = self.node2pos.get(query_node, set())
        ignore_nodes = true_positive_nodes
        negatives = []
        attempts = 0
        max_attempts = len(self._sampling_positions) * 2
        while len(negatives) < negative_size and attempts < max_attempts:
            if not self._sampling_positions: break
            if self.pointer >= len(self._sampling_positions):
                self.pointer = 0
                random.shuffle(self._sampling_positions)
            candidate_neg_pair = self._sampling_positions[self.pointer]
            self.pointer += 1
            attempts += 1
            if candidate_neg_pair not in ignore_nodes and candidate_neg_pair not in ignore:
                 negatives.append(candidate_neg_pair)
        if len(negatives) < negative_size:
             print(f"Warning: Could only find {len(negatives)} negatives out of {negative_size} requested for node {query_node}")
        return negatives
    



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


class QueryEvalDataset(Dataset):
    def __init__(self, query_nodes, feature_dir, 
                 tokenizer, num_tokens, max_len):
        self.query_nodes = query_nodes
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens
        self.feature_dir = feature_dir
        self.q_template = 'Query Node: Definition: "{}", Image: {}' 
        self.q_space, self.q_segs, _ = parse_and_tokenize_template(self.q_template, tokenizer, max_len)
        self.q_avail = self.q_space - self.num_tokens
        
        self.cls_token = [tokenizer.cls_token_id]
        self.sep_token = [tokenizer.sep_token_id]

    def __len__(self):
        return len(self.query_nodes)
    
    def _load_feat(self, term):
        # 读取 Stage 1 预处理好的 .pt 文件
        # 注意文件名处理需与 preprocess 脚本一致
        safe_name = term.replace("/", "_SPACE_").replace(" ", "_")
        path = os.path.join(self.feature_dir, f"{safe_name}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return torch.zeros(256)

    def __getitem__(self, idx):
        q_node = self.query_nodes[idx]
        vis_q = self._load_feat(q_node.display_name)
        
        desc = q_node.description
        
        q_def = self.tokenizer(desc, add_special_tokens=False)['input_ids'][:self.q_avail]
        q_seg_start = self.cls_token + self.q_segs[0] + q_def + self.q_segs[1]
        q_seg_end = self.sep_token
        
        return {
            'query_node': q_node, # 用于 ID 匹配
            'vis_q': vis_q,
            'q_seg_start': torch.tensor(q_seg_start, dtype=torch.long),
            'q_seg_end': torch.tensor(q_seg_end, dtype=torch.long),
        }
    
        

class CandidateEvalDataset(Dataset):
    def __init__(self, candidate_position_nodes, graph_dataset, feature_dir, tokenizer, 
                 num_tokens, max_len):
        self.candidate_position_nodes = candidate_position_nodes
        self.graph_dataset = graph_dataset
        self.feature_dir = feature_dir
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens
        self.cls_token = [tokenizer.cls_token_id]
        self.sep_token = [tokenizer.sep_token_id]
        
        # Template
        self.c_template = ('Parent Node: Definition: "{}", Image: {}; '
                           'Child Node: Definition: "{}"; Image: {}; '
                           'Sibling Node: Definition: "{}", Image: {}')
        self.c_space, self.c_segs, _ = parse_and_tokenize_template(self.c_template, tokenizer, max_len)
        self.c_avail = self.c_space - (num_tokens * 3)
        

    def __len__(self):
        return len(self.candidate_position_nodes)
    
    def _load_feat(self, term):
        # 读取 Stage 1 预处理好的 .pt 文件
        # 注意文件名处理需与 preprocess 脚本一致
        safe_name = term.replace("/", "_SPACE_").replace(" ", "_")
        path = os.path.join(self.feature_dir, f"{safe_name}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return torch.zeros(256)
    
    def _tokenize_and_truncate_right_priority(self, desc_p, desc_c, desc_s):
        tok = self.tokenizer; available_space = self.c_avail
        ids_p = tok(desc_p or "", add_special_tokens=False)['input_ids']
        ids_c = tok(desc_c or "", add_special_tokens=False)['input_ids']
        ids_s = tok(desc_s or "", add_special_tokens=False)['input_ids']
        current_total_len = len(ids_p) + len(ids_c) + len(ids_s)
        if current_total_len <= available_space: return ids_p, ids_c, ids_s
        overflow = current_total_len - available_space
        for _ in range(overflow):
            if len(ids_s) > 0: ids_s.pop()
            elif len(ids_c) > 0: ids_c.pop()
            elif len(ids_p) > 0: ids_p.pop()
            else: break
        return ids_p, ids_c, ids_s


    def __getitem__(self, idx):
        # 确保 pos_node_pair 是元组 (tuple)，因为列表不可哈希
        p_node, c_node = self.candidate_position_nodes[idx]
        pos_node_pair = (p_node, c_node) 

        try:
            s_node = self.graph_dataset._get_sibling(p_node, c_node)
            p_id = self.graph_dataset.taxon2id[p_node]
            c_id = self.graph_dataset.taxon2id[c_node]
            s_id = self.graph_dataset.taxon2id[s_node]
            
            vis_p = self._load_feat(p_node.display_name)
            vis_c = self._load_feat(c_node.display_name)
            vis_s = self._load_feat(s_node.display_name)
            if None in [vis_p, vis_c, vis_s]: return None

            
            # 2. 处理文本
            p_desc = self.graph_dataset.id2desc[p_id]
            c_desc = self.graph_dataset.id2desc[c_id]
            s_desc = self.graph_dataset.id2desc[s_id]
            p_def, c_def, s_def = self._tokenize_and_truncate_right_priority(p_desc, c_desc, s_desc)
            
            # [CLS] P... "Def" Image:
            c_seg_p = self.cls_token + self.c_segs[0] + p_def + self.c_segs[1]
            # ; C... "Def" Image:
            c_seg_c = self.c_segs[2] + c_def + self.c_segs[3]
            # ; S... "Def" Image:
            c_seg_s = self.c_segs[4] + s_def + self.c_segs[5]
            # ; [SEP]
            c_seg_end = self.c_segs[6] + self.sep_token

            return {
                # 'pos_node_pair': pos_node_pair, # 返回元组
                'pos_id_pair': (p_id, c_id),
                'vis_p': vis_p, 'vis_c': vis_c, 'vis_s': vis_s,
                'c_seg_p': torch.tensor(c_seg_p, dtype=torch.long),
                'c_seg_c': torch.tensor(c_seg_c, dtype=torch.long),
                'c_seg_s': torch.tensor(c_seg_s, dtype=torch.long),
                'c_seg_end': torch.tensor(c_seg_end, dtype=torch.long)
            }
        except (KeyError, AttributeError) as e:
            print(f"CandidateEvalDataset: Error for item {idx}: {e}")
            return None
        



class Dataset_Stage2(RawDataset):
    """
    Prepares data as a flat list of (query, candidate, label) pairs,
    specifically for OnlineContrastiveLoss. It pre-builds all pairs in memory.
    This version is simplified to work with Model_MM_Simplified.
    """
    def __init__(self, graph_dataset, json_data_path, img_feature_dir, negative_size=4,
                 tokenizer=None, num_tokens=4):
        # Initialize base RawDataset to get graph structure, mappings etc.
        
        mode = "train"
        sampling_mode=0
        super().__init__(graph_dataset, mode, sampling_mode, negative_size)

        self.feature_dir = img_feature_dir
        self.tokenizer = tokenizer
        self.num_tokens = num_tokens
        self.max_len = min(self.tokenizer.model_max_length, 512)
        self.cls_token = [tokenizer.cls_token_id]
        self.sep_token = [tokenizer.sep_token_id]
        
        self.node_metadata = self._load_json_data(json_data_path)
        
        self.q_template = 'Query Node: Definition: "{}", Image: ' 
        self.q_space, self.q_segs, _ = parse_and_tokenize_template(
            self.q_template, tokenizer, self.max_len
        )
        # 实际可用空间 = 总空间 - 图片Token数
        self.q_avail = self.q_space - self.num_tokens

        # --- Candidate Template ---
        # 占位符顺序: 0:P_Def, 1:P_Img, 2:C_Def, 3:S_Def, 4:S_Img
        # 结构: [Seg0] P_Def [Seg1] P_Img [Seg2] C_Def [Seg3] S_Def [Seg4] S_Img [Seg5]
        self.c_template = (
            'Parent Node: Definition: "{}", Image: {}; '
            'Child Node: Definition: "{}"; Image: {}; '
            'Sibling Node: Definition: "{}", Image: {}'
        )
        self.c_space, self.c_segs, _ = parse_and_tokenize_template(
            self.c_template, tokenizer, self.max_len
        )
        # 实际可用空间 = 总空间 - (P_Img + S_Img)
        self.c_avail = self.c_space - (self.num_tokens * 3)
        
        
        # ★★★ The "Big Pool": Create a flat list of all training pairs ★★★
        self._sampling_positions = list(self.all_edges)
        random.shuffle(self._sampling_positions)
        self.all_training_pairs = []
        if self.mode == "train":
            for query_node in tqdm(self.node_list, desc="Flattening dataset into a 'big pool'"):
                positive_positions = list(self.node2pos.get(query_node, []))
                
                # Sample N negatives for EACH positive to maintain a balanced ratio
                num_negatives_to_sample = self.negative_size * len(positive_positions)
                negative_positions = self._get_k_negatives(query_node, num_negatives_to_sample)
                
                for pos_pair in positive_positions:
                    self.all_training_pairs.append({'query': query_node, 'candidate': pos_pair, 'label': 1})
                
                for neg_pair in negative_positions:
                    self.all_training_pairs.append({'query': query_node, 'candidate': neg_pair, 'label': 0})
        
        print(f"Created {len(self.all_training_pairs)} total training pairs.")

    def __len__(self):
        # The length of the dataset is the total number of pairs in our "big pool"
        return len(self.all_training_pairs)
    
    def _load_feat(self, term):
        # 读取 Stage 1 预处理好的 .pt 文件
        # 注意文件名处理需与 preprocess 脚本一致
        safe_name = term.replace("/", "_SPACE_").replace(" ", "_")
        path = os.path.join(self.feature_dir, f"{safe_name}.pt")
        if os.path.exists(path):
            return torch.load(path)
        return torch.zeros(256)
    
    def _load_json_data(self, json_path):
        print(f"Loading metadata from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            # 兼容 JSONL 和 List JSON
            try:
                # 尝试一次性读取
                data = json.load(f)
                if isinstance(data, list):
                    data_list = data
                else:
                    data_list = [data] # 只有一个对象的情况
            except json.JSONDecodeError:
                # 尝试按行读取 (JSONL)
                f.seek(0)
                data_list = [json.loads(line) for line in f]
                
        metadata_map = {}
        for item in data_list:
            term = item.get('term')
            if term:
                metadata_map[term] = item
        
        # add pseduo root and pseudo leaf
        taxonomy_name = data_list[0]['image_path'].split('/')[0]
        term = self.pseudo_root_node.display_name
        pseudo_root = {
        "term": term, "definition": term, "image_path": f"{taxonomy_name}/{term}.png",
        "has_image": True, "dalle_caption": term
        }
        metadata_map[term] = pseudo_root
        
        term = self.pseudo_leaf_node.display_name
        pseudo_leaf = {
        "term": term, "definition": term, "image_path": f"{taxonomy_name}/{term}.png",
        "has_image": True, "dalle_caption": term
        }
        metadata_map[term] = pseudo_leaf

        print(f"Loaded {len(metadata_map)} metadata entries.")
        
        
        return metadata_map



    def _tokenize_and_truncate_right_priority(self, desc_p, desc_c, desc_s):
        """
        Tokenizes P, C, S and truncates from the right if needed,
        prioritizing removal from S, then C, then P.

        Returns:
            tuple: Tuple containing the final lists of token IDs for P, C, and S.
                   These lists might be empty if fully truncated.
        """
        tok = self.tokenizer
        available_space = self.c_avail # Total space for P, C, S tokens

        # 1. Tokenize individually without special tokens
        ids_p = tok(desc_p or "", add_special_tokens=False)['input_ids']
        ids_c = tok(desc_c or "", add_special_tokens=False)['input_ids']
        ids_s = tok(desc_s or "", add_special_tokens=False)['input_ids']

        # 2. Calculate initial total length
        current_total_len = len(ids_p) + len(ids_c) + len(ids_s)

        # 3. Check if truncation is needed
        if current_total_len <= available_space:
            return ids_p, ids_c, ids_s # No truncation needed

        # 4. Truncate from the right with priority
        overflow = current_total_len - available_space
        # Keep references to the lists for modification
        current_ids_p = ids_p
        current_ids_c = ids_c
        current_ids_s = ids_s

        for _ in range(overflow):
            # Prioritize truncating S from the end
            if len(current_ids_s) > 0:
                current_ids_s.pop() # Remove last token from S
            # Else, prioritize truncating C from the end
            elif len(current_ids_c) > 0:
                current_ids_c.pop() # Remove last token from C
            # Else, truncate P from the end
            elif len(current_ids_p) > 0:
                current_ids_p.pop() # Remove last token from P
            else:
                # Should not happen if overflow calculation is correct and available_space >= 0
                print("Warning: Could not truncate further, something went wrong.")
                break

        # Return the modified lists (which might now be empty)
        return current_ids_p, current_ids_c, current_ids_s
    
    def __getitem__(self, idx):
        # Fetches one pair from the "big pool"
        pair_info = self.all_training_pairs[idx]
        
        query_node = pair_info['query']
        p_node, c_node = pair_info['candidate']
        label = pair_info['label']
        
        s_node = self._get_sibling(p_node, c_node)

        try:
            query_node_id = self.taxon2id[query_node]
            p_id, c_id, s_id = self.taxon2id[p_node], self.taxon2id[c_node], self.taxon2id[s_node]
        except KeyError:
            return None # Skip corrupted samples

        # --- Process this single pair (image and text) ---
        vis_q = self._load_feat(query_node.display_name)
        vis_p = self._load_feat(p_node.display_name)
        vis_c = self._load_feat(c_node.display_name)
        vis_s = self._load_feat(s_node.display_name)
        if None in [vis_q, vis_p, vis_c, vis_s]: return None

        # Process text
        # query
        q_desc = self.id2desc[query_node_id]
        q_def  = self.tokenizer(q_desc or "", add_special_tokens=False,
                                  max_length=self.q_avail, truncation=True)['input_ids']
        q_seg_start = self.cls_token + self.q_segs[0] + q_def + self.q_segs[1]
        q_seg_end = self.sep_token
        
        # candidate
        p_desc = self.id2desc[p_id]
        c_desc = self.id2desc[c_id]
        s_desc = self.id2desc[s_id]
        p_def, c_def, s_def = self._tokenize_and_truncate_right_priority(p_desc, c_desc, s_desc)
        
        # [CLS] P... "Def" Image:
        c_seg_p = self.cls_token + self.c_segs[0] + p_def + self.c_segs[1]
        # ; C... "Def" Image:
        c_seg_c = self.c_segs[2] + c_def + self.c_segs[3]
        # ; S... "Def" Image:
        c_seg_s = self.c_segs[4] + s_def + self.c_segs[5]
        # ; [SEP]
        c_seg_end = self.c_segs[6] + self.sep_token
        
        
        q_text_raw = f'Query Node: Definition: {q_desc}'
        # 2. Position Raw Text
        
        pos_text_raw = (
            f'Parent Node: Definition: {p_desc}; '
            f'Child Node: Definition: {c_desc}; '
            f'Sibling Node: Definition: {s_desc}'
        )
        
        
        # Return a flat dictionary containing all necessary info for one pair
        return {
            # Query Inputs
            'vis_q': vis_q,
            'q_seg_start': torch.tensor(q_seg_start, dtype=torch.long),
            'q_seg_end': torch.tensor(q_seg_end, dtype=torch.long),
            
            # Candidate Position Inputs
            'vis_p': vis_p, 'vis_c': vis_c, 'vis_s': vis_s,
            'c_seg_p': torch.tensor(c_seg_p, dtype=torch.long),
            'c_seg_c': torch.tensor(c_seg_c, dtype=torch.long),
            'c_seg_s': torch.tensor(c_seg_s, dtype=torch.long),
            'c_seg_end': torch.tensor(c_seg_end, dtype=torch.long),
            
            'query_id': query_node_id,
            'pos_id_pair': (p_id, c_id),
            
            'label': torch.tensor(label, dtype=torch.float),
            
            'q_text_raw': q_text_raw,
            'pos_text_raw': pos_text_raw
        }
        
    def _get_k_negatives(self, query_node, negative_size, ignore=[]):
        # (Keep the implementation from the previous version - it returns node pairs)
        # ...
        if negative_size <= 0:
             return []
        true_positive_nodes = self.node2pos.get(query_node, set())
        ignore_nodes = true_positive_nodes
        negatives = []
        attempts = 0
        max_attempts = len(self._sampling_positions) * 2
        while len(negatives) < negative_size and attempts < max_attempts:
            if not self._sampling_positions: break
            if self.pointer >= len(self._sampling_positions):
                self.pointer = 0
                random.shuffle(self._sampling_positions)
            candidate_neg_pair = self._sampling_positions[self.pointer]
            self.pointer += 1
            attempts += 1
            if candidate_neg_pair not in ignore_nodes and candidate_neg_pair not in ignore:
                 negatives.append(candidate_neg_pair)
        if len(negatives) < negative_size:
             print(f"Warning: Could only find {len(negatives)} negatives out of {negative_size} requested for node {query_node}")
        return negatives
        
        
        
class Dataset_Stage1(RawDataset):
    """
    Provides processed training samples (positive/negative).
    Loads/Transforms images and Tokenizes/Templates text in __getitem__.
    """
    def __init__(self, graph_dataset, json_data_path, img_root_dir="", feature_dir="",num_image_tokens=4, 
                 tokenizer=None, img_transforms=None, test_transforms=None):
        # Initialize base RawDataset to get graph structure, mappings etc.
        super().__init__(graph_dataset, mode="train", sampling_mode=0, negative_size=0)

        # Store necessary components passed from DataLoader
        self.tokenizer = tokenizer
        self.img_transforms = img_transforms
        self.test_transforms = test_transforms
        self.cls_token_id = self.tokenizer.cls_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        
        self.max_seq_len = min(self.tokenizer.model_max_length, 512)
        
        self.img_root_dir = img_root_dir
        self.num_image_tokens = num_image_tokens
        
        self.feature_dir = feature_dir
        
        self.def_drop_prob = 0.1          # whole-definition dropout (5–15%)
        self.def_token_mask_prob = 0.5    # token-level masking (30–60%)
        self.noise_scale = 0.0             # feature noise scale (0.0–0.1)
        self.mask_token_id = self.tokenizer.mask_token_id
        
        # =========================================================================
        # 1. 对齐 Stage 2 的 Template Q (用于 Task B)
        # Template: Query Node: Definition: "...", Image: [MASK]
        # 我们把 [MASK] 视为 Concept/Token 的位置
        # =========================================================================
        self.task_b_template_str = 'Query Node: Definition: "{}", Image: {}'
        self.task_b_space, self.task_b_segments, _ = parse_and_tokenize_template(
            self.task_b_template_str, tokenizer, self.max_seq_len
        )
        
        # =========================================================================
        # 2. 对齐 Stage 2 的 Template P (用于 Task C)
        # Template: Parent Node: Definition: "...", Image: ...; Child Node ...
        # 注意：这里有 6 个空位：(P_Def, P_Img), (C_Def, C_Img), (S_Def, S_Img)
        # =========================================================================
        self.task_c_template_str = (
            'Parent Node: Definition: "{}", Image: {}; '
            'Child Node: Definition: "{}", Image: {}; '
            'Sibling Node: Definition: "{}", Image: {};'
        )
        self.task_c_space, self.task_c_segments, _ = parse_and_tokenize_template(
            self.task_c_template_str, tokenizer, self.max_seq_len
        )

        self.node_metadata = self._load_json_data(json_data_path)
        
        
    def _load_json_data(self, json_path):
        print(f"Loading metadata from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            # 兼容 JSONL 和 List JSON
            try:
                # 尝试一次性读取
                data = json.load(f)
                if isinstance(data, list):
                    data_list = data
                else:
                    data_list = [data] # 只有一个对象的情况
            except json.JSONDecodeError:
                # 尝试按行读取 (JSONL)
                f.seek(0)
                data_list = [json.loads(line) for line in f]
                
        metadata_map = {}
        for item in data_list:
            term = item.get('term')
            if term:
                metadata_map[term] = item
        
        # add pseduo root and pseudo leaf
        taxonomy_name = data_list[0]['image_path'].split('/')[0]
        term = self.pseudo_root_node.display_name
        pseudo_root = {
        "term": term, "definition": term, "image_path": f"{taxonomy_name}/{term}.png",
        "has_image": True, "dalle_caption": term
        }
        metadata_map[term] = pseudo_root
        
        term = self.pseudo_leaf_node.display_name
        pseudo_leaf = {
        "term": term, "definition": term, "image_path": f"{taxonomy_name}/{term}.png",
        "has_image": True, "dalle_caption": term
        }
        metadata_map[term] = pseudo_leaf

        print(f"Loaded {len(metadata_map)} metadata entries.")
        
        
        return metadata_map

    def _truncate_tokens(self, tokens, max_len):
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens
    
    def _get_node_info(self, node):
        """辅助函数：获取节点的 concept 和 definition"""
        node_name = node.display_name
        node_id = self.taxon2id[node]
        
        meta = self.node_metadata.get(node_name, {})
        
        concept = meta.get('term', node_name)
        # 优先从 JSON 拿 definition，没有则从 raw dataset 拿
        definition = meta.get('definition', self.id2desc[node_id])
        return concept, definition


    def _load_and_transform_image(self, node_id, transform=None):
        """Loads raw image by ID and applies transforms."""
        try:
            # Assume id2img provides PIL Image or path
            term = self.id2taxon[node_id].display_name
            img_path = self.node_metadata.get(term, {}).get('image_path', "")
            img_full_path = os.path.join(self.img_root_dir, img_path)
            
            if os.path.exists(img_full_path) and img_path:
                img = Image.open(img_full_path).convert("RGB")
            if transform:
                img = transform(img)
            else:
                img = self.img_transforms(img)
            return img
        except (KeyError, IndexError, FileNotFoundError, TypeError, Exception) as e:
            print(f"Warning: Error loading/transforming image for node_id {node_id}: {e}. Returning None.")
            # Handling None should happen later, maybe return a placeholder tensor if possible
            # Placeholder creation is tricky without knowing the transform output size/dtype
            return None
        
    def _load_feat(self, term):
        safe_name = term.replace("/", "_SPACE_").replace(" ", "_")
        path = os.path.join(self.feature_dir, f"{safe_name}.pt")
        if os.path.exists(path):
            feat = torch.load(path)
            # Feature Noise Injection (Augmentation)
            if self.noise_scale > 0:
                feat = feat + torch.randn_like(feat) * self.noise_scale
            return feat
        return torch.zeros(256)
        
    def set_config(self, drop, mask, noise):
        """在训练循环中调用此函数来改变难度"""
        self.def_drop_prob = drop
        self.token_mask_prob = mask
        self.noise_scale = noise
        
    def _maybe_drop_definition(self, tokens):
        """
        Instance-level: drop entire definition with prob p
        """
        if random.random() < self.def_drop_prob:
            return []
        return tokens
    
    def _mask_definition_tokens(self, tokens):
        """
        Token-level masking inside definition
        """
        if not tokens:
            return tokens

        masked = []
        for t in tokens:
            if random.random() < self.def_token_mask_prob:
                masked.append(self.mask_token_id)
            else:
                masked.append(t)
        return masked



    def __getitem__(self, idx):
        node = self.node_list[idx]
        node_id = self.taxon2id[node]
        node_name = node.display_name
        
        # 获取当前节点数据
        meta = self.node_metadata.get(node_name, {})
        concept_text = meta.get('term', node_name)
        definition_text = meta.get('definition', self.id2desc[node_id])
        long_caption_text = meta.get('dalle_caption', definition_text)
        
        # 加载图片
        # image_q = self._load_and_transform_image(node_id, self.img_transforms)
        visual_q = self._load_feat(node_name)

        # ================= Task A: Richness (不变) =================
        # 既然 Task A 是为了让 Token 包含视觉丰富性，用简单的 "A photo of" 也是可以的
        # 但如果为了极致对齐，也可以改成 template_q 的格式，不过 Task A 重点是 Teacher (Long Caption)，Student 形式次要
        task_a_teacher = self.tokenizer(long_caption_text, max_length=self.max_seq_len, truncation=True, padding='max_length', return_tensors='pt')
        prefix_a_tokens = self.tokenizer("A photo of ", add_special_tokens=False)['input_ids']
        task_a_prefix = [self.cls_token_id] + prefix_a_tokens

        # ================= Task B: Query Alignment =================
        # Template: Query Node: Definition: "{DEF}", Image: {CONCEPT/TOKEN}
        
        def_tokens = self.tokenizer(definition_text, add_special_tokens=False)['input_ids']
        def_tokens = self._maybe_drop_definition(def_tokens)
        def_tokens = self._mask_definition_tokens(def_tokens)
        
        concept_tokens = self.tokenizer(concept_text, add_special_tokens=False)['input_ids']
        
        # 空间分配：
        # 模板有两个槽位：Def 和 Image。
        # Teacher 填 Concept，Student 填 Image Token。
        # 我们优先保证 Concept 不被截断。
        
        avail_b = self.task_b_space - self.num_image_tokens # 预留给 Student 的 Token 空间
        trunc_concept = self._truncate_tokens(concept_tokens, avail_b // 4) # Concept 通常短
        trunc_def = self._truncate_tokens(def_tokens, avail_b - len(trunc_concept))
        
        # Teacher (纯文本): Definition: {DEF}, Image: {CONCEPT}
        # segments: ["Query Node: Definition: \"", "\", Image: ", ""]
        task_b_teacher_ids = (
            [self.cls_token_id] + 
            self.task_b_segments[0] + trunc_def + 
            self.task_b_segments[1] + trunc_concept + 
            [self.sep_token_id] # 最后一个 segment 是空的，直接接 SEP
        )
        
        # Student (图片): Definition: {DEF}, Image: [IMG_TOKEN]
        task_b_prefix = [self.cls_token_id] + self.task_b_segments[0] + trunc_def + self.task_b_segments[1]
        task_b_suffix = [self.sep_token_id] # Image Token 后面没有东西了

        # ================= Task C: Structure Composability =================
        # Template: P_Def, P_Img; C_Def, C_Img; S_Def, S_Img
        # random select an edge from self.all_edges
        parent_node, child_node = random.choice(self.all_edges)
        sibling_node = self._get_sibling(parent_node, child_node)
        
        p_con, p_def = self._get_node_info(parent_node)
        s_con, s_def = self._get_node_info(sibling_node)
        c_con, c_def = self._get_node_info(child_node)
        
        # Tokenize 所有组件
        # P
        t_p_con = self.tokenizer(p_con, add_special_tokens=False)['input_ids']
        t_p_def = self.tokenizer(p_def, add_special_tokens=False)['input_ids']
        # C
        t_c_con = self.tokenizer(c_con, add_special_tokens=False)['input_ids']
        t_c_def = self.tokenizer(c_def, add_special_tokens=False)['input_ids']
        # S
        t_s_con = self.tokenizer(s_con, add_special_tokens=False)['input_ids']
        t_s_def = self.tokenizer(s_def, add_special_tokens=False)['input_ids']
        
        # ===== Definition-level dropout =====
        t_p_def = self._maybe_drop_definition(t_p_def)
        t_c_def = self._maybe_drop_definition(t_c_def)
        t_s_def = self._maybe_drop_definition(t_s_def)

        # ===== Token-level masking =====
        t_p_def = self._mask_definition_tokens(t_p_def)
        t_c_def = self._mask_definition_tokens(t_c_def)
        t_s_def = self._mask_definition_tokens(t_s_def)
        
        # 空间分配 (非常拥挤！)
        # 总共有 3对 (Definition, Image_Text)
        avail_c = self.task_c_space - self.num_image_tokens
        
        # 策略：
        # 1. Concept 名字通常很短，尽量保留
        # 2. Definition 很长，需要大幅截断
        
        len_cons = len(t_p_con) + len(t_c_con) + len(t_s_con)
        remaining = avail_c - len_cons
        if remaining < 0: remaining = 0 # 极端情况
        
        # 平均分配剩余空间给 Definition
        limit_def = remaining // 3
        
        tr_p_def = self._truncate_tokens(t_p_def, limit_def)
        tr_c_def = self._truncate_tokens(t_c_def, limit_def)
        # 剩下的给 Sibling
        remain_s = remaining - len(tr_p_def) - len(tr_c_def)
        tr_s_def = self._truncate_tokens(t_s_def, remain_s)
        
        # 构建序列片段
        # Segments: 
        # 0: "Parent... Definition: \""
        # 1: "\", Image: "
        # 2: "; Child... Definition: \""
        # 3: "\", Image: "
        # 4: "; Sibling... Definition: \""
        # 5: "\", Image: "
        # 6: ";" (结尾)
        
        seg = self.task_c_segments
        
        # Teacher (全文本)
        task_c_teacher_ids = (
            [self.cls_token_id] +
            seg[0] + tr_p_def + seg[1] + t_p_con + 
            seg[2] + tr_c_def + seg[3] + t_c_con +
            seg[4] + tr_s_def + seg[5] + t_s_con +
            seg[6] + [self.sep_token_id]
        )
        
        # image_p = self._load_and_transform_image(self.taxon2id[parent_node], self.img_transforms)
        # image_c = self._load_and_transform_image(self.taxon2id[child_node], self.img_transforms)
        # image_s = self._load_and_transform_image(self.taxon2id[sibling_node], self.img_transforms)
        
        visual_p = self._load_feat(p_con)
        visual_c = self._load_feat(c_con)
        visual_s = self._load_feat(s_con)
        
        
        return {
            'visual_q': visual_q, 
            'visual_p': visual_p, 
            'visual_c': visual_c, 
            'visual_s': visual_s,
            
            'task_a_ids': task_a_teacher['input_ids'].squeeze(0),
            'task_a_mask': task_a_teacher['attention_mask'].squeeze(0),
            'task_a_prefix': torch.tensor(task_a_prefix, dtype=torch.long),
            
            'task_b_teacher_ids': torch.tensor(task_b_teacher_ids, dtype=torch.long),
            'task_b_prefix': torch.tensor(task_b_prefix, dtype=torch.long),
            'task_b_suffix': torch.tensor(task_b_suffix, dtype=torch.long),
            
            'task_c_teacher_ids': torch.tensor(task_c_teacher_ids, dtype=torch.long),
            
            'txt_p': torch.tensor(t_p_con, dtype=torch.long),
            'txt_c': torch.tensor(t_c_con, dtype=torch.long),
            'txt_s': torch.tensor(t_s_con, dtype=torch.long),
            'seg_p_def': torch.tensor(seg[0] + tr_p_def + seg[1], dtype=torch.long),
            'seg_c_def': torch.tensor(seg[2] + tr_c_def + seg[3], dtype=torch.long),
            'seg_s_def': torch.tensor(seg[4] + tr_s_def + seg[5], dtype=torch.long),
            'seg_end': torch.tensor(seg[6] + [self.sep_token_id], dtype=torch.long),
            
        }
        