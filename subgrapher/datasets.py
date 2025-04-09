import os
import timeit
import logging
import lmdb
import numpy as np
import json
import pickle
# import dgl
import torch
from torch.utils.data import Dataset
from .data_utils import process_files, save_to_file, plot_rel_dist
from .graph_sampler import sample_neg

class KnowledgeGraphDataset:
    """Base class for knowledge graph datasets."""
    
    def __init__(self, data_dir, dataset_name):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.entity2id = None
        self.relation2id = None
        self.id2entity = None
        self.id2relation = None
        self.triplets = None
        self.adj_list = None
        
    def load(self):
        """Load the dataset from files."""
        files = {
            'train': os.path.join(self.data_dir, 'train.txt'),
            'valid': os.path.join(self.data_dir, 'valid.txt'),
            'test': os.path.join(self.data_dir, 'test.txt')
        }
        
        self.adj_list, self.triplets, self.entity2id, self.relation2id, self.id2entity, self.id2relation = process_files(files)
        
        # Save mappings
        os.makedirs(os.path.join(self.data_dir, 'processed'), exist_ok=True)
        with open(os.path.join(self.data_dir, 'processed', 'entity2id.json'), 'w') as f:
            json.dump(self.entity2id, f)
        with open(os.path.join(self.data_dir, 'processed', 'relation2id.json'), 'w') as f:
            json.dump(self.relation2id, f)
            
        logging.info(f"Loaded {self.dataset_name} dataset with {len(self.entity2id)} entities and {len(self.relation2id)} relations")

class SubgraphDataset(Dataset):
    """Dataset for subgraph extraction and classification."""
    
    def __init__(self, data_dir, dataset_name, split='train', num_neg_samples=1, max_links=100000):
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.split = split
        self.num_neg_samples = num_neg_samples
        self.max_links = max_links
        
        # Load mappings
        with open(os.path.join(data_dir, 'processed', 'entity2id.json'), 'r') as f:
            self.entity2id = json.load(f)
        with open(os.path.join(data_dir, 'processed', 'relation2id.json'), 'r') as f:
            self.relation2id = json.load(f)
            
        self.id2entity = {int(v): k for k, v in self.entity2id.items()}
        self.id2relation = {int(v): k for k, v in self.relation2id.items()}
        
        # Load triplets
        with open(os.path.join(data_dir, f'{split}.txt'), 'r') as f:
            self.triplets = np.array([[self.entity2id[t[0]], self.entity2id[t[2]], self.relation2id[t[1]]] 
                                    for t in [line.split() for line in f.read().split('\n')[:-1]]])
        
        # Sample negative examples
        self.pos_triplets, self.neg_triplets = sample_neg(
            self.adj_list, 
            self.triplets, 
            self.num_neg_samples,
            max_size=self.max_links
        )
        
    def __len__(self):
        return len(self.pos_triplets)
    
    def __getitem__(self, idx):
        pos_triplet = self.pos_triplets[idx]
        neg_triplets = self.neg_triplets[idx]
        
        return {
            'pos': pos_triplet,
            'neg': neg_triplets
        }
        
    def get_entity_embedding(self, entity_id):
        """Get entity embedding if available."""
        # This can be extended to load pre-trained embeddings
        return None
        
    def get_relation_embedding(self, relation_id):
        """Get relation embedding if available."""
        # This can be extended to load pre-trained embeddings
        return None

def generate_subgraph_datasets(params, splits=['train', 'valid'], saved_relation2id=None):
    """
    Generate subgraph datasets for training and evaluation.
    
    Args:
        params: Parameters object containing dataset configuration
        splits: List of splits to generate
        saved_relation2id: Optional pre-existing relation2id mapping
    """
    testing = 'test' in splits
    
    # Process files and get mappings
    files = {split: os.path.join(params.data_dir, f'{split}.txt') for split in splits}
    adj_list, triplets, entity2id, relation2id, id2entity, id2relation = process_files(files, saved_relation2id)
    
    # Save relation2id if not testing
    if not testing:
        with open(os.path.join(params.data_dir, 'processed', 'relation2id.json'), 'w') as f:
            json.dump(relation2id, f)
    
    # Sample negative examples for each split
    graphs = {}
    for split_name in splits:
        graphs[split_name] = {'triplets': triplets[split_name], 'max_size': params.max_links}
        graphs[split_name]['pos'], graphs[split_name]['neg'] = sample_neg(
            adj_list,
            graphs[split_name]['triplets'],
            params.num_neg_samples_per_link,
            max_size=graphs[split_name]['max_size']
        )
    
    # Save negative examples for testing
    if testing:
        save_to_file(
            os.path.join(params.data_dir, 'processed'),
            f'neg_{params.test_file}.txt',
            graphs['test']['neg'],
            id2entity,
            id2relation
        )
    
    return graphs 