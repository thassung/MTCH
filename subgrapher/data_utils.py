import os
import numpy as np
from scipy.sparse import csc_matrix
import matplotlib.pyplot as plt
import json
import pickle
import logging

def plot_rel_dist(adj_list, filename):
    """Plot the distribution of relations in the knowledge graph."""
    rel_count = []
    for adj in adj_list:
        rel_count.append(adj.count_nonzero())

    fig = plt.figure(figsize=(12, 8))
    plt.plot(rel_count)
    plt.title('Relation Distribution')
    plt.xlabel('Relation ID')
    plt.ylabel('Number of Triplets')
    fig.savefig(filename, dpi=fig.dpi)

def process_files(files, saved_relation2id=None):
    """
    Process knowledge graph files into entity/relation mappings and adjacency lists.
    
    Args:
        files: Dictionary mapping file types to file paths
        saved_relation2id: Optional pre-existing relation2id mapping
    
    Returns:
        adj_list: List of adjacency matrices for each relation
        triplets: Dictionary of triplets for each split
        entity2id: Entity to ID mapping
        relation2id: Relation to ID mapping
        id2entity: ID to entity mapping
        id2relation: ID to relation mapping
    """
    entity2id = {}
    relation2id = {} if saved_relation2id is None else saved_relation2id

    triplets = {}
    ent = 0
    rel = 0

    for file_type, file_path in files.items():
        data = []
        with open(file_path) as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            if triplet[0] not in entity2id:
                entity2id[triplet[0]] = ent
                ent += 1
            if triplet[2] not in entity2id:
                entity2id[triplet[2]] = ent
                ent += 1
            if not saved_relation2id and triplet[1] not in relation2id:
                relation2id[triplet[1]] = rel
                rel += 1

            if triplet[1] in relation2id:
                data.append([entity2id[triplet[0]], entity2id[triplet[2]], relation2id[triplet[1]]])

        triplets[file_type] = np.array(data)

    id2entity = {v: k for k, v in entity2id.items()}
    id2relation = {v: k for k, v in relation2id.items()}

    # Construct adjacency matrices for each relation
    adj_list = []
    for i in range(len(relation2id)):
        idx = np.argwhere(triplets['train'][:, 2] == i)
        adj_list.append(csc_matrix(
            (np.ones(len(idx), dtype=np.uint8), 
             (triplets['train'][:, 0][idx].squeeze(1), 
              triplets['train'][:, 1][idx].squeeze(1))),
            shape=(len(entity2id), len(entity2id))
        ))

    return adj_list, triplets, entity2id, relation2id, id2entity, id2relation

def save_to_file(directory, file_name, triplets, id2entity, id2relation):
    """Save triplets to a file with entity and relation names."""
    file_path = os.path.join(directory, file_name)
    with open(file_path, "w") as f:
        for s, o, r in triplets:
            f.write('\t'.join([id2entity[s], id2relation[r], id2entity[o]]) + '\n')

def save_mappings(directory, entity2id, relation2id):
    """Save entity and relation mappings to JSON files."""
    with open(os.path.join(directory, 'entity2id.json'), 'w') as f:
        json.dump(entity2id, f)
    with open(os.path.join(directory, 'relation2id.json'), 'w') as f:
        json.dump(relation2id, f)

def load_mappings(directory):
    """Load entity and relation mappings from JSON files."""
    with open(os.path.join(directory, 'entity2id.json'), 'r') as f:
        entity2id = json.load(f)
    with open(os.path.join(directory, 'relation2id.json'), 'r') as f:
        relation2id = json.load(f)
    return entity2id, relation2id 