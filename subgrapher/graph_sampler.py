import numpy as np
import random
from collections import defaultdict

def sample_neg(adj_list, triplets, num_neg_samples_per_link=1, max_size=100000, constrained_neg_prob=0.0):
    """
    Sample negative examples for link prediction.
    
    Args:
        adj_list: List of adjacency matrices for each relation
        triplets: Positive triplets
        num_neg_samples_per_link: Number of negative samples per positive triplet
        max_size: Maximum number of negative samples to generate
        constrained_neg_prob: Probability of sampling constrained negative examples
    
    Returns:
        pos_triplets: Positive triplets
        neg_triplets: Negative triplets
    """
    pos_triplets = triplets
    neg_triplets = []
    
    # Get all entities
    all_entities = set()
    for adj in adj_list:
        all_entities.update(adj.nonzero()[0])
        all_entities.update(adj.nonzero()[1])
    all_entities = list(all_entities)
    
    # Create entity-relation mapping
    entity_relations = defaultdict(set)
    for i, adj in enumerate(adj_list):
        for e1, e2 in zip(*adj.nonzero()):
            entity_relations[e1].add(i)
            entity_relations[e2].add(i)
    
    # Sample negative examples
    for triplet in pos_triplets:
        e1, e2, r = triplet
        for _ in range(num_neg_samples_per_link):
            if random.random() < constrained_neg_prob:
                # Sample from entities that have the same relation
                candidates = list(entity_relations[e1] & entity_relations[e2])
                if candidates:
                    r_neg = random.choice(candidates)
                    neg_triplets.append([e1, e2, r_neg])
                else:
                    # Fall back to random sampling
                    r_neg = random.choice(range(len(adj_list)))
                    neg_triplets.append([e1, e2, r_neg])
            else:
                # Random sampling
                r_neg = random.choice(range(len(adj_list)))
                neg_triplets.append([e1, e2, r_neg])
    
    # Limit the number of negative samples
    if len(neg_triplets) > max_size:
        neg_triplets = random.sample(neg_triplets, max_size)
    
    return np.array(pos_triplets), np.array(neg_triplets)

def get_adjacency_list(triplets, num_entities, num_relations):
    """
    Get adjacency list from triplets.
    
    Args:
        triplets: List of triplets
        num_entities: Number of entities
        num_relations: Number of relations
    
    Returns:
        adj_list: List of adjacency matrices
    """
    adj_list = []
    for r in range(num_relations):
        idx = np.argwhere(triplets[:, 2] == r)
        adj_list.append(np.zeros((num_entities, num_entities)))
        adj_list[r][triplets[idx, 0], triplets[idx, 1]] = 1
    return adj_list 