import random
import torch



def hyperedge_split(data, mode = 'random', ratio = [0.6, 0.2, 0.2]):
    """
    only split hyperedge for heterogeneous graph
    """
    
    if mode == 'random':
        train_mask, test_mask, validate_mask= __random_split__(data['hyperedge'].num_nodes, ratio, seed = None)
    elif mode == 'ordered':
        train_mask, test_mask, validate_mask= __ordered_split__(data['hyperedge'].num_nodes, ratio)
        
    data['hyperedge'].train_mask, data['hyperedge'].test_mask, data['hyperedge'].val_mask = train_mask, test_mask, validate_mask     
    return data       


        
def __random_split__(total_num, ratio = [0.6, 0.2, 0.2], seed=None):
    """
    Split a list of labels into train, test, and validate sets randomly.
    
    :param seed: Seed for randomization (optional).
    :return: Three lists of boolean values indicating whether each label belongs to each set.
    """
    
    if seed is not None:
        random.seed(seed)
        
    train_size = int(ratio[0] * total_num)
    test_size = int(ratio[1] * total_num)
    validate_size = total_num - train_size - test_size
    
    labels = list(range(total_num))
    # Shuffle the labels randomly
    random.shuffle(labels)
    
    # Create lists to store the split results
    train_set = labels[:train_size]
    test_set = labels[train_size:train_size + test_size]
    validate_set = labels[train_size + test_size:]
    
    # Create lists of boolean values indicating the split
    labels = list(range(total_num))
    train_mask =  torch.tensor([label in train_set for label in labels], dtype=torch.bool)
    test_mask =  torch.tensor([label in test_set for label in labels], dtype=torch.bool)
    validate_mask =  torch.tensor([label in validate_set for label in labels], dtype=torch.bool)

    return train_mask, test_mask, validate_mask

def __ordered_split__(total_num, ratio = [0.6, 0.2, 0.2]):
    train_end = int(total_num * ratio[0])
    test_end = train_end + int(total_num * ratio[1])

    train_mask = [True] * train_end + [False] * (total_num - train_end)
    test_mask = [False] * train_end + [True] * (test_end - train_end) + [False] * (total_num - test_end)
    validate_mask = [False] * test_end + [True] * (total_num - test_end)

    return train_mask, test_mask, validate_mask
    