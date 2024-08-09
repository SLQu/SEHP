import torch
import numpy as np
import torch_geometric


def reverse_edge(source_to_target):
    
    if type(source_to_target) is not torch.Tensor and type(source_to_target) is not torch_geometric.edge_index.EdgeIndex :
        print(type(source_to_target))
        print(source_to_target)
        raise TypeError('source_to_target must be a tensor')
    
    if source_to_target.shape[0] != 2:
        raise ValueError('source_to_target must be a tensor with shape (2,num_edges). And now it is {}'.format(source_to_target.shape))
    
    device = source_to_target.device
    
    return torch.cat([source_to_target[1],source_to_target[0]]).reshape(2,-1).to(device)


def convert_bipartite_to_list_hyperedges(edge_index):  
    '''
    input: is a tensor with shape (2,num_edges)
    output: is a list of tensors with shape (num_nodes_in_hyperedge)
    
    example: edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                                        [0, 1, 0, 1, 2, 3, 2, 3, 4, 5,6]])
            
            output = [tensor([0, 1]), 
                        tensor([0, 1]), 
                        tensor([2, 3]), 
                        tensor([2, 3]), 
                        tensor([4, 5, 6 ])]
    '''
      
    unique_edges, inverse_indices = torch.unique(edge_index[1], return_inverse=True)
    hyperedges = torch.split(edge_index[0], torch.bincount(inverse_indices).tolist())

    return hyperedges

    # set_format_of_hyperedges = convert_bipartite_to_list_hyperedges(edge_index)
    # edge_list = [ e.tolist() for e, _mask_ in zip(set_format_of_hyperedges, mask) if _mask_]

def get_specific_hyperedges(edge_index, mask, device) :
    '''
    Get specific hyperedges based on the given edge index and mask.

    Args:
        edge_index (torch.Tensor): A tensor with shape (2, num_edges) representing the edge indices.
        mask (torch.Tensor): A tensor with shape (num_edges) representing the mask.

    Returns:
        list: A list of tensors with shape (num_nodes_in_hyperedge) representing the specific hyperedges.

    Example:
        edge_index = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4],
                                   [0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 6]])

        mask = torch.tensor([True, False, True, False, True])

        specific_hyperedges = torch.tensor([[0, 0, 2, 2, 4, 4, 4],
                                            [0, 1, 2, 3, 4, 5, 6]])
    '''

    set_format_of_hyperedges = convert_bipartite_to_list_hyperedges(edge_index)
    edge_list = [ e.tolist() for e, _mask_ in zip(set_format_of_hyperedges, mask) if _mask_]

    node_idx = []
    e_idx = []
    for eid,nodes in enumerate(edge_list):
        node_idx.extend(nodes)
        e_idx.extend([eid]*len(nodes))



    return torch.LongTensor(node_idx).to(device),torch.LongTensor(e_idx).to(device)
    

import random

def negative_generate_by_positive(num_nodes, num_negative, edge_index, device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')):

    if type(edge_index) is tuple:
        edge_index = torch.cat(edge_index).view(2,-1)

    NodeEdgePair  = edge_index.T
    EdgeNodePair = NodeEdgePair[:, [1, 0]]

    candidated = list(range(num_nodes))

    edeg_id = set()
    neg_node_ids = []
    neg_edge_ids = []
    start_ids = 0
    for j in range(num_negative):

        for i,_ in EdgeNodePair.tolist():
            neg_id = random.sample(candidated, 1)[0]
            edeg_id.add(i)
            neg_node_ids.append(neg_id)
            neg_edge_ids.append(len(edeg_id) + start_ids)
        
        start_ids += len(edeg_id)
        edeg_id = set()

    return torch.LongTensor((neg_node_ids,neg_edge_ids)).to(device)






def format_and_sort_list(input_list):
    '''
    convert a list of int to a string with sorted and unique elements
    
    example: input_list = [1,2,3,4,5,6,7,8,9,10,1,2,3,4]
            output = '1-2-3-4-5-6-7-8-9-10'
    '''
    
    if type(input_list) is set:
        input_list = list(input_list)
    
    if type(input_list) is not list and type(input_list) is not tuple:
        if type(input_list) is torch.Tensor:
            if input_list.device == torch.device('cuda'):
                input_list = input_list.cpu()
            input_list = input_list.tolist()
        
        else:
            print(type(input_list))
            print(input_list)
            raise TypeError('input_list must be a list')
    
    
    if type(input_list[0]) is not int and type(input_list[0]) is not np.int64:
        print(type(input_list[0]))
        raise TypeError('input_list must be a list of int')
    unique_sorted_list = sorted(set(input_list))
    string_list = [str(item) for item in unique_sorted_list]
    result_string = "-".join(string_list)
    return result_string