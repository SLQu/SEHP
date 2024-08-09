
import torch




def hedge_add(_edge_index_, _seed_e_ids_, max_node = 1000):
    '''
    
    Args:
        _edge_index_ (LongTensor): Hyperedge tensor
        with shape :obj:`[2, num_edges*num_nodes_per_edge]`, where
        :obj:`edge_index[1]` denotes the hyperedge index and
        :obj:`edge_index[0]` denotes the node indices that are connected
        by the hyperedge.
        _seed_e_ids_ (LongTensor): The hyperedges to keep.
        _m_ (int): The max number nodes of sub-hypergraph, *i.e.*

    '''

    def continue_flag(_node_ids_):
        return _node_ids_.size(0) < max_node 
    
    def remove_element(A,B):
        mask = ~torch.isin(A, B)
        stay = A[mask]
        return stay
    
    def remove_hyperedge_below_threshold(_edge_index_, _threshold_ = 1):
        
        unique_elements, counts = torch.unique(_edge_index_[1], return_counts=True)
        elements_to_remove = unique_elements[counts == 1]

        mask = ~torch.isin(_edge_index_[1], elements_to_remove)

        _edge_index_ = _edge_index_[:, mask]

        return mask, _edge_index_

    def final_out(final_mask, _threshold_=1):
        final_edge_index = _edge_index_[:, final_mask]
        _mask_index_ = torch.where(final_mask)[0]

        _idx_, final_edge_index = remove_hyperedge_below_threshold(final_edge_index, _threshold_)
        _mask_index_ = _mask_index_[_idx_]

        return  final_edge_index, _mask_index_
    

    _mask_e = torch.isin(_edge_index_[1], _seed_e_ids_)
    _seed_node_ids_ = _edge_index_[0][_mask_e].unique()
    _mask_n = torch.isin(_edge_index_[0], _seed_node_ids_)

    _new_n_ids_ = _edge_index_[0][_mask_n]
    _new_n_ids_unique = _new_n_ids_.unique()


    if continue_flag(_new_n_ids_unique):

        final_edge_index, _mask_index_ = final_out(_mask_n, _edge_index_)
        return  final_edge_index, _mask_index_, True

    else:
        _new_e_ids_ = _edge_index_[1][_mask_n]
        _new_e_ids_unique = _new_e_ids_.unique()
        tem_new_e_ids = remove_element(_new_e_ids_unique, _seed_e_ids_)
        total_len = tem_new_e_ids.size(0)
        stop_point = 10

        while not continue_flag(_new_n_ids_unique):
            final_e_ids = torch.cat([_seed_e_ids_, tem_new_e_ids[: max(0, total_len-stop_point)]])
            final_mask = torch.isin(_edge_index_[1], final_e_ids)
            final_n_ids = _edge_index_[0][final_mask]
            _new_n_ids_unique = final_n_ids.unique()
            
            if stop_point > total_len:
                break

            stop_point *= 5

        final_edge_index, _mask_index_ = final_out(final_mask, _edge_index_)

        return final_edge_index, _mask_index_, False

def sub_hypergraph_sample(edge_index, seed_e_ids, max_node=10000, hop_num=1):

    
    def id_map(_ids_):
        unique_elements, inverse_indices = torch.unique(_ids_, return_inverse=True)
        mapping_tensor = torch.arange(len(unique_elements)).to(_ids_.device)
        mapped_ids_ = mapping_tensor[inverse_indices]
        return mapped_ids_, unique_elements
    
    if edge_index.device != seed_e_ids.device:
        seed_e_ids = seed_e_ids.to(edge_index.device)
        # edge_index = edge_index.to(seed_e_ids.device)

    for _i_ in range(hop_num):
        new_edge_index, original_edge_ids, continue_flag = hedge_add(edge_index, seed_e_ids, max_node)

        if not continue_flag:
            break
        seed_e_ids = new_edge_index[1].unique()


    sorted_col, indices = torch.sort(new_edge_index[1])
    sorted_row = new_edge_index[0][indices]
    sorted_edge = original_edge_ids[indices]

    
    col, mapped_col_unique_ids  = id_map(sorted_col)
    row, mapped_row_unique_ids  = id_map(sorted_row)

    node = {'node':mapped_row_unique_ids, 'hyperedge':mapped_col_unique_ids}

    return node, {'node__in__hyperedge':row}, {'node__in__hyperedge':col}, {'node__in__hyperedge':sorted_edge} 



if __name__ == '__main__':
    print('subgraph')


    print(f"---------------    hyper subgraph ------------ ")
    edge_index = torch.tensor([[0, 3, 4, 5, 3, 6, 3, 4, 5, 2, 5, 6, 7, 8], 
                               [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4]])
    

    o_e_ids = torch.tensor([0])
    node, row, col, edge  = sub_hypergraph_sample(edge_index, o_e_ids, max_node=10000, hop_num=3)
    

    n_ids = row.unique()
    e_ids = col.unique()
    print(f"edge_index: {edge_index}")  
    print(f'o_e_ids: {o_e_ids}')
    print('new_edge_index')
    print(f"n_ids: {n_ids}")
    print(f"e_ids: {e_ids}")
    # print(f"mask: {mask_n}")
    print(f"mask_index: {edge}")
    # print(f"continue_flag: {continue_flag}")


    # res = hyper_subgraph(subset, edge_index, edge_attr, return_edge_mask=False)
    # print('original edge_index')
    # print(edge_index)
    # print(f'subset: {subset}')
    # print(res)



