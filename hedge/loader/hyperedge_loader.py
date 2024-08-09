


from typing import Union,List, Dict
from torch_geometric.data import Data, HeteroData
from torch_geometric.typing import OptTensor,EdgeType,NodeType
import torch

from torch_geometric.sampler import BaseSampler,HeteroSamplerOutput,NodeSamplerInput
from torch_geometric.typing import NodeType

from torch_geometric.sampler.utils import remap_keys, to_hetero_csc
from torch_geometric.typing import (
    WITH_TORCH_SPARSE
)

from .hypergraph_sample import sub_hypergraph_sample


class HyperEdgeSamplerV2(BaseSampler):
    r"""  
    wrong comments
    An implementation of an in-memory heterogeneous layer-wise sampler
    user by :class:`~torch_geometric.loader.HGTLoader`."""
    
    
    def __init__(
        self,
        data: HeteroData,
        num_samples: Union[List[int], Dict[NodeType, List[int]]],
        is_sorted: bool = False,
        share_memory: bool = False,
        num_hops: int = 1,
        max_node: int = 10000,
    ):
        if not WITH_TORCH_SPARSE:
            raise ImportError(
                f"'{self.__class__.__name__}' requires 'torch-sparse'")

        if isinstance(data, Data) or isinstance(data, tuple):
            raise NotImplementedError(
                f'{self.__class__.__name__} does not support a data object of '
                f'type {type(data)}.')

        if isinstance(num_samples, (list, tuple)):
            num_samples = {key: num_samples for key in data.node_types}

        self.data = data
        self.node_types, self.edge_types = data.metadata()
        self.num_samples = num_samples
        self.num_hops = num_hops #max([len(v) for v in num_samples.values()])
        self.max_node = max_node

        # Conversion to/from C++ string type (see `NeighborSampler`):
        self.to_rel_type = {k: '__'.join(k) for k in self.edge_types}
        self.to_edge_type = {v: k for k, v in self.to_rel_type.items()}

        # Convert the graph data into a suitable format for sampling:
        colptr_dict, row_dict, self.perm = to_hetero_csc(data, device='cpu', share_memory=share_memory, is_sorted=is_sorted)
        self.row_dict = remap_keys(row_dict, self.to_rel_type)
        self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> HeteroSamplerOutput:
        
        # print(f"inputs.node: {inputs.node}")
        # print(f"inputs.input_type: {inputs.input_type}")
        # print(f"self.colptr_dict: {self.colptr_dict}")
        # print(f"self.row_dict: {self.row_dict}")
        # print(f"self.num_samples: {self.num_samples}")
        # print(f"self.num_hops: {self.num_hops}")
        
        '''
        Given a hypergraph, which has m nodes and n hyperedges. we want to sample a subgraph from it, where the subgraph has m' nodes and n' hyperedges.

        When sampling the subgraph, we need to consider the following constraints:
        1. The number of nodes in the subgraph should be less than or equal to the number of nodes in the original hypergraph, i.e., m' <= m.
        2. The number of hyperedges in the subgraph should be less than or equal to the number of hyperedges in the original hypergraph, i.e., n' <= n.
        3. every hyperedge in the subgraph should have the all same nodes like when they in original hypergraph, i.e., for a hyperedge e in the subgraph, if a node n is in e, then n should be in e in the original hypergraph.
        4. At the beginning, we need to randomly select a set of hyperedges from the original hypergraph, and then we can sample the a subhypergraph based on the selected hyperedges.
        5. The sampling process should be iterative, i.e., given a seed set of hyperedges, we can construct a seed set of nodes, and then we can sample the next set of hyperedges based on the seed set of nodes, and so on.

        The sampling process can be divided into two steps:
        1. Sample a set of hyperedges from the original hypergraph.
        2. Sample a subhypergraph based on the selected hy
        
        '''

        # print(f"------   start  -------")

        node, row, col, edge = sub_hypergraph_sample(self.data['node','in','hyperedge'].edge_index, inputs.node, max_node=self.max_node, hop_num=self.num_hops)
        # print(f"node: {node}")
        # print(f"row: {row}  {row['node__in__hyperedge'].size()}")
        # print(f"col: {col} {col['node__in__hyperedge'].size()}")
        # print(f"edge: {edge}  {edge['node__in__hyperedge'].size()}")
        # print(f"------   end  -------")


        # node, row, col, edge = torch.ops.torch_sparse.hgt_sample(
        #     self.colptr_dict,
        #     self.row_dict,
        #     {inputs.input_type: inputs.node},
        #     self.num_samples,
        #     self.num_hops,
        # )

        # print(f"node: {node}")
        # print(f"row: {row}")
        # print(f"col: {col}")
        # print(f"edge: {edge}")
        # print(f"------   end  -------")
        return HeteroSamplerOutput(
            node=node,
            row=remap_keys(row, self.to_edge_type),
            col=remap_keys(col, self.to_edge_type),
            edge=remap_keys(edge, self.to_edge_type),
            batch=None,
            metadata=(inputs.input_id, inputs.time),
        )

    @property
    def edge_permutation(self) -> Union[OptTensor, Dict[EdgeType, OptTensor]]:
        return self.perm

    

