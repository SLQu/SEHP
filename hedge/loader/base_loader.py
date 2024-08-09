from typing import Callable, Optional, Tuple, Union,List, Any
from torch_geometric.data import Data, FeatureStore, GraphStore, HeteroData
from torch_geometric.sampler import  NegativeSampling
from torch_geometric.typing import InputEdges, OptTensor,EdgeType,NodeType
from torch import Tensor
from torch.utils.data import DataLoader
import torch

from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor

from torch_geometric.data import FeatureStore, GraphStore, HeteroData
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler import BaseSampler,HeteroSamplerOutput,NodeSamplerInput
from torch_geometric.typing import NodeType

from torch_geometric.sampler.utils import remap_keys, to_hetero_csc
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.typing import (
    WITH_TORCH_SPARSE
)

from .hyperedge_loader import HyperEdgeSamplerV2

class HyperEdgeSampler(BaseSampler):
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
        colptr_dict, row_dict, self.perm = to_hetero_csc(
            data, device='cpu', share_memory=share_memory, is_sorted=is_sorted)
        self.row_dict = remap_keys(row_dict, self.to_rel_type)
        self.colptr_dict = remap_keys(colptr_dict, self.to_rel_type)

    def sample_from_nodes(
        self,
        inputs: NodeSamplerInput,
        **kwargs,
    ) -> HeteroSamplerOutput:
        
        

        # # print(f"inputs.node: {inputs.node}")
        # print(f"inputs.input_type: {inputs.input_type}")
        # # print(f"self.colptr_dict: {self.colptr_dict}")
        # print(f"self.row_dict: {self.row_dict}")

        # print(f"self.num_samples: {self.num_samples}")
        # print(f"self.num_hops: {self.num_hops}")
        
        # print(f"------   start  -------")
        node, row, col, edge = torch.ops.torch_sparse.hgt_sample(
            self.colptr_dict,
            self.row_dict,
            {inputs.input_type: inputs.node},
            self.num_samples,
            self.num_hops,
        )

        # print(f"node: {node}")
        # print(f"row: {row}")
        # print(f"col: {col}")
        # print(f"edge: {edge}")
        # print(f"------   end  -------")

        # print(f"remap_keys(row, self.to_edge_type) {remap_keys(row, self.to_edge_type)}")
        # print(f"remap_keys(col, self.to_edge_type) {remap_keys(col, self.to_edge_type)}")
        # print(f"remap_keys(edge, self.to_edge_type) {remap_keys(edge, self.to_edge_type)}")
        
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

class BaseLoader(NodeLoader):
    
    def __init__(self,
                data: Union[HeteroData, Tuple[FeatureStore, GraphStore]],
                num_samples: Union[List[int], Dict[NodeType, List[int]]],
                input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
                is_sorted: bool = False,
                transform: Optional[Callable] = None,
                transform_sampler_output: Optional[Callable] = None,
                filter_per_worker: bool = False,        
                num_hops: int = 1,
                sampler: str = 'HyperEdgeSamplerV2',   # HyperEdgeSampler     HyperEdgeSamplerV2
                max_node: int = 10000,
                **kwargs):
        

        ##num_samples = {key: [20000000] for key in data.node_types}

        if sampler == 'HyperEdgeSampler':
            sampler_class = HyperEdgeSampler
        elif sampler == 'HyperEdgeSamplerV2':
            sampler_class = HyperEdgeSamplerV2
        else:
            raise NotImplementedError(f"Sampler {sampler} is not implemented")
        
        hedge_sampler = sampler_class(
            data,
            num_samples=num_samples,
            is_sorted=is_sorted,
            share_memory=kwargs.get('num_workers', 0) > 0,
            num_hops= num_hops,
            max_node = max_node
            
        )

        super().__init__(
            data=data,
            node_sampler=hedge_sampler,
            input_nodes=input_nodes,
            transform=transform,
            transform_sampler_output=transform_sampler_output,
            filter_per_worker=filter_per_worker,
            **kwargs,
        )
        

    
    

