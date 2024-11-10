import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import Uniform
from hgat_models import *
class HGATLinkPrediction(nn.model):
    """
        Parameters
        ----------
        num_layers: int
            the number of layers we used in the computing
        in_dim: int
            the input dimension
        hidden_dim: int
            the hidden dimension
        num_classes: int
            the number of the output classes
        ntypes: list
            the list of the node type in the graph
        negative_slope: float
            the negative slope used in the LeakyReLU
"""
    def __init__(self, ntypes, num_classes=16, num_layers=2, hidden_dim=64, negative_slope=0.2):
        super(HGATLinkPrediction, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu

        self.HGAT = HGAT(ntypes, num_classes, num_layers, hidden_dim, negative_slope)

    def forward(self, hg, h_dict, link_subgraph):
        """
        The forward part of the HGAT.
        
        Parameters
        ----------
        hg : dgl.Heterograph
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types
        link_subgraph: dgl.Heterograph
            the prediction graph only contains the edges of the target link

            
        Returns
        -------
        score: th.Tensor
            the prediction of the edges in link_subgraph
        """
        x = self.HGAT.forward(hg, h_dict)
        with link_subgraph.local_scope( ):
            for ntype in link_subgraph.ntypes:
                link_subgraph.nodes[ntype].data['x'] = x[ntype]
                for etype in link_subgraph.canonical_etypes:
                    link_subgraph.apply_edges(
                        dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
            score = link_subgraph.edata['score']
            if isinstance(score, dict):
                result = []
                for _, value in score.items( ):
                    result.append(value)
                score = torch.cat(result)
            return score.squeeze( )