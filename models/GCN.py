import dgl
import torch
import torch.nn.functional as F


# class GCN(torch.nn.Module):
#     def __init__(self, input_features, hidden_size, num_classes, dropout=0.5, num_gcn=0):
#         super().__init__()
#         self.dropout = torch.nn.Dropout(dropout)
#         self.num_gcn = num_gcn
#         self.input_layer = torch.nn.Linear(input_features, hidden_size)
#         self.conv1 = dgl.nn.GraphConv(hidden_size, hidden_size)
#         self.conv2 = dgl.nn.GraphConv(hidden_size, hidden_size)
#         self.output_layer = torch.nn.Linear(hidden_size, num_classes)
#         self.input_bias = torch.nn.Parameter(torch.zeros(hidden_size))
#         torch.nn.init.xavier_uniform_(self.input_layer.weight)
#         torch.nn.init.xavier_uniform_(self.output_layer.weight)
        
        
#     def forward(self, blocks, x):
#         outputs = self.dropout(F.relu(self.input_layer(x)) + self.input_bias)
#         outputs = self.conv1(blocks[0], outputs)
#         outputs = self.conv2(blocks[1], outputs)
#         outputs = self.output_layer(outputs)
#         return outputs

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv

class GCN(nn.Module):

    def __init__(self, input_size, labels_num, hidden_size=128, num_gcn=2, dropout=0.5, residual=True, **kwargs):
        super().__init__()
        self.labels_num = labels_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_gcn = num_gcn
        self.dropout = nn.Dropout(dropout)
        self.residual = residual
        self.input_bias = torch.nn.Parameter(torch.zeros(hidden_size))
        
        self.input = nn.Linear(input_size, hidden_size)
        
        self.ppi_linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_gcn)])

        self.output = nn.Linear(hidden_size, labels_num)
        
        nn.init.xavier_uniform_(self.input.weight)
        nn.init.xavier_uniform_(self.output.weight)
        for ppi_linear in self.ppi_linears:
            nn.init.xavier_uniform_(ppi_linear.weight)
        
    def forward(self, blocks, inputs):
        h = self.dropout(F.relu(self.input(inputs) + self.input_bias))
        # h = self.dropout(F.relu(self.input(blocks[0].srcdata['h']) + self.input_bias))
        blocks[0].srcdata['h'] = h
        for i, ppi_linear in enumerate(self.ppi_linears):
            with blocks[i].local_scope():
                if self.residual:
                    m_res = dgl.function.u_mul_e('h', 'self', out='m_res')
                    res = dgl.function.sum(msg='m_res', out='res')
                    blocks[i].update_all(m_res, res)
                ppi_m_out = dgl.function.u_mul_e('h', 'ppi', out='ppi_m_out')
                ppi_out = dgl.function.sum(msg='ppi_m_out', out='ppi_out')
                blocks[i].update_all(ppi_m_out, ppi_out)
                h = blocks[i].dstdata['ppi_out']
                h = self.dropout(F.relu(ppi_linear(h)))
                h = h + blocks[i].dstdata['res']
                if i != self.num_gcn-1:
                    blocks[i+1].srcdata['h'] = h
        output = self.output(h)
        return output