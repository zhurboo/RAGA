import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import spmm
from torch_geometric.utils import softmax, degree

  
class GCN(nn.Module):
    def __init__(self, hidden):
        super(GCN, self).__init__()

    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        deg = degree(edge_index_i, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[edge_index_j]*deg_inv_sqrt[edge_index_i]
        x = F.relu(spmm(edge_index[[1, 0]], norm, x.size(0), x.size(0), x))
        return x

    
class Highway(nn.Module):
    def __init__(self, x_hidden):
        super(Highway, self).__init__()
        self.lin = nn.Linear(x_hidden, x_hidden)

    def forward(self, x1, x2):
        gate = torch.sigmoid(self.lin(x1))
        x = torch.mul(gate, x2)+torch.mul(1-gate, x1)
        return x


class GAT_E_to_R(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_E_to_R, self).__init__()
        self.a_h1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_h2 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t1 = nn.Linear(r_hidden, 1, bias=False)
        self.a_t2 = nn.Linear(r_hidden, 1, bias=False)
        self.w_h = nn.Linear(e_hidden, r_hidden, bias=False)
        self.w_t = nn.Linear(e_hidden, r_hidden, bias=False)
        
    def forward(self, x_e, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        x_r_h = self.w_h(x_e)
        x_r_t = self.w_t(x_e)
        
        e1 = self.a_h1(x_r_h).squeeze()[edge_index_h]+self.a_h2(x_r_t).squeeze()[edge_index_t]
        e2 = self.a_t1(x_r_h).squeeze()[edge_index_h]+self.a_t2(x_r_t).squeeze()[edge_index_t]
        
        alpha = softmax(F.leaky_relu(e1).float(), rel)
        x_r_h = spmm(torch.cat([rel.view(1, -1), edge_index_h.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_h)
        
        alpha = softmax(F.leaky_relu(e2).float(), rel)
        x_r_t = spmm(torch.cat([rel.view(1, -1), edge_index_t.view(1, -1)], dim=0), alpha, rel.max()+1, x_e.size(0), x_r_t)
        x_r = x_r_h+x_r_t
        return x_r

    
class GAT_R_to_E(nn.Module):
    def __init__(self, e_hidden, r_hidden):
        super(GAT_R_to_E, self).__init__()
        self.a_h = nn.Linear(e_hidden, 1, bias=False)
        self.a_t = nn.Linear(e_hidden, 1, bias=False)
        self.a_r = nn.Linear(r_hidden, 1, bias=False)
        
    def forward(self, x_e, x_r, edge_index, rel):
        edge_index_h, edge_index_t = edge_index
        e_h = self.a_h(x_e).squeeze()[edge_index_h]
        e_t = self.a_t(x_e).squeeze()[edge_index_t]
        e_r = self.a_r(x_r).squeeze()[rel]
        alpha = softmax(F.leaky_relu(e_h+e_r).float(), edge_index_h)
        x_e_h = spmm(torch.cat([edge_index_h.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
        alpha = softmax(F.leaky_relu(e_t+e_r).float(), edge_index_t)
        x_e_t = spmm(torch.cat([edge_index_t.view(1, -1), rel.view(1, -1)], dim=0), alpha, x_e.size(0), x_r.size(0), x_r)
        x = torch.cat([x_e_h, x_e_t], dim=1)
        return x
    

class GAT(nn.Module):
    def __init__(self, hidden):
        super(GAT, self).__init__()
        self.a_i = nn.Linear(hidden, 1, bias=False)
        self.a_j = nn.Linear(hidden, 1, bias=False)
        self.a_r = nn.Linear(hidden, 1, bias=False)
        
    def forward(self, x, edge_index):
        edge_index_j, edge_index_i = edge_index
        e_i = self.a_i(x).squeeze()[edge_index_i]
        e_j = self.a_j(x).squeeze()[edge_index_j]
        e = e_i+e_j
        alpha = softmax(F.leaky_relu(e).float(), edge_index_i)
        x = F.relu(spmm(edge_index[[1, 0]], alpha, x.size(0), x.size(0), x))
        return x
    
    
class RAGA(nn.Module):
    def __init__(self, e_hidden=300, r_hidden=100):
        super(RAGA, self).__init__()
        self.gcn1 = GCN(e_hidden)
        self.highway1 = Highway(e_hidden)
        self.gcn2 = GCN(e_hidden)
        self.highway2 = Highway(e_hidden)
        self.gat_e_to_r = GAT_E_to_R(e_hidden, r_hidden)
        self.gat_r_to_e = GAT_R_to_E(e_hidden, r_hidden)
        self.gat = GAT(e_hidden+2*r_hidden)

    def forward(self, x_e, edge_index, rel, edge_index_all, rel_all):
        x_e = self.highway1(x_e, self.gcn1(x_e, edge_index_all))
        x_e = self.highway2(x_e, self.gcn2(x_e, edge_index_all))
        x_r = self.gat_e_to_r(x_e, edge_index, rel)
        x_e = torch.cat([x_e, self.gat_r_to_e(x_e, x_r, edge_index, rel)], dim=1)
        x_e = torch.cat([x_e, self.gat(x_e, edge_index_all)], dim=1)
        return x_e
