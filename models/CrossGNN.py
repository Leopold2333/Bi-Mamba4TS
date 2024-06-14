import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from layers.RevIn import RevIN

def FFT_for_Period(x, k=4):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1) 
    frequency_list[0] = float('-inf')
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period=[1]
    for top in top_list:
        #print(x.shape[1],top,x.shape[1] / top)
        period = np.concatenate((period,[math.ceil(x.shape[1] / top)])) #    
    return period, abs(xf).mean(-1)[:, top_list] #

class moving_avg(nn.Module):
    """
    Moving average block
    """
    def __init__(self, kernel_size):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=kernel_size, padding=0) 

    def forward(self, x):
        # batch seq_len channel
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x #batch seq_len channel

class multi_scale_data(nn.Module):
    def __init__(self, kernel_size, return_len):
        super(multi_scale_data, self).__init__()
        self.kernel_size = kernel_size
        self.max_len = return_len   # equals to 2xseq_len by default
        self.moving_avg = [moving_avg(kernel) for kernel in kernel_size]
    def forward(self, x):
        # x: [B, L, C]
        different_scale_x = []
        for func in self.moving_avg:
            moving_avg = func(x)
            different_scale_x.append(moving_avg)
        # x: [B, L, C] -> [B, L', C]
        multi_scale_x = torch.cat(different_scale_x, dim=1)
        # ensure fixed shape: [batch, max_len, variables]
        if multi_scale_x.shape[1] < self.max_len: # padding to resize L' to max_len
            padding = torch.zeros([x.shape[0], (self.max_len - (multi_scale_x.shape[1])), x.shape[2]]).to(x.device)
            multi_scale_x = torch.cat([multi_scale_x,padding], dim=1)
        elif multi_scale_x.shape[1] > self.max_len: # trunc to resize L' to max_len
            multi_scale_x = multi_scale_x [:,:self.max_len,:]
        return multi_scale_x

class nconv(nn.Module):
    def __init__(self, gnn_type):
        super(nconv,self).__init__()
        self.gnn_type = gnn_type
    def forward(self, x, A):
        # GCN on L' elements
        if self.gnn_type =='time':
            x = torch.einsum('btdc,tw->bwdc', (x, A))
        # GCN on M elements
        else:
            x = torch.einsum('btdc,dw->btwc', (x, A))
        return x.contiguous()
    
class gcn(nn.Module):
    def __init__(self, c_in, c_out, dropout, gnn_type, order=2):
        super(gcn, self).__init__()
        self.nconv = nconv(gnn_type)
        self.gnn_type = gnn_type
        self.c_in = (order + 1) * c_in 
        self.mlp = nn.Linear(self.c_in, c_out)
        self.dropout = dropout
        self.order = order
        self.act = nn.GELU()
    def forward(self,x,a):
        # in: b t dim d_model
        # out: b t dim d_model
        out = [x]
        x1 = self.nconv(x, a)
        out.append(x1)
        for k in range(2, self.order + 1):
            x2 = self.nconv(x1, a)
            out.append(x2)
            x1 = x2
        h=torch.cat(out, dim=-1)
        h=self.mlp(h)
        h=self.act(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h#.unsqueeze(1)

class single_scale_gnn(nn.Module):
    def __init__(self, configs):
        super(single_scale_gnn, self).__init__()

        self.tk = configs.neighbor_k
        self.min_cross_scale_number = configs.subgraph_size               # original cross-scale topk num for argTop(K_s)
        self.scale_number = configs.scale_number                # how many kinds of scales preserved
        self.use_tgcn = configs.use_gcn
        self.use_ngcn = configs.use_gcn
        self.init_seq_len = configs.seq_len 
        self.pred_len = configs.pred_len
        self.ln = nn.ModuleList()
        self.channels = configs.enc_in
        self.individual = False
        self.dropout = configs.dropout
        self.device = 'cuda:' + str(configs.gpu)
        self.GraphforPre = False

        self.tvechidden = configs.d_node
        self.d_model = configs.d_model
        self.start_linear = nn.Linear(1, self.d_model)          # expand dimension 1 to d_model
        self.seq_len = self.init_seq_len + self.init_seq_len    # max_len (i.e., multi-scale shape)
        # [2L, 1] & [1, 2L] for generating intra-series graph
        self.timevec1 = nn.Parameter(torch.randn(self.seq_len, configs.d_node), requires_grad=True).to(self.device) 
        self.timevec2 = nn.Parameter(torch.randn(configs.d_node, self.seq_len), requires_grad=True).to(self.device)
        self.tgcn = gcn(self.d_model, self.d_model, self.dropout, gnn_type='time')

        # [C, 1] & [1, C] for generating inter-series graph
        self.nodevec1 = nn.Parameter(torch.randn(self.channels, configs.d_node), requires_grad=True).to(self.device)
        self.nodevec2 = nn.Parameter(torch.randn(configs.d_node, self.channels), requires_grad=True).to(self.device)
        self.gconv = gcn(self.d_model, self.d_model, self.dropout, gnn_type='nodes')
        
        self.layer_norm = nn.LayerNorm(self.channels) 
        self.grang_emb_len = math.ceil(self.d_model//4)
        self.graph_mlp = nn.Linear(2 * self.tvechidden, self.grang_emb_len)
        self.act = nn.Tanh()
        if self.use_tgcn:
            dim_seq = 2*self.d_model
            if self.GraphforPre:
                dim_seq = 2*self.d_model+self.grang_emb_len#2*self.seq_len+self.grang_emb_len
        else:
            dim_seq = 2*self.seq_len   #2*self.seq_len   
        self.Linear = nn.Linear(dim_seq, 1) # map to intial scale

    def logits_warper_softmax(self, adj, indices_to_remove, filter_value=-float("Inf")):
        adj = F.softmax(adj.masked_fill(indices_to_remove, filter_value), dim=0)
        return adj
    
    def logits_warper(self, adj, indices_to_remove, mask_pos,mask_neg,filter_value=-float("Inf")):
        #print('adj:',adj)
        mask_pos_inverse = ~mask_pos
        mask_neg_inverse = ~mask_neg
        # Replace values for mask_pos rows
        processed_pos =  mask_pos * F.softmax(adj.masked_fill(mask_pos_inverse,filter_value),dim=-1) 
        # Replace values for mask_neg rows
        processed_neg = -1 * mask_neg * F.softmax((1/(adj+1)).masked_fill(mask_neg_inverse,filter_value),dim=-1) 
        # Combine processed rows for both cases
        processed_adj = processed_pos + processed_neg
        return processed_adj
    
    def add_adjecent_connect(self, mask, periods):
        s=np.arange(0, self.seq_len - 1)
        e=np.arange(1, self.seq_len)
        start = 0
        end = 0
        for p in periods:
            ls = self.init_seq_len // p
            end = start + ls
            s = s[s != end - 1]
            e = e[e != end]
            start = end
        forahead = np.stack([s,e], 0)
        back = np.stack([e,s], 0)
        all = np.concatenate([forahead, back], 1)
        mask[all] = False
        return mask
    
    def add_cross_scale_connect(self, adj, periods):
        max_L = self.seq_len
        mask = torch.tensor([], dtype=bool).to(adj.device)
        k = self.tk
        start = 0
        end = 0
        for period in periods:
            ls = self.init_seq_len//period          # time node number at this scale
            end = start+ls
            if end > max_L:
                end = max_L
                ls = max_L-start
            # K_P constraint for each period
            kp = k//period 
            # kp = max(kp, self.min_cross_scale_number)         # finding argTop(K_s) with constraints K_P
            kp = min(kp, ls)                        # prevent K exceeding ls
            mask = torch.cat([mask, 
                adj[:, start:end] < 
                    (torch.topk(adj[:,start:end], k=kp)[0][..., -1, None] if kp > 0 else
                    torch.full((adj.size(0), 1), float('inf'), dtype=adj.dtype, device=adj.device))
            ], dim=1) 
            start = end
            if start == max_L:
                break  
        if start < max_L:
            mask = torch.cat([mask,torch.zeros(self.seq_len, max_L-start, dtype=bool).to(mask.device)],dim=1)
        return mask
    
    def add_cross_var_adj(self,adj):
        k=3
        k=min(k,adj.shape[0])
        mask = (adj < torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None]) * (adj > torch.topk(adj, k=adj.shape[0]-k)[0][..., -1, None])
        mask_pos = adj >= torch.topk(adj, k=k)[0][..., -1, None] 
        mask_neg = adj <= torch.kthvalue(adj, k=k)[0][..., -1, None]
        return mask,mask_pos,mask_neg
    
    def get_time_adj(self, periods):
        adj = F.relu(torch.einsum('td,dm->tm', self.timevec1, self.timevec2))
        # k_s-based adj mask
        mask = self.add_cross_scale_connect(adj, periods)
        # trend-based adj mask
        mask = self.add_adjecent_connect(mask, periods)
        # adj re-normalization
        adj = self.logits_warper_softmax(adj=adj, indices_to_remove=mask)
        return adj
    
    def get_var_adj(self):
        adj=F.relu(torch.einsum('td,dm->tm', self.nodevec1, self.nodevec2))
        mask, mask_pos, mask_neg = self.add_cross_var_adj(adj)
        adj = self.logits_warper(adj, mask, mask_pos, mask_neg)
        return adj
    
    def get_time_adj_embedding(self,b):
        graph_embedding = torch.cat([self.timevec1,self.timevec2.transpose(0,1)],dim=1) 
        graph_embedding = self.graph_mlp(graph_embedding)
        graph_embedding = graph_embedding.unsqueeze(0).unsqueeze(2).expand([b,-1,self.channels,-1])
        return graph_embedding
    
    def expand_channel(self,x):
        # x: [B, 2L, C] -> [B, 2L, C, 1] -> [B, 2L, C, D]
        x = x.unsqueeze(-1)
        x = self.start_linear(x)
        return x
    
    def forward(self, x):
        # x: [Batch, Input length, Dim]
        periods, _ = FFT_for_Period(x,self.scale_number)\
        # using avgpool to generate multi-scale results; moving_avg's kernal is the period
        multi_scale_func = multi_scale_data(kernel_size=periods, return_len=self.seq_len)
        # x: [B, L, C] -> [B, 2L, C, D]
        x = multi_scale_func(x)
        x = self.expand_channel(x)
        batch_size=x.shape[0]
        x_ = x
        if self.use_tgcn:
            time_adp = self.get_time_adj(periods)
            x = self.tgcn(x,time_adp) + x
        if self.use_ngcn:
            gcn_adp = self.get_var_adj()
            x = self.gconv(x, gcn_adp) + x
        x = torch.cat([x_ , x],dim=-1)
        if self.use_tgcn and self.GraphforPre:
            graph_embedding = self.get_time_adj_embedding(b=batch_size)
            x=torch.cat([x,graph_embedding],dim=-1)
        x = self.Linear(x).squeeze(-1)
        x = x + self.layer_norm(F.dropout(x,p=self.dropout,training=self.training))
        return x[:,:self.init_seq_len,:] # [Batch, init_seq_len(96), variables]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        assert configs.use_gcn, 'This model must use GCN!'
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.e_layers = configs.e_layers
        self.graph_encs = nn.ModuleList([single_scale_gnn(configs=configs) for _ in range(self.e_layers)])
        self.anti_ood = 1   # a trick using the ground truth of last value
        self.ln = nn.Linear(self.seq_len, self.pred_len)

        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=False)
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        # x: [B, L, C]
        if self.anti_ood:
            seq_last = x_enc[:, -1:, :].detach()
            output = x_enc - seq_last
        else:
            output = x_enc
        # output: [B, L, C]
        for layer in self.graph_encs:
            output = layer(output)
        output = self.ln(output.permute(0, 2, 1)).permute(0, 2, 1)

        if self.anti_ood:
            output = output + seq_last
        if self.revin:
            output = self.revin_layer(output, 'denorm')
        return output, None