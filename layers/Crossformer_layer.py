import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Attention import TwoStageAttentionLayer, MultiheadAttention
from utils.tools import Transpose, get_activation_fn


class TruncateModule(nn.Module):
    def __init__(self, target_length):
        super(TruncateModule, self).__init__()
        self.target_length = target_length

    def forward(self, x, truncate_length):
        return x[: ,: ,:truncate_length]
    

class SegMerging(nn.Module):
    '''
    Segment Merging Layer.
    The adjacent `win_size' segments in each dimension will be merged into one segment to
    get representation of a coarser scale
    we set win_size = 2 in our paper
    '''
    def __init__(self, d_model, win_size, norm_layer=nn.LayerNorm):
        super().__init__()
        self.d_model = d_model
        self.win_size = win_size
        self.linear_trans = nn.Linear(win_size * d_model, d_model)
        self.norm = norm_layer(win_size * d_model)

    def forward(self, x):
        # x: [batch_size * channels, patch_num, d_model
        bc, patch_num, d_model = x.shape
        pad_num = patch_num % self.win_size
        if pad_num != 0: 
            pad_num = self.win_size - pad_num
            x = torch.cat((x, x[:, -pad_num:, :]), dim = -2)

        seg_to_merge = []
        for i in range(self.win_size):
            seg_to_merge.append(x[:, i::self.win_size, :])
        # x: [B, channels, patch_num/win_size, win_size*d_model]
        x = torch.cat(seg_to_merge, -1)

        x = self.norm(x)
        x = self.linear_trans(x)

        return x


class TSAEncoder(nn.Module):
    def __init__(self, batch_size:int=32, channels:int=21, win_size:int=2, d_model:int=128, n_heads:int=8, d_ff:int=256, 
                 attn_dropout:float=0., dropout:float=0.2, patch_num:int=10, factor:int=10, activation:str='gelu'):
        super(TSAEncoder, self).__init__()
        if (win_size > 1):
            self.merge_layer = SegMerging(d_model, win_size, nn.LayerNorm)
        else:
            self.merge_layer = None
        self.TSA = TwoStageAttentionLayer(batch_size=batch_size, channels=channels, patch_num=patch_num, factor=factor, 
                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                                          attn_dropout=attn_dropout, dropout=dropout, 
                                          activation=get_activation_fn(activation), transpose=Transpose)
    
    def forward(self, x):
        # x: [batch_size * channels x patch_num x d_model]
        if self.merge_layer is not None:
            x = self.merge_layer(x)
        output = self.TSA(x)
        return output


class TSADecoder(nn.Module):
    def __init__(self, batch_size:int=32, channels:int=21, pred_len:int=96, d_model:int=128, n_heads:int=8, d_ff:int=256, 
                 attn_dropout:float=0., dropout:float=0.2, patch_num:int=10, patch_len:int=16, factor:int=10, norm:str='BatchNorm', activation:str='gelu'):
        super(TSADecoder, self).__init__()
        self.batch_size = batch_size
        self.patch_len = patch_len
        self.TSA = TwoStageAttentionLayer(batch_size=batch_size, channels=channels, patch_num=patch_num, factor=factor, 
                                          d_model=d_model, n_heads=n_heads, d_ff=d_ff, 
                                          attn_dropout=attn_dropout, dropout=dropout, 
                                          activation=get_activation_fn(activation), transpose=Transpose)
  
        self.cross_attention = MultiheadAttention(d_model=d_model, n_heads=n_heads, 
                                                  attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=False)
        if "Batch" in norm:
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.BatchNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)
            self.norm_ffn = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff),
                                get_activation_fn(activation),
                                nn.Linear(d_ff, d_model))
        self.linear_pred = nn.Linear(d_model, patch_len)

    def forward(self, x, cross):
        # x: [batch_size * channels x patch_num_out x d_model]
        x = self.TSA(x)
        # cross: [batch_size * channels x patch_num_in x d_model]
        tmp, _ = self.cross_attention(
            x, cross, cross,
        )
        x = x + self.dropout(tmp)
        y = x = self.norm_attn(x)
        y = self.ff(y)
        dec_output = self.norm_ffn(x+y)
        # dec_output: [batch_size * channels x patch_num_out x d_model]
        layer_predict = torch.reshape(dec_output, (self.batch_size, -1, dec_output.shape[1], dec_output.shape[2]))
        # layer_predict: [batch_size x channels x patch_num_out x patch_len]
        layer_predict = self.linear_pred(layer_predict)

        return dec_output, layer_predict