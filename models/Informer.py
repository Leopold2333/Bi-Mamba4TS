import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding, DataEmbedding_wo_pos, DataEmbedding_wo_temp, DataEmbedding_wo_pos_temp
from layers.Transformer_layer import Encoder, EncoderLayer, Decoder, DecoderLayer, ConvLayer
from layers.Attention import AttentionLayer, ProbAttention
from layers.RevIn import RevIN


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.pred_len = configs.pred_len

        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=False)

        if configs.embed_type == 0:
            self.enc_embedding = DataEmbedding_wo_pos_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 1:
            self.enc_embedding = DataEmbedding_wo_temp(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_temp(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 2:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        elif configs.embed_type == 3:
            self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Encoder定义
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            [
                ConvLayer(
                    configs.d_model
                ) for _ in range(configs.e_layers-1)
            ] if configs.distil else None,
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, 
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        
        # 保证gpu位置
        x_mark_enc, x_dec, x_mark_dec = x_mark_enc.to(x_enc.device), x_dec.to(x_enc.device), x_mark_dec.to(x_enc.device)
        # 整合输入内容的嵌入
        # x: [batch_size x seq_len x channels]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        # Informer是基于每个点的，主要使用1维卷积将每个数据点的特征映射到不同通道上去
        # enc_out: [batch_size x seq_len x d_model] -> [batch_size x (seq_len / 2^e_layers) x d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # 以相同的过程对标整理出预测内容的嵌入
        # x_dec: [batch_size x (label_len + pred_len) x channels]
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # dec_out: [batch_size x (label_len + pred_len) x channels] -> [batch_size x (label_len + pred_len) x c_out]
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        dec_out = self.projection(dec_out)

        if self.revin:
            dec_out = self.revin_layer(dec_out, 'denorm')
        
        return dec_out[:,-self.pred_len:,:], attns 
