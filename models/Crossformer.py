import torch
from torch import nn
from layers.Embed import PatchEmbedding
from layers.Crossformer_layer import TruncateModule, TSAEncoder, TSADecoder
from layers.RevIn import RevIN
from math import ceil
import warnings

warnings.filterwarnings('ignore')


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super(Model, self).__init__()
        assert configs.embed_type == 1, 'Crossformer has decoders, Decoder Positional Embeddings Needed!'
        assert configs.e_layers == configs.d_layers - 1, 'Crossformer uses cross-attention across encoders and decoders, must have another decoder for processing the initial state!'
        self.batch_size = configs.batch_size
        self.e_layers = configs.e_layers
        self.d_layers = configs.d_layers
        
        # RevIn
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=False)

        # DSW embedding settings
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = configs.patch_len
        self.stride = configs.stride

        # being able to divide indicates that there is no need to forcefully padding
        if configs.seq_len % self.stride==0:
            self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
            process_layer = nn.Identity()
        else:
            # padding at tail
            if configs.padding_patch=="end":
                padding_length = configs.stride - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 2)
                process_layer = nn.ReplicationPad1d((0, padding_length))
            # if not padding, then execute cutting
            else:
                truncated_length = configs.seq_len - (configs.seq_len % configs.stride)
                self.patch_num = int((configs.seq_len - configs.patch_len)/configs.stride + 1)
                process_layer = TruncateModule(truncated_length)
        self.process_layer_enc = PatchEmbedding(self.patch_num, configs.d_model, configs.patch_len, configs.stride, configs.dropout,
                                                process_layer, configs.pos_embed_type)
        # for the output segmentation
        if configs.seq_len % self.stride==0:
            self.patch_num_out = int((configs.pred_len - configs.patch_len)/configs.stride + 1)
        else:
            if configs.padding_patch=="end":
                self.patch_num_out = int((configs.pred_len - configs.patch_len)/configs.stride + 2)
            else:
                self.patch_num_out = int((configs.pred_len - configs.patch_len)/configs.stride + 1)
        
        self.pre_norm = nn.LayerNorm(configs.d_model)
        
        # Encoders - SegMerge will only be done when distil is set
        self.encoder = nn.ModuleList([TSAEncoder(
            batch_size=configs.batch_size, channels=configs.enc_in, win_size=1 if i==0 else 1 if not configs.distil else 2, 
            d_model=configs.d_model, n_heads=configs.n_heads, d_ff=configs.d_ff, 
            dropout=configs.dropout, patch_num=self.patch_num if not configs.distil else ceil(self.patch_num/2**i), 
            factor=configs.factor, activation=configs.activation
        ) for i in range(configs.e_layers)])
        
        # position encoding - decoder
        self.D_P = nn.Parameter(torch.randn(1, configs.dec_in, self.patch_num_out, configs.d_model))

        # Decoders
        self.decoder = nn.ModuleList([TSADecoder(
            batch_size=configs.batch_size, channels=configs.enc_in, pred_len=configs.pred_len, 
            d_model=configs.d_model, n_heads=configs.n_heads, d_ff=configs.d_ff, 
            dropout=configs.dropout, patch_num=self.patch_num_out, patch_len=configs.patch_len, factor=configs.factor, activation=configs.activation
        ) for _ in range(configs.d_layers)])

        self.count_list = torch.ones(configs.pred_len)
        max_count = configs.patch_len // configs.stride
        i = 0
        while i < max_count and i*configs.stride <= configs.pred_len//2:
            self.count_list[i*configs.stride:(i+1)*configs.stride] += i
            if i==0:
                self.count_list[-(i+1)*configs.stride:] += i
            else:
                self.count_list[-(i+1)*configs.stride:-i*configs.stride] += i
            i += 1
        if i < max_count:
            max_count = i
        self.count_list[i:-i] = max_count
        self.count_list.unsqueeze(0).unsqueeze(1)


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):        
        # x: [batch_size x input_length x channel] -> [batch size x channel x input length]
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        x_enc = x_enc.permute(0, 2, 1)
        # do patching
        output, _ = self.process_layer_enc(x_enc)
        output = self.pre_norm(output)

        # go through several Encoders
        encoder_list = [output]
        for layer in self.encoder:
            output = layer(output)
            encoder_list.append(output)
        
        # output: [batch_size * channel x patch_num x d_model]
        # transform the decoder position encodings
        dec_in = self.D_P.repeat(self.batch_size, 1, 1, 1)
        dec_in = torch.reshape(dec_in, (-1, dec_in.shape[2], dec_in.shape[3]))
        final_predict = None
        i = 0
        for layer in self.decoder:
            cross_enc = encoder_list[i]
            dec_in, layer_predict = layer(dec_in, cross_enc)
            if final_predict is None:
                final_predict = layer_predict
            else:
                final_predict = final_predict + layer_predict
            i += 1
        # concat the prediction from the final decoder layer
        # output: [batch_size x channel x target_window]
        # final_predict: [batch_size x channels x patch_num_out x patch_len]
        count_list = self.count_list.to(x_enc.device)
        output = torch.zeros(final_predict.shape[0], final_predict.shape[1], self.pred_len).to(x_enc.device)
        for i in range(self.patch_num_out):
            start = self.stride*i
            output[:,:,start:start+self.patch_len] += final_predict[:,:,i,:]
        output = output/count_list
        output = output.permute(0,2,1)

        if self.revin:
            output = self.revin_layer(output, 'denorm')
        return output, None
