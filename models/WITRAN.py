import torch
import torch.nn as nn
from layers.Embed import WITRAN_Temporal_Embedding
from layers.WITRAN_layer import WITRAN_2DPSGMU_Encoder
from layers.RevIn import RevIN

class Model(nn.Module):
    def __init__(self, configs, WITRAN_dec='Concat', WITRAN_res='none', WITRAN_PE='add'):
        super(Model, self).__init__()
        self.standard_batch_size = configs.batch_size
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.dec_in = configs.dec_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.num_layers = configs.e_layers
        self.dropout = configs.dropout
        self.WITRAN_dec = WITRAN_dec
        self.WITRAN_deal = configs.WITRAN_deal
        self.WITRAN_res = WITRAN_res
        self.PE_way = WITRAN_PE
        self.WITRAN_grid_cols = configs.WITRAN_grid_cols
        self.WITRAN_grid_enc_rows = int(configs.seq_len / self.WITRAN_grid_cols)
        self.WITRAN_grid_dec_rows = int(configs.pred_len / self.WITRAN_grid_cols)
        self.device = configs.gpu
        if configs.freq== 'h':
            Temporal_feature_dim = 4
        # Encoder
        self.encoder_2d = WITRAN_2DPSGMU_Encoder(self.enc_in + Temporal_feature_dim, self.d_model, self.num_layers, 
            self.dropout, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, self.WITRAN_res)
        # Embedding
        self.dec_embedding = WITRAN_Temporal_Embedding(Temporal_feature_dim, configs.d_model,
            configs.embed, configs.freq, configs.dropout)
        
        if self.PE_way == 'add':
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model, self.c_out)
        else:
            if self.WITRAN_dec == 'FC':
                self.fc_1 = nn.Linear(self.num_layers * (self.WITRAN_grid_enc_rows + self.WITRAN_grid_cols) * self.d_model, 
                    self.pred_len * self.d_model)
            elif self.WITRAN_dec == 'Concat':
                self.fc_1 = nn.Linear(self.num_layers * 2 * self.d_model, self.WITRAN_grid_dec_rows * self.d_model)
            self.fc_2 = nn.Linear(self.d_model * 2, self.c_out)
        
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(configs.enc_in, affine=configs.affine, subtract_last=False)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # x: [B, L, M]
        if self.revin:
            x_enc = self.revin_layer(x_enc, 'norm')
        if self.WITRAN_deal == 'standard':
            seq_last = x_enc[:,-1:,:].detach()
            x_enc = x_enc - seq_last
        # with time feature embedding: x_input_enc: [B, L, (C_in+C_time)]
        x_input_enc = torch.cat([x_enc, x_mark_enc], dim=-1)
        batch_size, _, input_size = x_input_enc.shape
        # [B, L, (C_in+C_time)] -> [B, R, C, (C_in+C_time)]
        x_input_enc = x_input_enc.reshape(batch_size, self.WITRAN_grid_enc_rows, self.WITRAN_grid_cols, input_size)

        if self.WITRAN_grid_enc_rows <= self.WITRAN_grid_cols:
            flag = 0
        else: # need permute
            flag = 1
        
        _, enc_hid_row, enc_hid_col = self.encoder_2d(x_input_enc, batch_size, input_size, flag)
        dec_T_E = self.dec_embedding(x_mark_dec)

        if self.WITRAN_dec == 'FC':
            hidden_all = torch.cat([enc_hid_row, enc_hid_col], dim = 2)
            hidden_all = hidden_all.reshape(hidden_all.shape[0], -1)
            last_output = self.fc_1(hidden_all)
            last_output = last_output.reshape(last_output.shape[0], self.pred_len, -1)
            
        elif self.WITRAN_dec == 'Concat':
            enc_hid_row = enc_hid_row[:, :, -1:, :].expand(-1, -1, self.WITRAN_grid_cols, -1)
            output = torch.cat([enc_hid_row, enc_hid_col], dim = -1).permute(0, 2, 1, 3)
            output = output.reshape(output.shape[0], 
                output.shape[1], output.shape[2] * output.shape[3])
            last_output = self.fc_1(output)
            last_output = last_output.reshape(last_output.shape[0], last_output.shape[1], 
                self.WITRAN_grid_dec_rows, self.d_model).permute(0, 2, 1, 3)
            last_output = last_output.reshape(last_output.shape[0], 
                last_output.shape[1] * last_output.shape[2], last_output.shape[3])
            
        if self.PE_way == 'add':
            last_output = last_output + dec_T_E[:, -self.pred_len:, :]
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(last_output) + seq_last
            else:
                last_output = self.fc_2(last_output)
        else:
            if self.WITRAN_deal == 'standard':
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1)) + seq_last
            else:
                last_output = self.fc_2(torch.cat([last_output, dec_T_E], dim = -1))
        
        if self.revin:
            last_output = self.revin_layer(last_output, 'denorm')
        return last_output, None