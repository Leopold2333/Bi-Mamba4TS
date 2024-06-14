import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class WITRAN_2DPSGMU_Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='none'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.W_first_layer = nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        self.W_other_layer = nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        # input: [R*B, 2*d_model+input_size] --> [R*B, 6*d_model]
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a

    def forward(self, input, batch_size, input_size, flag):
        # input: [B, R, C, input_size]
        if flag == 1: # cols < rows
            input = input.permute(2, 0, 1, 3)   # [C, B, R, input_size]
        else:
            input = input.permute(1, 0, 2, 3)   # [R, B, C, input_size]
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1  # R+C-1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device) # [R*B, d_model]
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
        # 计算的过程中，前R个相当于行内的，后C个相当于行间的。因此对应于后面对slice与R-1数值大小的判断，因为列的隐状态是另一种存储方式
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size) # R*B R+C-1 D
                W = self.W_first_layer
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer-1, :, :]
                # new layer, so hidden state start from 0
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            # calculating the hidden output per column
            for slice in range (Water2sea_slice_len):
                # gate generate. cat(x, h_{t-1}^{hor}, h_{t-1}^{ver})
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]], dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
                # matrix W serves as a combination of W_σ and W_tanh for all σ and tanh gates of row and col, with total number "6"
                # information after concat needs to be fed into update gate, input gate and output gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                # sigma_gate: [R*B, 4*d_model]; tanh_gate: [R*B, 2*d_model]
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1)
                # the tanh for updated primary hidden state only serves as a function, thus no weight matrix W for it
                hidden_slice_row = torch.tanh(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col
                # output generate
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :])
                # hidden transfer because we are sliding to the row direction
                # assume that current (r,c)
                # after rolling, h_{r-1,c}^{ver} will be set ahead to go with h_{r,c-1}^{hor}
                # after rolling, h_{r-1,c-1}^{ver} will be postposed to go with h_{r,c-1}^{hor}
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)
            if self.res_mode == 'layer_res' and layer >= 1: # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all
