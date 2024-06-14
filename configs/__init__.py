from .task import add_task_parser
from .dataset import add_dataset_parser
from .optimization import add_optim_parser
from .gpu import add_gpu_parser

from .Linear import add_linear_parser
from .Transformer import add_transformer_parser
from .RNN import add_rnn_parser
from .GCN import add_gcn_parser
from .patch import add_patch_parser
from .CrossGNN import add_crossgnn_parser
from .WITRAN import add_ran_parser

from .Mamba import add_4mamba_parser