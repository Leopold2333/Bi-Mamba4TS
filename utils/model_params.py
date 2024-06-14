from configs import add_linear_parser, add_transformer_parser, add_patch_parser, \
add_rnn_parser, add_gcn_parser, add_crossgnn_parser, add_ran_parser, \
add_4mamba_parser

model_parser_dict={
    'DLinear': [add_linear_parser],
    'Transformer': [add_transformer_parser],
    'Autoformer': [add_transformer_parser],
    'Informer': [add_transformer_parser],
    'Crossformer': [add_transformer_parser, add_patch_parser],
    'PatchTST': [add_transformer_parser, add_patch_parser, add_rnn_parser, add_gcn_parser],
    'iTransformer': [add_transformer_parser],
    'CrossGNN': [add_gcn_parser, add_crossgnn_parser],
    'WITRAN': [add_ran_parser],
    'TimeMachine': [add_4mamba_parser],
    'BiMamba4TS': [add_4mamba_parser, add_patch_parser],
    'DMamba': [add_4mamba_parser]
}