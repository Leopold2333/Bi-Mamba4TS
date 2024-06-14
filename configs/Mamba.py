def add_4mamba_parser(parser):
    # token embedding
    parser.add_argument('--d_model',        type=int,       default=64,         help='Sequence Elements embedding dimension')
    parser.add_argument('--d_ff',           type=int,       default=128,        help='Second Embedded representation')
    
    # mamba block
    parser.add_argument('--bi_dir',         type=int,       default=1,          help='use bidirectional Mamba?')
    parser.add_argument('--d_state',        type=int,       default=32,         help='d_state parameter of Mamba')
    parser.add_argument('--d_conv',         type=int,       default=2,          help='d_conv parameter of Mamba')
    parser.add_argument('--e_fact',         type=int,       default=1,          help='expand factor parameter of Mamba')

    parser.add_argument('--e_layers',       type=int,       default=1,          help='layers of encoder')
    parser.add_argument('--dropout',        type=float,     default=0.2,        help='dropout')
    parser.add_argument('--activation',     type=str,       default='gelu',     help='activation')
