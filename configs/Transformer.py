def add_transformer_parser(parser):
    parser.add_argument('--fc_dropout',         type=float, default=0.2,            help='fully connected dropout')
    parser.add_argument('--head_dropout',       type=float, default=0.0,            help='head dropout')

    # Autoformer Decompose
    parser.add_argument('--decomposition',                  default=False,  action='store_true',
                        help='decomposition')
    parser.add_argument('--kernel_size',        type=int,   default=25,             help='decomposition-kernel of AVGPool')
    
    parser.add_argument('--pos_learnable',                  default=False,  action='store_true',
                        help='use fixed or learned Position Encoding')
    parser.add_argument('--pos_embed_type',     type=str,   default='sincos',       help='how do you generate positional encoding for Encoder')
    parser.add_argument('--embed_type',         type=int,   default=1,              help='0: value \
                                                                                          1: value + positional \
                                                                                          2: value + temporal \
                                                                                          3: value + positional + temporal')
    parser.add_argument('--d_model',            type=int,   default=128,            help='dimension of model')
    parser.add_argument('--n_heads',            type=int,   default=8,              help='num of heads')
    parser.add_argument('--e_layers',           type=int,   default=3,              help='num of encoder layers')
    parser.add_argument('--d_layers',           type=int,   default=4,              help='num of decoder layers')
    parser.add_argument('--d_ff',               type=int,   default=256,            help='dimension of fcn')
    parser.add_argument('--factor',             type=int,   default=10,              help='attn factor')
    parser.add_argument('--distil',                         default=False, action='store_true', 
                        help='whether to use distilling in encoder for Crossformer and Informer')
    parser.add_argument('--dropout',            type=float, default=0.2,            help='dropout')
    parser.add_argument('--activation',         type=str,   default='gelu',         help='activation')
    parser.add_argument('--output_attention',               default=False, action='store_true', 
                        help='whether to output attention in ecoder')
