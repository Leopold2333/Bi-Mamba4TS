def add_gcn_parser(parser):
    # Graph Structure Construct
    parser.add_argument('--use_gcn',                        default=False, action='store_true', help='generate a graph?')
    parser.add_argument('--gcn_bi',                         default=False, action='store_true', help='generate bi-directional graph?')
    parser.add_argument('--gcn_heterogeneity',              default=False, action='store_true', help='whether to preserve heterogeneity?')

    parser.add_argument('--threshold',          type=float, default=0.6,            help='λ for Pearson Correlation Coefficient')
    
    parser.add_argument('--subgraph_size',      type=int,   default=4,              help='top-k')
    parser.add_argument('--gcn_depth',          type=int,   default=2,              help='graph convolution depth')
    parser.add_argument('--tanh_alpha',         type=float, default=3,              help='adj alpha')
    parser.add_argument('--prop_alpha',         type=float, default=0.05,           help='prop alpha')
    parser.add_argument('--mlp_type',           type=int,   default=2,              help='0:gc 1:ln 2:custom-ln for aggregating information')

    parser.add_argument('--d_node',             type=int,   default=32,             help='dim of nodes')
