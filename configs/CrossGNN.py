def add_crossgnn_parser(parser):
    parser.add_argument('--d_model',        type=int,   default=16,         help='dimension of expanding')
    parser.add_argument('--d_ff',        type=int,   default=16,         help='dimension of expanding')
    parser.add_argument('--e_layers',       type=int,   default=3,          help='num of encoder layers')
    parser.add_argument('--dropout',        type=float, default=0.2,        help='dropout')
    parser.add_argument('--neighbor_k',     type=int,   default=10,         help='constraints constant K in each period')
    parser.add_argument('--scale_number',   type=int,   default=4,          help='scale types')