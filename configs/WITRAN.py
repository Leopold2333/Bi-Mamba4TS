def add_ran_parser(parser):
    parser.add_argument('--e_layers',           type=int,   default=3,              help='num of encoder layers')
    parser.add_argument('--dropout',            type=float, default=0.2,            help='dropout')
    parser.add_argument('--d_model',            type=int,   default=32,             help='dimension of')
    parser.add_argument('--d_ff',               type=int,   default=32,             help='dimension of')
    parser.add_argument('--WITRAN_deal',        type=str,   default='None',         help='WITRAN deal data type, options:[None, standard]')
    parser.add_argument('--WITRAN_grid_cols',   type=int,   default=24,             help='Numbers of data grid cols for WITRAN')
