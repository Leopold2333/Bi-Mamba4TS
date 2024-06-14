def add_dataset_parser(parser):
    # data loader
    parser.add_argument('--dataset_name',       type=str,   default='weather',      help='dataset name')
    parser.add_argument('--root_path',          type=str,   default='data/weather', help='root path of the data file')
    parser.add_argument('--data_path',          type=str,   default='weather.csv',  help='data file')
    parser.add_argument('--dataset_type',       type=str,   default='custom',       help='dataset type')
    parser.add_argument('--freq',               type=str,   default='h', 
        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--embed',              type=str,   default='timeF',        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--num_workers',        type=int,   default=0,              help='data loader num workers')

    # model saving path
    parser.add_argument('--checkpoints',        type=str,   default='./checkpoints',    help='location of model checkpoints')
    parser.add_argument('--results',            type=str,   default='./results',        help='location of model results')
    parser.add_argument('--predictions',        type=str,   default='./predictions',    help='location of model predictions')

    # dataset channels
    parser.add_argument('--enc_in',             type=int,   default=21,             help='encoder input size')
    parser.add_argument('--dec_in',             type=int,   default=21,             help='decoder input size')
    parser.add_argument('--c_out',              type=int,   default=21,             help='output size')
    parser.add_argument('--timestamp_dim',      type=int,   default=1,              help='how many values does each timestamp record?')
    parser.add_argument('--relation',           type=str,   default='spearman',     help='algorithm for relationship')

    # Whether to utilize RevIn for the original data samples
    parser.add_argument('--revin',              type=int,   default=1,              help='RevIN; True 1 False 0')
    parser.add_argument('--affine',             type=int,   default=0,              help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last',      type=int,   default=0,              help='0: subtract mean; 1: subtract last')