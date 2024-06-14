def add_task_parser(parser):
    # random seed
    parser.add_argument('--seed',        type=int,   default=2023,           help='random seed')
    parser.add_argument('--is_training',        type=int,   default=1,              help='status')
    # basic config
    parser.add_argument('--seq_len',            type=int,   default=96,            help='input sequence length')
    parser.add_argument('--label_len',          type=int,   default=48,             help='start token length')
    parser.add_argument('--pred_len',           type=int,   default=96,             help='prediction sequence length')
    
    parser.add_argument('--task',               type=str,   default='M', 
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target',             type=str,   default='OT',           help='target feature in S or MS task')
