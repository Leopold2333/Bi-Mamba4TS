def add_optim_parser(parser):
    # optimization
    parser.add_argument('--itr',                type=int,   default=1,              help='experiments times')
    parser.add_argument('--train_epochs',       type=int,   default=60,            help='train epochs')
    parser.add_argument('--batch_size',         type=int,   default=32,             help='batch size of train input data')
    parser.add_argument('--patience',           type=int,   default=5,             help='early stopping patience')
    parser.add_argument('--learning_rate',      type=float, default=4e-04,        help='optimizer learning rate')
    parser.add_argument('--loss',               type=str,   default='mse',          help='loss function, choose [mse, rmse, mae, mape, huber]')
    parser.add_argument('--lradj',              type=str,   default='type3',        help='adjust learning rate')
    parser.add_argument('--pct_start',          type=float, default=0.3,            help='how many epochs should the lr_rate get to max?')
    parser.add_argument('--opt',                type=str,   default='adam',         help='optimizer chosen from [Adam, SGD]')