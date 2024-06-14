def add_rnn_parser(parser):
    # RNN settings for GTformer
    parser.add_argument('--r_layers',           type=int,   default=1,              help='the multi-layer RNN')