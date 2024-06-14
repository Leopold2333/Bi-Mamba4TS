def add_gpu_parser(parser):
    # GPU
    parser.add_argument('--use_cpu',                        default=False, action='store_true', help='use gpu')

    parser.add_argument('--gpu',                type=int,   default=0, help='gpu')
    parser.add_argument('--use_multi_gpu',                  default=False, action='store_true', help='use multiple gpus')
    parser.add_argument('--device_ids',         type=str,   default='0,1',                    help='logic gpu ids, but assigned')