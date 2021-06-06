import argparse


def configs_base():
    """ Set base configs, parent configs of each model. """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--backbone', choices=['pointnet2_ssg', 'dgcnn'],
                        default='dgcnn', help='Backbone.')
    parser.add_argument('--multi', type=float,
                        default=1.0, help='Model channel size.')
    parser.add_argument('--output_dim', type=int,
                        default=1024, help='Output dim of encoder.')
    parser.add_argument('--crop_norm', action='store_true',
                        default=False, help='Normalize the crop.')
    parser.add_argument('--data', choices=['ModelNet40', 'ShapeNet', 'ShapeNetScan'],
                        default='ModelNet40', help='Pretrain dataset.')
    parser.add_argument('--partial_id', default=1, type=int, help='Use partial data id.')
    parser.add_argument('--txt',
                        default='train.txt', help='Filename txt')
    parser.add_argument('--num_points', type=int,
                        default=1024, help='Sample points for ShapeNetScan dataset.')
    parser.add_argument('--add_test',
                        default=False, action='store_true', help='Add test dataset to unsupervised training.')
    parser.add_argument('--batch_size', type=int,
                        default=30, help='Batch size.')
    parser.add_argument('--epochs', type=int,
                        default=300, help='Train epoch.')
    parser.add_argument('--save', type=int,
                        default=5, help='Save checkpoints.')
    parser.add_argument('--optimizer', choices=['sgd', 'larc', 'lars', 'adam', 'adadelta', 'adamax'],
                        default='sgd', help='optimizer')
    parser.add_argument(
        '--scheduler', choices=['auto', 'warmup_cos', 'warmup_step', 'cos', 'step'], help='Learning rate scheduler.')
    parser.add_argument('--warmup_epochs', type=int,
                        default=0, help='learning rate will be linearly scaled during warm up period')
    parser.add_argument('--warmup_lr', type=float,
                        default=0, help='Initial warmup learning rate')
    parser.add_argument('--step_decay_lr', type=float,
                        default=0.7, help='Step decay rate.')
    parser.add_argument('--step_epochs', type=int,
                        default=20, help='Decay lr after step_epochs.')
    parser.add_argument('--constant_predictor_lr', action='store_false',
                        help='')
    parser.add_argument('--lr', type=float,
                        default=0.05)
    parser.add_argument('--final_lr', type=float,
                        default=0)
    parser.add_argument('--momentum', type=float,
                        default=0.9)
    parser.add_argument('--weight_decay', type=float,
                        default=0.0001, help='L2 normalize coffe.')
    parser.add_argument('--last_epoch', default=None)
    parser.add_argument('--device',
                        default="0,1,2,3", help='GPU device.')
    parser.add_argument('--seed', type=int,
                        default=28, help='Random seed.')
    return parser


def configs_random_transform():
    """ Set random transform configs. """
    parser = argparse.ArgumentParser(add_help=False)
    # scale
    parser.add_argument('--scale', action='store_true',
                        default=False, help='Random scale.')
    parser.add_argument('--scale_low', type=float, default=.5)
    parser.add_argument('--scale_high', type=float, default=1.5)
    # translate
    parser.add_argument('--translate', action='store_true',
                        default=False, help='Random translate.')
    parser.add_argument('--shift_range', type=float, default=.2)
    # jitter
    parser.add_argument('--jitter', action='store_true',
                        default=False, help='Random jitter.')
    parser.add_argument('--sigma', type=float, default=0.02)
    # rotate z
    parser.add_argument('--rotate_z', action='store_true',
                        default=False, help='Random rotate along z axis.')
    # rotate group
    parser.add_argument('--rotate_group', action='store_true',
                        default=False, help='Random rotate SO3.')
    parser.add_argument('--rot_range', type=float,
                        default=30, help='Random rotate angle range.')
    return parser


def configs_simsiam():
    """ Set simsiam configs. """
    base_parser = configs_base()
    random_parser = configs_random_transform()
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--crop_size', nargs='+', type=int,
                        default=None, help='List of crop size.')
    parser.add_argument('--proj_layers', type=int,
                        default=None, help='Number of projector layers.')
    parser.add_argument('--project_hdim', type=int,
                        default=2048, help='Hidden dim of projection MLP.')
    parser.add_argument('--project_odim', type=int,
                        default=2048, help='Output dim of projection MLP.')
    parser.add_argument('--predict_hdim', type=int,
                        default=512, help='Hidden dim of prediction MLP.')
    all_parser = argparse.ArgumentParser(
        parents=[base_parser, random_parser, parser])
    return all_parser.parse_args()
