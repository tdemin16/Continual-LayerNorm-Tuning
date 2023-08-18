from .cifar100 import get_args_parser as cifar100_parser
from .imr import get_args_parser as imr_parser

CONFIGS = {
    'cifar100': cifar100_parser,
    'imr': imr_parser
}