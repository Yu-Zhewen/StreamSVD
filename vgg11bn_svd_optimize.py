import argparse
import os
from svd_optimize_main import svd_main

parser = argparse.ArgumentParser(description='SVD optimization')

parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id (main) to use.')
parser.add_argument('--data_parallel', default=None, type=int, metavar='N', nargs='+',
                    help='GPU ids to use.')


parser.add_argument('--data', default="ILSVRC2012_img", type=str,
                    help='directory of the dataset')
parser.add_argument('--batch_size', default='32', type=int, 
                    help='')
parser.add_argument('--workers', default='4', type=int, 
                    help='')


parser.add_argument('--model_path', default="checkpoint/vgg11_bn_pretrained", type=str,
                    help='path to the pretrained model')                    


parser.add_argument('-s', '--approximate_scheme', default=None, type=int, metavar='N', nargs='+',
                    help='choose the approximate scheme')                    
parser.add_argument('-r', '--rank', default=None, type=int, metavar='N', nargs='+',
                    help='rank limitation')
parser.add_argument('--rank_selection_method', default="rank_taylor", type=str,
                    help='rank selection method')


parser.add_argument('--quantization', default='2', type=int,
                    help='0-disable, 1-post quantization, 2-svd iterative quantization, 3-bfp post quantization, 4-bfp iterative quantization')
parser.add_argument('--quantise_first', default=0, type=int,
                    help='0-quantisation after svd, 1-quantisation before svd')
parser.add_argument('--weight_width', default=8, type=int,
                    help='wordlength of weight')
parser.add_argument('--data_width', default=16, type=int,
                    help='wordlength of activation')


parser.add_argument('--output_path', default=None, type=str,
                    help='output path')


args = parser.parse_args()

args.model_name = "vgg11_bn"
args.print_freq = 2560 / args.batch_size 
if args.output_path == None:
    args.output_path = os.getcwd() + "/output"

print(args)

svd_main(args)