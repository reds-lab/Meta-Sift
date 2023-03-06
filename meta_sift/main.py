import argparse
import torch
from meta_sift import *
from util import *

def main(args):
    train_poi_set, test_poi_set, poi_idx = get_dataset(args)
    vnet_list, mnet_list = train_sifter(args, train_poi_set)
    v_res = test_sifter(args, test_poi_set, vnet_list, mnet_list)
    new_idx = get_sifter_result(args, test_poi_set, v_res)
    # NCR for meta_sift
    print('NCR for Meta Sift is：%.3f%%' % get_NCR(train_poi_set, poi_idx, new_idx))

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_root', type=str)
    parser.add_argument('--corruption_type', type=str, default="badnets")
    parser.add_argument('--corruption_ratio', type=float, default=0.33)
    parser.add_argument('--tar_lab', type=int, default=5)
    parser.add_argument('--repeat_rounds', type=int, default=5)
    parser.add_argument('--res_epochs', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    parser.add_argument('--random_seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--gpu', type=int, default=5)
    
    
    parser.add_argument('--v_lr', type=float, default=0.0005)
    parser.add_argument('--meta_lr', type=float, default=0.1)
    
    parser.add_argument('--top_k', type=int, default=15)
    parser.add_argument('--go_lr', type=float, default=0.1)
    parser.add_argument('--num_act', type=int, default=4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)

    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    set_seed(args.random_seed)
    main(args)