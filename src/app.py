import os
import setproctitle
import argparse
from trainer import Train
import torch
import random
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run persuasion")
    parser.add_argument('--epoch_num', type=int,nargs='?', default=500,
                        help='epoch_num')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7,
                        help='weight_decay')
    parser.add_argument('--patience', type=int, default=3,
                        help='weight_decay')
    parser.add_argument('--early_stop_num', type=int, default=10,
                        help='weight_decay')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='batch_size')
    parser.add_argument('--full_node_feature', type=int, default=0,
                        help='whether to include #neighbor as featrue')
    parser.add_argument('--dataset_path', type=str, default='/home/yuanyuan/workplace/influence/data/sample2_dataset_unc_norm.npy',
                        help='node feature matrix data path')
    parser.add_argument('--dataset_path2', type=str, default='/home/yuanyuan/workplace/influence/data/feature_embedding20.npy',
                        help='deepwalk embedding feature')
    parser.add_argument('--seed', type=int, default=630, help='random seed')
    parser.add_argument('--gpu', type=float, default='1', help='gpu id')
    parser.add_argument('--n_folds', type=int, default=10, help='number of folds')
    parser.add_argument('--device', type=str, default='cuda:6')
    return parser.parse_args()

args = parse_args()
args.rnd_state = np.random.RandomState(args.seed)
setproctitle.setproctitle("DW@yy")
#os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

if __name__ == '__main__':
    app = Train(args)
    #app.train_classic()
    app.trainstep()

'''
{
    "name": "dl2",
    "host": "101.6.68.156",
    "protocol": "sftp",
    "port": 35167,
    "username": "yuanyuan",
    "privateKeyPath": "/Users/yuanyuan/.ssh/id_rsa",
    "remotePath": "/home/yuanyuan/workplace/influence/src",
    "uploadOnSave": true
}
'''

'''
{
    "name": "dl1",
    "host": "dl.fib-lab.com",
    "protocol": "sftp",
    "port": 35167,
    "username": "mas",
    "privateKeyPath": "/Users/yuanyuan/.ssh/id_rsa",
    "remotePath": "/home/mas/yuanyuan/workplace/Community_value_prediction/baseline",
    "uploadOnSave": true
}
'''