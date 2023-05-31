import argparse
import random
#
import numpy as np
import torch
#
from recbole_gnn.quick_start import run_recbole_gnn
#
#
if __name__ == '__main__':

    seed = 2020
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SimDCL', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()
    #
    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_gnn(model=args.model, dataset=args.dataset)
