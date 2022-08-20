import argparse
import os
import random
import numpy as np
from copy import deepcopy
from functools import partial

import torch
import torch.optim as optim

from dataset import get_dataset
from train import train, test
from method import get_method
from utils import get_sparsity
from hyperparameter import get_hyperparameters
from activation import get_activation


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--seed', type=int, default=718, help='random seed')
  parser.add_argument('--dataset', type=str, default='mnist', help='dataset (mnist|fmnist|cifar10)')
  parser.add_argument('--network', type=str, default='mlp', help='network (mlp|lenet|conv6|vgg19|resnet18)')
  parser.add_argument('--method', type=str, default='mp', help='method (mp|rp|labp)')
  parser.add_argument('--pruning_type', type=str, default='oneshot', help='(oneshot|iterative|global)')
  parser.add_argument('--pruning_iteration_start', type=int, default=1, help='start iteration for pruning')
  parser.add_argument('--pruning_iteration_end', type=int, default=30, help='end iteration for pruning')
  args = parser.parse_args()  

  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.cuda.manual_seed_all(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False  
  
  train_dataset, test_dataset = get_dataset(args.dataset)
  if args.pruning_type == 'global':
  	network, prune_ratios, optimizer, pretrain_iteration, finetune_iteration, batch_size = get_hyperparameters(args.network + '_global')
  else:
  	network, prune_ratios, optimizer, pretrain_iteration, finetune_iteration, batch_size = get_hyperparameters(args.network)  
  # load pre-trained network
  base_path = f'./checkpoint/{args.dataset}_{args.network}_{args.pruning_type}_{args.seed}'
  if not os.path.exists(base_path):
  	os.makedirs(base_path)  
  if not os.path.exists(os.path.join(base_path, 'base_model.pth')):
    print('Pre-train network')
  # pre-train network if not exits
    pre_train_acc, pre_train_loss,_ = test(network, train_dataset)
    pre_test_acc, pre_test_loss,oktime = test(network, test_dataset)
    train_acc, train_loss, test_acc, test_loss,_ = train(train_dataset, test_dataset, network, optimizer, pretrain_iteration, batch_size) 
 
    # save network and logs
    torch.save(network.state_dict(), os.path.join(base_path, 'base_model.pth'))
    with open(os.path.join(base_path, 'logs.txt'), 'w') as f:
    	f.write(f'{pre_train_loss:.3f}\t{pre_test_loss:.3f}\t{train_loss:.3f}\t{test_loss:.3f}\t'
    	        f'{pre_train_acc:.2f}\t{pre_test_acc:.2f}\t{train_acc:.2f}\t{test_acc:.2f}\n')
  else:
    print('Load pre-trained network')
    state_dict = torch.load(os.path.join(base_path, 'base_model.pth'))
    network.load_state_dict(state_dict) 
    
    pre_test_acc, pre_test_loss,X100 = test(network, test_dataset)

  # prune and fine-tune network
  exp_path = os.path.join(base_path, args.method)
  if not os.path.exists(exp_path):
  	os.makedirs(exp_path) 
  original_network = network  # keep the original network
  original_prune_ratio = prune_ratios  # keep the original prune ratio
  pruning_method = get_method(args.method)
  for it in range(args.pruning_iteration_start, args.pruning_iteration_end + 1):
    print(f'Pruning iter. {it}')

		# get pruning ratio for current iteration
		# list for layer-wise pruning, and constant for global pruning
    if args.pruning_type == 'oneshot':
	    network = deepcopy(original_network).cuda()
	    prune_ratios = []
	    for idx in range(len(original_prune_ratio)):
	  	  prune_ratios.append(1.0 - ((1.0 - original_prune_ratio[idx]) ** it))
    
    weights = network.get_weights()

    masks = network.get_masks() 

    
    masks = pruning_method(weights, masks, prune_ratios)

    pre_train_acc1, pre_train_loss1, intime = test(network, test_dataset)	
    network.set_masks(masks)

    sparsity = get_sparsity(network)
    pre_train_acc, pre_train_loss,_ = test(network, train_dataset)
    pre_test_acc, pre_test_loss,_ = test(network, test_dataset)
    train_acc, train_loss, test_acc, test_loss, intime2 = train(train_dataset, test_dataset, network, optimizer, finetune_iteration, batch_size)
    torch.save(network.state_dict(), os.path.join(base_path, 'prune{}.pth'.format(it)))
    print("inference latency now = ", intime2)
		# save network and logs
    with open(os.path.join(exp_path, 'logs.txt'), 'a') as f:
	    f.write(f'{it}\t{sparsity:.6f}\t'
			        f'{pre_train_loss:.3f}\t{pre_test_loss:.3f}\t{train_loss:.3f}\t{test_loss:.3f}\t'
			        f'{pre_train_acc:.2f}\t{pre_test_acc:.2f}\t{train_acc:.2f}\t{test_acc:.2f}\n')


if __name__ == '__main__':
	main()

