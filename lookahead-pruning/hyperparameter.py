from functools import partial
import network
from network import *

import torch.optim as optim

def get_hyperparameters(network_type):

  
  if network_type == 'resnet18':
  	network = MaskedResNet18().cuda()
  	prune_ratios = [0] + [.15] * 16 + [.10]
  	optimizer = partial(optim.Adam, lr=0.0003)
  	pretrain_iteration = 35000
  	finetune_iteration = 1000
  	batch_size = 60 
  
  elif network_type == 'distill':
    network = distill().cuda()
    prune_ratios = [.15] * 4 + [.10]
    optimizer = partial(optim.Adam, lr=0.0003)
    pretrain_iteration = 60000
    finetune_iteration = 40000
    batch_size = 60   
  else:
    raise ValueError('Unknown network') 
  return network, prune_ratios, optimizer, pretrain_iteration, finetune_iteration, batch_size