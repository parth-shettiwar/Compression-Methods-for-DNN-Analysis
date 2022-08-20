import torch
import torch.nn as nn
import os
import utils 
import sys
from network.masked_modules import MaskedLinear, MaskedConv2d

# setting path
# sys.path.append('../lookahead_pruning')
# from lookahead_pruning.utils import *
def is_base_module(m):
	if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		return True
	else:
		return False


def is_masked_module(m):
	if isinstance(m, MaskedLinear) or isinstance(m, MaskedConv2d):
		return True
	else:
		return False


def is_batch_norm(m):
	if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
		return True
	else:
		return False

class BaseModel(nn.Module):
  def __init__(self, init_scheme=None):
	  super(BaseModel, self).__init__()
	  self.init_scheme = init_scheme

  def get_weights(self):
    weights = []
    print(self.modules())
    for m in self.modules():
      # print(m.weight)
      if is_masked_module(m):
        weights.append(m.weight.data.cpu().detach())
    return weights

  def get_masks(self):
	  masks = []
	  for m in self.modules():
	  	if is_masked_module(m):
	  		masks.append(m.mask.cpu().detach())
	  return masks

  def set_weights(self, weights_to_set):
  	idx = 0
  	for m in self.modules():
  		if is_masked_module(m):
  			m.weight.data = weights_to_set[idx].cpu().to(m.weight.data.device)
  			idx += 1
  	assert idx == len(weights_to_set) 
  def set_masks(self, mask_to_set):
  	idx = 0
  	for m in self.modules():
  		if is_masked_module(m):
  			m.mask = mask_to_set[idx].cpu().to(m.mask.device)
  			m.weight.data *= m.mask
  			idx += 1
  	assert idx == len(mask_to_set)  
  def reinit(self):
  	for m in self.modules():
  		if is_masked_module(m):
  			self.init_scheme(m.weight.data)
  			if m.bias is not None:
  				nn.init.zeros_(m.bias.data)
  		elif isinstance(m, nn.BatchNorm2d):
  			m.weight.data.fill_(1)
  			m.bias.data.zero_() 
  def get_bn_weights(self):
  	weights = []
  	for m in self.modules():
  		if is_masked_module(m):
  			weights.append(None)
  		if is_batch_norm(m):
  			del weights[-1]
  			r_var = m.running_var.cpu().detach()
  			w = m.weight.cpu().detach()
  			weight = w / torch.sqrt(r_var + 0.0000000001)
  			weights.append(weight)  
  	return weights

