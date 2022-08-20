import random
import numpy as np
from tqdm.notebook import tqdm
import torch
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler


class BatchSampler(Sampler):
	def __init__(self, dataset, num_iterations, batch_size):
		self.dataset = dataset
		self.num_iterations = num_iterations
		self.batch_size = batch_size

	def __iter__(self):
		for _ in range(self.num_iterations):
			indices = random.sample(range(len(self.dataset)), self.batch_size)
			yield indices

	def __len__(self):
		return self.num_iterations


def train(train_dataset, test_dataset, network, optimizer, num_iterations, batch_size, print_step=100):
  network.train()
  batch_sampler = BatchSampler(train_dataset, num_iterations, batch_size) 
  train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler, num_workers=4)
  optimizer = optimizer(network.parameters()) 
  for i, (x, y) in enumerate(tqdm(train_loader)):
    x = x.cuda()
    y = y.cuda()

    optimizer.zero_grad()
    out = network(x)
    loss = F.cross_entropy(out, y)
    loss.backward()
    optimizer.step()
    
    if (i + 1) % print_step == 0:

      test_acc, test_loss,train_time_test = test(network, test_dataset)

      print("Steps:", i + 1, "Test loss:", test_loss, "Test acc:", test_acc, "test time:", train_time_test)

      network.train()
  
  train_acc, train_loss,_ = test(network, train_dataset)
  test_acc, test_loss, intime = test(network, test_dataset)
  print(f'Train loss: {train_loss:.3f}\tTrain acc: {train_acc:.2f}\tTest loss: {test_loss:.3f}\tTest acc: {test_acc:.2f}')
  return train_acc, train_loss, test_acc, test_loss, intime


def test(network, dataset, batch_size=64):
  network.eval()
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4) 
  correct = 0
  loss = 0
  total_time = 0
  for i, (x, y) in enumerate(loader):
    x = x.cuda()
    y = y.cuda()  
    with torch.no_grad():
      a = time.time()
      out = network(x)
      b = time.time()
      total_time += b-a
      _, pred = out.max(1)  
    correct += pred.eq(y).sum().item()
    loss += F.cross_entropy(out, y) * len(x)  
  acc = correct / len(dataset) * 100.0
  loss = loss / len(dataset)  
  return acc, loss, total_time
