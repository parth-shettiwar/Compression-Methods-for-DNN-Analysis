"""Evaluates the model"""

import argparse
import logging
import os
import time
import numpy as np
import torch
from torch.autograd import Variable
import utils
import model.net as net
import model.resnet as resnet
import model.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/base_model', help="Directory of params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    

    model.eval()

    summ = []
    total_time = 0
    for data_batch, labels_batch in dataloader:

      
        data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        a = time.time()
        output_batch = model(data_batch)
        b = time.time()
        total_time += b-a
        loss = loss_fn(output_batch, labels_batch)

        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss.data[0]
        summ.append(summary_batch)
    print("inference latency = ",total_time)
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean



def evaluate_kd(model, dataloader, metrics, params):
    

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    total_time = 0
    for i, (data_batch, labels_batch) in enumerate(dataloader):

        if params.cuda:
            data_batch, labels_batch = data_batch.cuda(), labels_batch.cuda()
        data_batch, labels_batch = Variable(data_batch), Variable(labels_batch)
        
        a = time.time()
        output_batch = model(data_batch)
        b = time.time()
        total_time += b-a

        loss = 0.0  

        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        summary_batch = {metric: metrics[metric](output_batch, labels_batch)
                         for metric in metrics}
        summary_batch['loss'] = loss
        summ.append(summary_batch)
    print("inference latency = ", total_time)
    # compute mean of all metrics in summary
    metrics_mean = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]} 
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)
    return metrics_mean


