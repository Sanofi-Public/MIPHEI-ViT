"""General-purpose training script for image-to-image translation.

This script works for various models (with option '--model': e.g., pix2pix, cyclegan, colorization) and
different datasets (with option '--dataset_mode': e.g., aligned, unaligned, single, colorization).
You need to specify the dataset ('--dataroot'), experiment name ('--name'), and model ('--model').

It first creates model, dataset, and visualizer given the option.
It then does standard network training. During the training, it also visualize/save the images, print/save the loss plot, and save models.
The script supports continue/resume training. Use '--continue_train' to resume your previous training.

Example:
    Train a CycleGAN model:
        python train.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan
    Train a pix2pix model:
        python train.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/train_options.py for more training options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import pyvips
import time
import torch
from options.train_options import TrainOptions
from options.test_options import ValOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from post_process import validation_train
from torch.utils.data import random_split
from util import util
from data.aligned_dataset import AlignedDataset
import csv
from tqdm import tqdm
import os

import torch
import random
import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    set_seed(seed=42)
    dataset = create_dataset(opt)
    n_train = len(dataset)
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    # Update opt for the validation phase
    opt.phase = 'val'
    dataset_val = create_dataset(opt)
    n_val = len(dataset_val)
    print('The number of validation images = %d' % n_val)


    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    f = open('./checkpoints/' + '%s/' % opt.name + 'validation_train.csv', 'w', encoding='utf-8', newline='') # record validation result
    csv_writer = csv.writer(f)
    csv_writer.writerow(['epoch', 'dapi', 'cd3', 'panck', 'average'])

    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        print('TRAINING')
        for i, data in tqdm(enumerate(dataset), total=len(dataset), desc="Training Epoch %d" % epoch):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)
                if opt.display_id == 0:
                    visualizer.novisdom_plot_losses(epoch, float(epoch_iter) / dataset_size, losses, opt.n_epochs + opt.n_epochs_decay)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        model.update_learning_rate()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        if epoch % opt.val_freq == 0:  #run validation on the validation set
            average = 0
            with torch.no_grad():
                print('VALIDATION')
                for i, data_val in tqdm(enumerate(dataset_val), total=len(dataset_val), desc="Validating Epoch %d" % epoch):
                    imgs = data_val['A']
                    truemasks = data_val['B']
                    imgs = imgs.to(device='cuda',dtype=torch.float)
                    net = getattr(model, 'net' + 'G')
                    maskpred = net(imgs)
                    #maskpred = maskpred.cpu().numpy()
                    #truemasks = truemasks.cpu().numpy()
                    _, average_score = validation_train(truemasks, maskpred)
                    average += average_score
                csv_writer.writerow([epoch, average/n_val])
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    f.close()
