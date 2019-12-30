# coding=utf-8

import torch
import torch.nn as nn
import numpy as np
import argparse
from torch.utils.data import DataLoader
from imgaug import augmenters as iaa
import tqdm
import os

import lit.logger as logger
from lit.model import unet
from lit.loss import dice_loss
from lit.metrics import get_ious


def main(args):
    # tensorboard
    logger_tb = logger.Logger(log_dir=args.experiment_name)

    # TODO: Create Dataset.
    # get dataset

    # if args.dataset == "nuclei":
    #     train_dataset = NucleiDataset(args.train_data, 'train', args.transform, args.target_channels)
    # elif args.dataset == "hpa":
    #     train_dataset = HPADataset(args.train_data, 'train', args.transform, args.max_mean, args.target_channels)
    # elif args.dataset == "hpa_single":
    #     train_dataset = HPASingleDataset(args.train_data, 'train', args.transform)
    # else:
    #     train_dataset = NeuroDataset(args.train_data, 'train', args.transform)

    if args.dataet == "lit":
        train_dataset = "lit"
    # create data loader
    train_params = {'batch_size': args.batch_size,
                    'shuffle': False,
                    'num_workers': args.num_workers}
    train_dataloader = DataLoader(train_dataset, **train_params)

    # model
    model = unet(args.num_kernel, args.kernel_size, train_dataset.dim, train_dataset.target_dim)

    # device
    device = torch.device(args.device)
    if args.device == "cuda":
        # parse gpu_ids for data parallel
        if ',' in args.gpu_ids:
            gpu_ids = [int(ids) for ids in args.gpu_ids.split(',')]
        else:
            gpu_ids = int(args.gpu_ids)
        # parallelize computation
        if type(gpu_ids) is not int:
            model = nn.DataParallel(model, gpu_ids)
    model.to(device)

    # optimizer
    parameters = model.parameters()
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(parameters, args.lr)
    else:
        optimizer = torch.optim.SGD(parameters, args.lr)

    # loss
    loss_function = dice_loss

    count = 0
    # train model
    for epoch in range(args.epoch):
        model.train()

        with tqdm.tqdm(total=len(train_dataloader.dataset), unit=f"epoch {epoch} itr") as progress_bar:
            total_loss = []
            total_iou = []
            for i, (x_train, y_train) in enumerate(train_dataloader):
                with torch.set_grad_enabled(True):

                    # send data to device
                    x = torch.Tensor(x_train.float()).to(device)
                    y = torch.Tensor(y_train.float()).to(device)

                    # predict segmentation
                    pred = model.forward(x)

                    # calculate loss
                    loss = loss_function(pred, y)
                    total_loss.append(loss.item())

                    # calculate IoU
                    # to numpy array
                    predictions = pred.clone().squeeze().detach().cpu().numpy()
                    gt = y.clone().squeeze().detach().cpu().numpy()
                    ious = [get_ious(p, g, 0.5) for p, g in zip(predictions, gt)]
                    total_iou.append(np.mean(ious))

                    # back prop
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # mean loss and iou
                avg_loss = np.mean(total_loss)
                avg_iou = np.mean(total_iou)

                logger_tb.update_value("train loss", avg_loss, count)
                logger_tb.update_value("train iou", avg_iou, count)

                # display segmentation on tensorboard
                if i == 0:
                    original = x_train[0].squeeze()
                    truth = y_train[0].squeeze()
                    seg = pred[0].cpu().squeeze().detach().numpy()

                    # TODO display segmentations based on number of ouput
                    logger_tb.update_image("truth", truth, count)
                    logger_tb.update_image("segmentation", seg, count)
                    logger_tb.update_image("original", original, count)

                    count += 1
                    progress_bar.update(len(x))

    # save model
    ckpt_dict = {'model_name': model.__class__.__name__,
                 'model_args': model.args_dict(),
                 'model_state': model.to('cpu').state_dict()
                 }
    ckpt_path = os.path.join(args.save_dir, "saved_name")
    torch.save(ckpt_dict, ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_kernel', type=int, default=8)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_data', type=str, default="PATH_TO_TRAIN_DATA")
    parser.add_argument('--save_dir', type=str, default="./")
    parser.add_argument('--dataset', type=str, default="Hpa")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--model', type=str, default='unet')
    parser.add_argument('--max_mean', type=str, default='max')
    parser.add_argument('--target_channels', type=str, default='0,2,3')
    parser.add_argument('--batch_size', type=int, default='8')
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--gpu_ids', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default='16')
    parser.add_argument('--experiment_name', type=str, default='test')

    args = parser.parse_args()
    main(args)
