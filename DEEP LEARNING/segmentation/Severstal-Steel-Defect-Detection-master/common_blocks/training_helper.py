import os
from sklearn.model_selection import StratifiedKFold
import cv2
import pdb
import time
import warnings
import random
import numpy as np
import pandas as pd
from tqdm import tqdm as tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
from matplotlib import pyplot as plt
from .metric import Meter, epoch_log
from .dataloader import provider_cv, provider_trai_test_split
import sys
from .losses import BCEDiceLoss, FocalLoss, JaccardLoss, DiceLoss
from .lovasz_losses import LovaszLoss, LovaszLossSymmetric

sys.path.append("..")
from configs.train_params import *
from .optimizers import RAdam, Over9000, Adam


class Trainer_cv(object):
    """This class takes care of training and validation of our model"""

    def __init__(
        self,
        model,
        num_epochs,
        current_fold=0,
        batch_size={"train": 4, "val": 4},
        optimizer_state=None,
    ):
        self.current_fold = current_fold
        self.total_folds = TOTAL_FOLDS
        self.num_workers = 4
        self.batch_size = batch_size
        self.accumulation_steps = 32 // self.batch_size["train"]
        self.lr = LEARNING_RATE
        self.num_epochs = num_epochs
        self.best_metric = INITIAL_MINIMUM_DICE  # float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model  # torch.nn.BCEWithLogitsLoss()
        self.criterion = (
            BCEDiceLoss()
        )  # JaccardLoss()#LovaszLossSymmetric(per_image=True, classes=[0,1,2,3])
        # BCEDiceLoss()  # BCEDiceLoss()#FocalLoss(num_class=4)  # BCEDiceLoss()  # torch.nn.BCEWithLogitsLoss()
        self.optimizer = RAdam(
            [
                {"params": self.net.decoder.parameters(), "lr": self.lr},
                {"params": self.net.encoder.parameters(), "lr": self.lr},
            ]
        )  # optim.Adam(self.net.parameters(), lr=self.lr)

        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, factor=0.9, mode="min", patience=3, verbose=True
        )
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider_cv(
                fold=self.current_fold,
                total_folds=self.total_folds,
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        # self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
            tk0.update(1)
            tk0.set_postfix(loss=(running_loss / (itr + 1)))
        tk0.close()
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        # self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss, dice

    def start(self):
        epoch_wo_improve_score = 0
        for epoch in range(self.num_epochs):
            if EARLY_STOPING is not None and epoch_wo_improve_score >= EARLY_STOPING:
                print("Early stopping {}".format(EARLY_STOPING))
                torch.save(
                    state,
                    "./model_weights/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
                        unet_encoder, self.current_fold, epoch, val_dice
                    ),
                )
                break
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_metric": self.best_metric,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            val_loss, val_dice = self.iterate(epoch, "val")
            self.scheduler.step(val_loss)
            if val_dice > self.best_metric:
                print("******** New optimal found, saving state ********")
                state["best_metric"] = self.best_metric = val_dice
                torch.save(
                    state,
                    "./model_weights/model_{}_fold_{}_epoch_{}_dice_{}.pth".format(
                        unet_encoder, self.current_fold, epoch, val_dice
                    ),
                )
                epoch_wo_improve_score = 0
            else:
                epoch_wo_improve_score += 1
            print()
        if num_epochs > 1:
            torch.save(
                state,
                "./model_weights/model_{}_fold_{}_last_epoch_{}_dice_{}.pth".format(
                    unet_encoder, self.current_fold, epoch, val_dice
                ),
            )


""" WARNING DEPRECATED
class Trainer_split(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model):
        self.num_workers = 6
        self.batch_size = {"train": 4, "val": 4}
        self.accumulation_steps = 32 // self.batch_size['train']
        self.lr = 5e-4
        self.num_epochs = 20
        self.best_loss = float("inf")
        self.phases = ["train", "val"]
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode="min", patience=3, verbose=True)
        self.net = self.net.to(self.device)
        cudnn.benchmark = True
        self.dataloaders = {
            phase: provider_trai_test_split(
                data_folder=data_folder,
                df_path=train_df_path,
                phase=phase,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                batch_size=self.batch_size[phase],
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        # self.iou_scores = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}

    def forward(self, images, targets):
        images = images.to(self.device)
        masks = targets.to(self.device)
        outputs = self.net(images)
        loss = self.criterion(outputs, masks)
        return loss, outputs

    def iterate(self, epoch, phase):
        meter = Meter(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print(f"Starting epoch: {epoch} | phase: {phase} | ⏰: {start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        #         tk0 = tqdm(dataloader, total=total_batches)
        self.optimizer.zero_grad()
        for itr, batch in enumerate(dataloader):  # replace `dataloader` with `tk0` for tqdm
            images, targets = batch
            loss, outputs = self.forward(images, targets)
            loss = loss / self.accumulation_steps
            if phase == "train":
                loss.backward()
                if (itr + 1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            meter.update(targets, outputs)
        #             tk0.set_postfix(loss=(running_loss / ((itr + 1))))
        epoch_loss = (running_loss * self.accumulation_steps) / total_batches
        dice = epoch_log(phase, epoch, epoch_loss, meter, start)
        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(dice)
        # self.iou_scores[phase].append(iou)
        torch.cuda.empty_cache()
        return epoch_loss

    def start(self):
        for epoch in range(self.num_epochs):
            self.iterate(epoch, "train")
            state = {
                "epoch": epoch,
                "best_metric": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch, "val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                # TODO save weights on last epoch too
                print("******** New optimal found, saving state ********")
                state["best_metric"] = self.best_loss = val_loss
                torch.save(state, "./model.pth")
            print()
"""
