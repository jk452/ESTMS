
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets


import os
from random import randint
import urllib
import zipfile
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
import plotly
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
#from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
#from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
import torch.optim
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
#from optuna.samplers import TPESampler
import plotly
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

#from pl_bolts.models.self_supervised import SimCLR
#from pytorch_lightning.metrics.functional import accuracy
#from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics import Accuracy
from optuna.samplers import RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import pandas as pd
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
from pl_bolts.models.self_supervised import SimCLR
import numpy as np
from torchvision.transforms.functional import InterpolationMode
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets

from torchvision import transforms
from torch.optim import Adam
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from optuna.samplers import TPESampler
import plotly
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
#from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
#from torchmetrics.functional import accuracy
import argparse
import os
from typing import List
from typing import Optional
import logging
import sys
import optuna
#from optuna.integration import PyTorchLightningPruningCallback
from packaging import version
import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import datasets
from torchvision import transforms
from torch.optim import Adam
import torch.optim
from torch.nn.functional import cross_entropy
#from pl_bolts.models.self_supervised import SimCLR
import optuna
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice
from torchvision import datasets
#from optuna.samplers import TPESampler
import plotly
#from pytorch_lightning.callbacks import LearningRateMonitor
#from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
#from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

#from pl_bolts.models.self_supervised import SimCLR
#from pytorch_lightning.metrics.functional import accuracy
#from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics import Accuracy
from optuna.samplers import RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
import pandas as pd
import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import numpy as np
import torch
import numpy as np
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt

import torch
import numpy as np
import torchvision
from torch import nn

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from lightly.loss import BarlowTwinsLoss
from PIL import Image, ImageOps, ImageFilter

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()



BATCHSIZE = 128
CLASSES = 10
EPOCHS = 100
DIR = os.getcwd()
pl.seed_everything(42)



torch.cuda.empty_cache()




transform_train = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),

    transforms.RandomHorizontalFlip(p=.5),
    transforms.RandomRotation(15),
    transforms.RandomApply([
        transforms.ColorJitter(brightness=0.5,
                               contrast=0.5,
                               saturation=0.5,
                               hue=0.1)
        ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
transform_test = transforms.Compose([
   
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

CIFAR10_test = datasets.CIFAR10(root = DIR, train=False, download=False,transform=transform_test)
CIFAR10_full = datasets.CIFAR10(root = DIR, train=True, download=False)

CIFAR10_full.transform = transform_train

train_dataloader = DataLoader(CIFAR10_full, batch_size=BATCHSIZE,num_workers = 48, shuffle=True)

val_dataloader = DataLoader(CIFAR10_test, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)
test_dataloader = DataLoader(CIFAR10_test, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)

#CIFAR10_train, CIFAR10_val = random_split(CIFAR10_full, [45000, 5000])
#CIFAR10_train.dataset.transform = transform_train
#CIFAR10_full.transform = transform_train
#CIFAR10_val.dataset.transform = transform_test
#train_dataloader = DataLoader(CIFAR10_train, batch_size=BATCHSIZE,num_workers = 48, shuffle=True)
#val_dataloader = DataLoader(CIFAR10_val, batch_size=BATCHSIZE, shuffle=True)
#val_dataloader = DataLoader(CIFAR10_val, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)
#test_dataloader = DataLoader(CIFAR10_test, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)

class Net_teacher(nn.Module):   
    def __init__(self):
        super(Net_teacher, self).__init__()
        Resnet50 = torchvision.models.resnet50(pretrained=False)  
        self.backbone = nn.Sequential(*(list(Resnet50.children())[:-2]))
        #barlotwins = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')


        self.avgpool = nn.AdaptiveAvgPool2d((1,1))


        self.linear_layers = Linear(2048, CLASSES)

    # Defining the forward pass    

    def forward(self, x):

        x = self.backbone(x)

        
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)

        x = self.linear_layers(x)

        return x
        
    def get_embed(self, x):
        x = self.backbone(x)

        
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)


        return x

class LightningNet_teacher(pl.LightningModule):
    def __init__(self):
        super(LightningNet_teacher,self).__init__()

        self.teacher_model = Net_teacher()
    def forward(self, data): 
        return self.teacher_model(data)

chk_path = "t_simclr_10_new.ckpt"
model_teacher = LightningNet_teacher.load_from_checkpoint(chk_path)
model_teacher.cuda()

loss_fn1 = nn.CrossEntropyLoss()


def loss_fn(student_output, teacher_output, labels, alpha, beta, delta, temperature):
  # Compute the KL divergence loss
  #soft_targets = nn.functional.softmax(teacher_outputs/temperature, dim=1)
  # Compute the cross-entropy loss between the student outputs and the soft targets
  #distillation_loss = nn.functional.kl_div(nn.functional.log_softmax(student_outputs/temperature, dim=1), soft_targets, reduction='batchmean')
  kl_loss = nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(student_output / temperature), 
                           nn.Softmax(dim=1)(teacher_output / temperature)) * temperature * temperature
  
  #kl_loss = (kl_loss1 + kl_loss2) / 2
  #kl_loss = F.kl_div(nn.LogSoftmax(dim=1)(student_output / temperature), nn.Softmax(dim=1)(teacher_output / temperature), reduction='batchmean') * temperature * temperature
  # Compute the cross-entropy loss
  ce_loss = nn.CrossEntropyLoss()(student_output, labels)
  #student_output_norm = (student_output - torch.mean(student_output, dim=0)) / torch.std(student_output, dim=0)
  #teacher_output_norm = (teacher_output - torch.mean(teacher_output, dim=0)) / torch.std(teacher_output, dim=0)
  #cross_corr = torch.matmul(student_output_norm.T, teacher_output_norm) / 128
  #off_diag = off_diagonal_ele(cross_corr).pow_(2).sum()
  #on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
  #c_barlo = on_diag + 0.005 * off_diag

  #c_barlo = BarlowTwinsLoss()(student_output, teacher_output)
  c_barlo = BarlowTwinsLoss()(teacher_output, student_output)
  #cnt_loss = NTXentLoss()(student_output, teacher_output)
  cs_loss = torch.mean(1 - nn.CosineSimilarity(dim=1)(student_output, teacher_output))
  #cs_loss1 = torch.mean((cs_loss - 1) ** 2)
  # Combine the losses using a weighted sum
  #loss = alpha * kl_loss + (1 - alpha) * (beta*ce_loss + (1-beta)*cs_loss)
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * cs_loss
  #loss = alpha * kl_loss + (1 - alpha) * (beta*ce_loss + (1-beta)*cs_loss)
  #loss =  ce_loss
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss 
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * c_barlo
  loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * c_barlo + delta * cs_loss
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * c_barlo + (1-beta) * cs_loss
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss  + delta * cs_loss
  #loss = ce_loss 
  return loss



class Net_student(nn.Module):   
    def __init__(self):
        super(Net_student, self).__init__()
        Resnet18 = torchvision.models.resnet18(pretrained=False)
        #Resnet34 = torchvision.models.resnet34(pretrained=True)

        self.backbone_student = nn.Sequential(*(list(Resnet18.children())[:-2]))



        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(512, 704),

            nn.BatchNorm1d(704),
            nn.ReLU(), 
            nn.Dropout(p = .1)
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(704, 640),

            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(p = .1)
            
        )

        self.linear_layers = Linear(640, CLASSES)

    # Defining the forward pass    

    def forward(self, x):
        x = self.backbone_student(x)
 
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        x = self.linear_layers(x)

        return x
    def get_embed(self, x):
        x = self.backbone_student(x)
 
        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

class LightningNet_student(pl.LightningModule):
    def __init__(self):
        super(LightningNet_student,self).__init__()

        self.student_model = Net_student()

        

    def forward(self, data): 
        return self.student_model(data)

    def training_step(self,batch,batch_idx):
        data, target = batch

        student_output = self.student_model(data)
                
        with torch.no_grad():
            teacher_output = model_teacher(data)

        loss = loss_fn(student_output, teacher_output, target , alpha=0.7, beta= .09, delta = 6, temperature=18)
        #loss = loss_fn(student_output, teacher_output, target , alpha=0.5, beta= .05, delta = 8, temperature=18)
        return loss


    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self.student_model(data)
        #features = features.view(features.size(0), -1)
      
        loss = loss_fn1(output, target) 
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True) 
        return {'val_loss': loss, 'val_acc': accuracy}      
       
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()  
        Accuracy = 100 * avg_acc.item()
        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_acc}
        print('Val Loss:', round(avg_loss.item(),2), 'Val Accuracy: %f %%' % Accuracy) 
        return {'avg_val_loss': avg_loss, 'progress_bar': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        data, target = batch

        output = self(data)
        loss = loss_fn1(output, target)
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        print(accuracy)
        self.log("test_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True) 
        return {'test_loss': loss, 'test_acc': accuracy}  

    def configure_optimizers(self):

        #optimizer = torch.optim.SGD(self.parameters(), lr=0.0320003854172148,weight_decay=1e-4, momentum=0.9,nesterov=True)
        optimizer = torch.optim.SGD(self.parameters(), lr = .1,weight_decay=1e-4, momentum=0.9,nesterov=True)
        #scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [75,90], gamma=0.1)
 
        return [optimizer],[scheduler]
    
model = LightningNet_student()

checkpoint_callback = ModelCheckpoint(
     monitor='val_acc',
     dirpath=DIR,
     filename='student_10_simclr_gptpo_18_new_btcs',
     mode='max'
 )
trainer=pl.Trainer(
    logger=True,
    callbacks=[checkpoint_callback],

    max_epochs=EPOCHS,

    gpus=1 if torch.cuda.is_available() else None,

    )
trainer.fit(model, train_dataloader,val_dataloader)
#result = trainer.test(model, test_loader, verbose=True)
#print(result)


checkpoint_callback.best_model_path










