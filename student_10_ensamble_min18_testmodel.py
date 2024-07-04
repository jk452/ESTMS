
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
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler
from lightly.loss import BarlowTwinsLoss
use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()



BATCHSIZE = 128
CLASSES = 10
EPOCHS = 120
DIR = os.getcwd()
pl.seed_everything(42)
#SEED = 42

#np.random.seed(SEED)
torch.cuda.empty_cache()

#The mean and std of ImageNet are: mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. 
#(tensor([0.5055, 0.4740, 0.4245]), tensor([0.1780, 0.1751, 0.1755]))
transform_train = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),

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
   
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

CIFAR10_test = datasets.CIFAR10(root = DIR, train=False, download=False,transform=transform_test)
CIFAR10_full = datasets.CIFAR10(root = DIR, train=True, download=False)
#CIFAR10_train, CIFAR10_val = random_split(CIFAR10_full, [45000, 5000])
#CIFAR10_train.dataset.transform = transform_train
CIFAR10_full.transform = transform_train
#CIFAR10_val.dataset.transform = transform_test
train_dataloader = DataLoader(CIFAR10_full, batch_size=BATCHSIZE,num_workers = 48, shuffle=True)
#val_dataloader = DataLoader(CIFAR10_val, batch_size=BATCHSIZE, shuffle=True)
val_dataloader = DataLoader(CIFAR10_test, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)
test_dataloader = DataLoader(CIFAR10_test, batch_size=BATCHSIZE,num_workers = 48, shuffle=False)

class Net_teacher(nn.Module):   
    def __init__(self):
        super(Net_teacher, self).__init__()
        
        Resnet50_swav = torchvision.models.resnet50(pretrained=False)  
        self.backbone1 = nn.Sequential(*(list(Resnet50_swav.children())[:-2]))

        #weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt'
        #simclr = SimCLR.load_from_checkpoint(weight_path, strict=False) 
        #self.backbone2 = nn.Sequential(*(list(simclr.encoder.children())[:-2])) 
        Resnet50_simclr = torchvision.models.resnet50(pretrained=False)  
        self.backbone2 = nn.Sequential(*(list(Resnet50_simclr.children())[:-2]))
        #barlotwins = torch.hub.load('facebookresearch/barlowtwins:main', 'resnet50')
        #self.backbone3 = nn.Sequential(*(list(barlotwins.children())[:-2]))    
        Resnet50_barlo = torchvision.models.resnet50(pretrained=False)  
        self.backbone3 = nn.Sequential(*(list(Resnet50_barlo.children())[:-2]))        
        #for param in self.backbone.parameters():
            #param.requires_grad = False
       

        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        self.linear_layers = Linear(6144, CLASSES)

    # Defining the forward pass    

    def forward(self, x):

        u = self.backbone1(x)
        v = self.backbone2(x)
        w = self.backbone3(x)
        
        u = self.avgpool(u)
        v = self.avgpool(v)
        w = self.avgpool(w)
        
        u = u.view(x.size(0), -1)
        v = v.view(x.size(0), -1)
        w = w.view(x.size(0), -1)
        y = torch.cat((u, v, w), dim=1)
        y = self.linear_layers(y)
        
        return y

class LightningNet_teacher(pl.LightningModule):
    def __init__(self):
        super(LightningNet_teacher,self).__init__()
        #self.save_hyperparameters()
        self.teacher_model = Net_teacher()
    def forward(self, data): 
        return self.teacher_model(data)

chk_path = "t_ensamble_10.ckpt"
model_teacher = LightningNet_teacher.load_from_checkpoint(chk_path)
model_teacher.cuda()
#model_teacher.eval()
#back_teacher = nn.Sequential(*(list(model_teacher.children())[:0]))
loss_fn1 = nn.CrossEntropyLoss()
def loss_fn(student_output, teacher_output, labels, alpha, beta, temperature):
  # Compute the KL divergence loss
  #soft_targets = nn.functional.softmax(teacher_outputs/temperature, dim=1)
  # Compute the cross-entropy loss between the student outputs and the soft targets
  #distillation_loss = nn.functional.kl_div(nn.functional.log_softmax(student_outputs/temperature, dim=1), soft_targets, reduction='batchmean')
  kl_loss = nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(student_output / temperature), 
                           nn.Softmax(dim=1)(teacher_output / temperature)) * temperature * temperature
  #kl_loss = F.kl_div(nn.LogSoftmax(dim=1)(student_output / temperature), nn.Softmax(dim=1)(teacher_output / temperature), reduction='batchmean') * temperature * temperature
  # Compute the cross-entropy loss
  ce_loss = nn.CrossEntropyLoss()(student_output, labels)
  #ce_loss = 2
  # Compute the cosine similarity loss
  #cs_loss = torch.mean(1 - nn.CosineSimilarity(dim=1)(student_output  , teacher_output ))
  #cs_loss = nn.CosineEmbeddingLoss()(student_embed, teacher_embed, ones)
  #cs_loss = nn.CosineEmbeddingLoss()(student_output, student_output, ones)
  #cs_loss = torch.mean(1 - nn.CosineSimilarity(dim=1)(nn.Softmax(dim=1)(student_output), nn.Softmax(dim=1)(teacher_output)))
  c_barlo = BarlowTwinsLoss()(student_output, teacher_output)
  #cs_loss = torch.mean(1 - nn.CosineSimilarity(dim=1)(student_output, teacher_output))
  #cs_loss1 = torch.mean((cs_loss - 1) ** 2)
  # Combine the losses using a weighted sum
  #loss = alpha * kl_loss + (1 - alpha) * (beta*ce_loss + (1-beta)*cs_loss)
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * cs_loss
  #loss = alpha * kl_loss + (1 - alpha) * (beta*ce_loss + (1-beta)*cs_loss)
  #loss =  ce_loss
  #loss = alpha * kl_loss + (1 - alpha) * ce_loss 
  loss = alpha * kl_loss + (1 - alpha) * ce_loss + beta * c_barlo
  #loss = ce_loss 
  return loss


class Net_student(nn.Module):   
    def __init__(self):
        super(Net_student, self).__init__()
        #Resnet18 = torchvision.models.resnet18(pretrained=False)
        Resnet34 = torchvision.models.resnet34(pretrained=True)

        self.backbone_student = nn.Sequential(*(list(Resnet34.children())[:-3]))
        #for param in self.backbone_student.parameters():
            #param.requires_grad = False
        #self.backbone_student = nn.Sequential(*(list(Resnet34.children())[:-2]))
        #input_feature = student_model.fc.in_features

        '''self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
             
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=1),
            #Defining another 2D convolution layer
            Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(512),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=3, stride=1),
            #Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            #BatchNorm2d(1024),
            #ReLU(inplace=True),
            #MaxPool2d(kernel_size=2, stride=1),
        )'''
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(256, 640),
            #nn.LayerNorm((n_units), eps=1e-05, elementwise_affine=True),
            nn.BatchNorm1d(640),
            nn.ReLU(),
            nn.Dropout(p = .1)
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(640, 1024),
            #nn.LayerNorm((896), eps=1e-05, elementwise_affine=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p = .1)
            
        )
        '''self.fc3 = nn.Sequential(
            nn.Linear(512, 1024),
            #nn.LayerNorm((768), eps=1e-05, elementwise_affine=True),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p = .4)
            
        )'''
        self.linear_layers = Linear(1024, CLASSES)

    # Defining the forward pass    

    def forward(self, x):
        x = self.backbone_student(x)
        #x = self.cnn_layers(x)  
        x = self.avgpool(x)
        #x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        x = self.linear_layers(x)
        #return F.log_softmax(x, dim=1)
        return x
    def get_embed(self, x):
        x = self.backbone_student(x)
        #x = self.cnn_layers(x)  
        x = self.avgpool(x)
        #x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        #x = self.fc3(x)
        return x

class LightningNet_student(pl.LightningModule):
    def __init__(self):
        super(LightningNet_student,self).__init__()
        #self.save_hyperparameters()
        #chk_path = "teacher1.ckpt"
        #self.model_teacher = LightningNet_teacher.load_from_checkpoint(chk_path)
        self.student_model = Net_student()
        #self.backbone_s = nn.Sequential(*(list(self.student_model.children())[:-1]))
        

    def forward(self, data): 
        return self.student_model(data)

    def training_step(self,batch,batch_idx):
        data, target = batch

        student_output = self.student_model(data)
        #student_embed = self.student_model.get_embed(data)
        #student_embed = self.backbone_s(data)                  
        with torch.no_grad():
            teacher_output = model_teacher(data)
            #teacher_embed =  model_teacher.get_embed(data)
            #teacher_embed = back_teacher(data)
        #print(type(teacher_output))
        #print(type(student_output))
        #ones = torch.ones(data.shape[0])
        #ones = torch.ones(len(data)).to(data.device)
        #ones = torch.ones(len(x)).to(x.device)
        # Compute the loss and gradients
        loss = loss_fn(student_output, teacher_output, target , alpha=0.75, beta= .01, temperature=4)
        #loss = loss_fn1(student_output, target) 
        return loss
        #return loss_fn1

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
        #optimizer = torch.optim.Adam(self.parameters(), lr=.001,weight_decay=1e-2)
        #optimizer = torch.optim.Adagrad(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        #optimizer = torch.optim.Adadelta(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4)
        #Scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
        #optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4, momentum=0.9,nesterov=True)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=EPOCHS)
        #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        #learning_rate = trial.suggest_float("learning_rate", 1e-4,1e-1, log=True)
        optimizer = torch.optim.SGD(self.parameters(), lr=0.08,weight_decay=1e-4, momentum=0.9,nesterov=True)
        #optimizer = torch.optim.Adam(self.parameters(), lr=.005,weight_decay=1e-4)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,.005, epochs=EPOCHS, 
                                                #steps_per_epoch=len(train_loader))
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=1, eta_min=1e-6, last_epoch=- 1, verbose=False)
        #scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [150,180,210], gamma=0.1)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=EPOCHS)

        #optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        #scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=EPOCHS)
        #scheduler = pl_bolts.optimizers.lr_scheduler.LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=EPOCHS)
        #scheduler = MultiStepLR(optimizer, milestones=[12,16,20], gamma=.1)
        return [optimizer],[scheduler]
    
model = LightningNet_student()



chk_path = "student_10_ensamble_34_min_kd.ckpt"
model_st = LightningNet_student.load_from_checkpoint(chk_path)
model_st.cuda()

trainer=pl.Trainer(
    logger=True,
    #checkpoint_callback=model.configure_checkpoint_callback(),
    #callbacks=[checkpoint_callback],
    #progress_bar_refresh_rate=10,
    
    #limit_train_batches=0.25,
    #max_epochs=EPOCHS,
    #callbacks=[early_stop_callback],
    gpus=1 if torch.cuda.is_available() else None,
    #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
result = trainer.test(model_st, test_dataloader, verbose=True)
print(result)









