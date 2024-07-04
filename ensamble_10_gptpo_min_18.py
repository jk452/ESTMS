
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
from torchvision import models
from torchvision import transforms
from torchvision.datasets import CIFAR100
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from pl_bolts.datamodules import CIFAR10DataModule
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
from pl_bolts.models.self_supervised import SimCLR
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
from pl_bolts.datamodules import CIFAR10DataModule
import pytorch_lightning as pl
#from torchmetrics.functional import accuracy
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
from pl_bolts.models.self_supervised import SimCLR
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
from pytorch_lightning.callbacks import LearningRateMonitor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform, SimCLREvalDataTransform

from pl_bolts.models.self_supervised import SimCLR
#from pytorch_lightning.metrics.functional import accuracy
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization

from torchmetrics import Accuracy
from optuna.samplers import RandomSampler
from pytorch_lightning.callbacks import ModelCheckpoint
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
import torch
from torch import optim, nn
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision.utils import make_grid
from torchvision import transforms as T
from torchvision import models, datasets
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
import numpy as np
from GPyOpt.methods import BayesianOptimization
import GPyOpt
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import torchvision
import numpy as np


import GPyOpt
from GPyOpt.methods import BayesianOptimization
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingLR

use_cuda = torch.cuda.is_available()
torch.cuda.empty_cache()

BATCHSIZE = 128
CLASSES = 10
EPOCHS = 20
DIR = os.getcwd()
#pl.seed_everything(42)
SEED = 42

np.random.seed(SEED)
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
        
        #swav = torch.hub.load('facebookresearch/swav:main', 'resnet50')  
        #self.backbone1 = nn.Sequential(*(list(swav.children())[:-2]))
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

class Net_student(nn.Module):

    def __init__(self,output_dim_fc1, output_dim_fc2, dropout_rate):
        super(Net_student, self).__init__()
        
        
        
        self.output_dim_fc1 = output_dim_fc1
        self.output_dim_fc2 = output_dim_fc2
        self.dropout_rate = dropout_rate
        Resnet18 = torchvision.models.resnet18(pretrained=False)
        #Resnet34 = torchvision.models.resnet34(pretrained=True)

        self.backbone_student = nn.Sequential(*(list(Resnet18.children())[:-3]))
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(256, output_dim_fc1),
            #nn.LayerNorm((320), eps=1e-05, elementwise_affine=True),
            nn.BatchNorm1d(output_dim_fc1),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            
        )
        self.fc2 = nn.Sequential(
            nn.Linear(output_dim_fc1, output_dim_fc2),
            #nn.LayerNorm((768), eps=1e-05, elementwise_affine=True),
            nn.BatchNorm1d(output_dim_fc2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
            
        )
        self.linear_layers = Linear(output_dim_fc2, CLASSES)
    # Defining the forward pass    
    def forward(self, x):
        x = self.backbone_student(x)  
        x = self.avgpool(x)
        #x = self.flatten(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.linear_layers(x)
        #return F.log_softmax(x, dim=1)
        return x




class LightningNet_student(pl.LightningModule):
    def __init__(self,output_dim_fc1, output_dim_fc2, dropout_rate, alpha, temperature,learning_rate):
        super().__init__()
        self.save_hyperparameters()


        self.student_model = Net_student(output_dim_fc1, output_dim_fc2, dropout_rate)


    def forward(self, data): 
        return self.student_model(data)

    def training_step(self,batch,batch_idx):
        data, target = batch
        student_output = self.student_model(data)
        with torch.no_grad():
            teacher_output = model_teacher(data)     
        
        kl_loss = nn.KLDivLoss(reduction="batchmean")(nn.LogSoftmax(dim=1)(student_output / self.hparams.temperature), 
                           nn.Softmax(dim=1)(teacher_output / self.hparams.temperature)) * self.hparams.temperature * self.hparams.temperature 
        ce_loss = nn.CrossEntropyLoss()(student_output, target)
        #c_barlo = BarlowTwinsLoss()(student_output, teacher_output)

        loss = self.hparams.alpha * kl_loss + (1 - self.hparams.alpha) * ce_loss 

        

        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        data, target = batch
        output = self.student_model(data)

      
        loss = cross_entropy(output, target) 
        pred = output.argmax(dim=1, keepdim=True)
        accuracy = pred.eq(target.view_as(pred)).float().mean()
        self.log("val_acc", accuracy)
        self.log("hp_metric", accuracy, on_step=False, on_epoch=True) 
        return {'val_loss': loss, 'val_acc': accuracy}  

    def configure_optimizers(self):

        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate,weight_decay=1e-4, momentum=0.9,nesterov=True)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones= [10,15,18], gamma=0.1)
        return [optimizer],[scheduler]

def objective(trial: optuna.trial.Trial) -> float:
    
    #logger = pl.loggers.TensorBoardLogger("tb_logs", name="swav_10")
    output_dim_fc1 = trial.suggest_int("output_dim_fc1", 64, 1024, step = 64)
    output_dim_fc2 = trial.suggest_int("output_dim_fc2", 64, 1024, step = 64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.6,step=.1)
    alpha = trial.suggest_float("alpha", 0.5, 0.9,step=.1)
    #beta = trial.suggest_float("beta", 0.01, 0.05,step=.01)
    temperature = trial.suggest_int("temperature", 2, 20, step = 2)
    learning_rate = trial.suggest_float("learning_rate", 0.01,0.5)
    #learning_rate = 0.00670619052983754
    #logger = DictLogger(trial.number)
    model = LightningNet_student(output_dim_fc1, output_dim_fc2, dropout_rate,alpha, temperature,learning_rate)
    #checkpoint_callback = ModelCheckpoint(monitor="val_acc")
    #datamodule = FashionMNISTDataModule(data_dir=DIR, batch_size=BATCHSIZE)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc")
    trainer = pl.Trainer(
        logger=True,
        enable_checkpointing = True,
        max_epochs=EPOCHS,
        gpus=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        #callbacks=[PyTorchLightningPruningCallback(trial, monitor="val_acc")],
    )
    hyperparameters = dict(output_dim_fc1 = output_dim_fc1,output_dim_fc2 = output_dim_fc2,dropout_rate=dropout_rate,
                           alpha=alpha,temperature=temperature,learning_rate = learning_rate)
    trainer.logger.log_hyperparams(hyperparameters)    
    trainer.fit(model, train_dataloader,val_dataloader)

    return trainer.callback_metrics["val_acc"].item()



study_name = "ensamble_100_min18_gptpo"  # Unique identifier of the study.
storage_name = "sqlite:///{}.db".format(study_name)
sampler = TPESampler(n_startup_trials=0,seed=SEED)
study = optuna.create_study(direction="maximize",sampler=sampler,study_name=study_name, storage=storage_name,load_if_exists=True)
study.optimize(objective, n_trials=2,timeout=None)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


print(df)
df.to_csv('ensamble_100_min18_gptpo.csv',index=False)

df_saved_files = pd.read_csv('ensamble_100_min18_gptpo.csv')
print(df_saved_files)











