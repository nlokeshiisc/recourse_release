from pathlib import Path
root_dir = Path("../")
this_dir = root_dir / "Experiments"
import sys
sys.path.insert(0, str(root_dir.absolute()))

import constants
from src.abstract.abs_nn_theta import NNthHelper
from src.nn_theta import ResNETNNthHepler
import src.main_helper as main_helper
from src.abstract.abs_data import DataHelper
import torch
import pandas as pd
from collections import defaultdict
import utils.common_utils as cu
from tqdm import tqdm
import numpy as np
import torch.utils.data as data_utils
from utils import torch_utils as tu
import copy
import time
import torch.nn as nn
#from plot_utils import plot_beta
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def set_cuda_device(gpu_num: int):
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)

set_cuda_device(1)

device = torch.device("cuda:0")


dataset_name = constants.SHAPENET_NOISE_FULL
dh = main_helper.get_data_helper(dataset_name=dataset_name, logger=None)


nnth_args = {
    constants.BATCH_SIZE: 32,
}
nnth_type = constants.RESNET
nnth_name = "final-baseline-noise-full"
nnth_epochs = 1
fit_th = False
data_subset = constants.FULL_DATA
logger = None
nnth_mh = main_helper.fit_theta(nn_theta_type=nnth_type, models_defname=nnth_name,
                                    dh = dh, nnth_epochs=nnth_epochs,
                                    fit=fit_th, data_subset=data_subset, logger=logger, **nnth_args)

    
pred_losses = nnth_mh.get_loaderlosses_perex(loader=dh._train_test.get_theta_loader(batch_size=32, shuffle=False))

import utils.torch_data_utils as tdu

losses = pred_losses.tolist()
loss_dataset = tdu.CustomThetaDataset(data_ids=torch.arange(len(losses)),X=dh._train._X,y=losses,transform=constants.RESNET_TRANSFORMS['train'])

dataloader = torch.utils.data.DataLoader(loss_dataset, batch_size=128,shuffle=True)
dataset_size = len(losses)



def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    # best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        model.train()  # Set model to training mode
        

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        tq = tqdm(total=len(dataloader))

        for _, inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = torch.reshape(labels,(labels.shape[0],1))
            labels = labels.to(dtype=torch.float32)
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)

            tq.set_postfix({"Loss": loss.item()})
            tq.update(1)

        # running_corrects += torch.sum(preds == labels.data)
        scheduler.step()

        epoch_loss = running_loss / dataset_size
        # epoch_acc = running_corrects.double() / dataset_size

        print('Loss: {:.4f} '.format(epoch_loss))

        # deep copy the model
        # if epoch_acc > best_acc:
        #     best_acc = epoch_acc
        #     best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model


from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model_ft = models.resnet18(pretrained=False)
num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)
torch.save(model_ft,"../baselines/models/final-baseline-full-noise-losses.pt")
