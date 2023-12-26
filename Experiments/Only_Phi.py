# %%
#%load_ext autoreload
#%autoreload 2

# %%
from pathlib import Path
root_dir = Path("../")
this_dir = root_dir / "Experiments"
import sys
sys.path.insert(0, str(root_dir.absolute()))

# %%
import constants
from src.abstract.abs_nn_theta import NNthHelper
from src.nn_theta import ResNETNNthHepler
import src.main_helper as main_helper
from src.abstract.abs_data import DataHelper
import torch
import pandas as pd
from collections import defaultdict
import utils.common_utils as cu
import tqdm
import numpy as np
import torch.utils.data as data_utils
from utils import torch_utils as tu
import copy
import time
import torch.nn as nn
#from plot_utils import plot_beta
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


# %%
dataset_name = constants.SHAPENET_NOISE_FULL
dh = main_helper.get_data_helper(dataset_name=dataset_name, logger=None)

# %%
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

# %%
# losses = []
# for id,x,y in dh._train.get_theta_loader(batch_size=1,shuffle=False):
#     x,y = x.to(device),y.to(device)
#     outputs = nnth_mh._model.forward(x)
#     criterion = nn.MSELoss()
#     losses.append(criterion(x,y))
    
pred_losses = nnth_mh.get_loaderlosses_perex(loader=dh._train_test.get_theta_loader(batch_size=32, shuffle=False))


# %%
ref_loss = torch.median(pred_losses)
pred_grp_losses = pred_losses.view(-1,4)
X = dh._train._X.view(-1,4,3,224,224)
y = dh._train._Beta.view(-1,4,3)

train_indices_X = []
train_indices_y = []
#y_train = torch.tensor([])

for grp_id,grp_losses in enumerate(pred_grp_losses):
    min_loss = torch.min(grp_losses)
    max_loss = torch.max(grp_losses)
    arg_min = torch.argmin(grp_losses)
    arg_max = torch.argmax(grp_losses)
    if min_loss < ref_loss and max_loss > ref_loss:
        #X_train = torch.cat((X_train,torch.unsqueeze(X[grp_id][arg_max],0)))
        #y_train = torch.cat((y_train,torch.unsqueeze(y[grp_id][arg_min],0)))
        train_indices_X.append(grp_id*4+int(arg_max))
        train_indices_y.append(grp_id*4+int(arg_min))


# %%
X_train = dh._train._X[train_indices_X]
y_train = dh._train._Beta[train_indices_y]


# %%
import torch.nn.functional as F

# Beta_0 = F.one_hot(y_train[:,0])
# Beta_1 = F.one_hot(y_train[:,1])
# Beta_2 = F.one_hot(y_train[:,2])
y_train = F.one_hot(y_train)
#y = torch.stack((Beta_0.view(-1,1,6),Beta_1.view(-1,1,3),Beta_2.view(-1,2,4)),1)

# %%
import utils.torch_data_utils as tdu

#losses = pred_losses.tolist()
loss_dataset = tdu.CustomThetaDataset(data_ids=torch.arange(int(X_train.shape[0])),X=X_train,y=y_train,transform=constants.RESNET_TRANSFORMS['train'])

dataloader = torch.utils.data.DataLoader(loss_dataset, batch_size=128,shuffle=True)
dataset_size = int(X_train.shape[0])

# %%
from  torchvision import models as tv_models
class ResNET(nn.Module):
    def __init__(self, out_dim, *args, **kwargs):
        super().__init__()
        self.out_dim = out_dim

        self.resnet_features =  tv_models.resnet18(pretrained=True)
        self.emb_dim = self.resnet_features.fc.in_features
        self.resnet_features.fc = nn.Identity()

        self.fc1 = nn.Linear(self.emb_dim, self.out_dim[0])
        self.fc2 = nn.Linear(self.emb_dim, self.out_dim[1])
        self.fc3 = nn.Linear(self.emb_dim, self.out_dim[2])

        self.sm = nn.Softmax(dim=1)

    def forward_proba(self, input):
        out1,out2,out3 = self.forward(input)
        return self.sm(out1),self.sm(out2),self.sm(out3)
    
    def forward(self, input):
        out1 = self.resnet_features(input)
        out2 = self.resnet_features(input)
        out3 = self.resnet_features(input)
        #print(out1.shape)
        return self.fc1(out1),self.fc2(out2),self.fc3(out3)
        
    
    def forward_labels(self, input):
        probs1,probs2,probs3 = self.forward_proba(input)
        probs1, labels1 = torch.max(probs1, dim=1)
        probs2, labels2 = torch.max(probs2, dim=1)
        probs3, labels3 = torch.max(probs3, dim=1)
        return labels1,labels2,labels3



# %%
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
        for _, inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            #labels = torch.reshape(labels,(labels.shape[0],1))
            labels = labels.to(dtype=torch.float32)
            #print(labels)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(True):
                out1,out2,out3 = model(inputs)
                #_, preds = torch.max(outputs, 1)
                #print(labels.shape)
                loss1 = criterion(out1, labels[:,0])
                loss2 = criterion(out2, labels[:,1])
                loss3 = criterion(out3, labels[:,2])
                loss = (loss1 + loss2 + loss3)/3
                
                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
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


# %%
from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

model_ft = ResNET(out_dim=[6,6,6])
#print(model_ft.resnet_features.fc.in_features)
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
#model_ft.fc = nn.Linear(num_ftrs, 1)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=30)
torch.save(model_ft,"../baselines/models/theirs_phi-full.pt")


# %%
