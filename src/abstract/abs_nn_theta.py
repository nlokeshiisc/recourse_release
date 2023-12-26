from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import utils.common_utils as cu
from torch.utils.tensorboard import SummaryWriter
import constants as constants
from src.abstract.abs_data import DataHelper, Data

class NNthHelper(ABC):
    def __init__(self, model:nn.Module, dh:DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.dh = dh
        self.__model_copy = None
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16
        self.momentum = 0
        self.lr_scheduler = None

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):

        self.kwargs = kwargs

        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.MOMENTUM in kwargs.keys():
            self.momentum = kwargs[constants.MOMENTUM]

    def __init_loaders(self):
        self.trn_loader = self._dh._train.get_theta_loader(batch_size=self.batch_size, shuffle=True)
        self.tst_loader = self._dh._test.get_theta_loader(shuffle=False, batch_size=128)
        self.val_loader = self._dh._val.get_theta_loader(shuffle=False, batch_size=128)

# %% properties
    @property
    def _model(self) -> nn.Module:
        return self.model
    @_model.setter
    def _model(self, value):
        self.model = value

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _batch_size(self):
        return self.batch_size

    @property
    def _trn_loader(self):
        return self.trn_loader

    @property
    def _tst_loader(self):
        return self.tst_loader

    @property
    def _val_loader(self):
        return self.val_loader

    @property
    def _trn_data(self) -> Data:
        return self.dh._train
    @_trn_data.setter
    def _trn_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _tst_data(self) -> Data:
        return self._dh._test
    @_tst_data.setter
    def _tst_data(self, value):
        raise ValueError("Why are u setting the data object once again?")   

    @property
    def _val_data(self) -> Data:
        return self._dh._val
    @_val_data.setter
    def _val_data(self, value):
        raise ValueError("Why are u setting the data object once again?")

    @property
    def _optimizer(self) -> optim.Optimizer:
        if  self.optimizer == None:
            raise ValueError("optimizer not yet set")
        return self.optimizer
    @_optimizer.setter
    def _optimizer(self, value):
        self.optimizer = value

    @property
    def _lr(self) -> nn.Module:
        return self.lr
    @_lr.setter
    def _lr(self, value):
        self.lr = value

    @property
    def _sw(self) -> SummaryWriter:
        return self.sw
    @_sw.setter
    def _sw(self, value):
        self.sw = value

    @property
    def _def_dir(self):
        return Path("./results/models/nn_theta")

    @property
    def _momentum(self):
        return self.momentum
    @_momentum.setter
    def _momentum(self, value):
        self.momentum = value

    @property
    def _lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        return self.lr_scheduler
    @_lr_scheduler.setter
    def _lr_scheduler(self, value):
        self.lr_scheduler = value

    @property
    def _xecri(self):
        return nn.CrossEntropyLoss()

    @property
    def _xecri_perex(self):
        return nn.CrossEntropyLoss(reduction="none")

    @property
    def _bcecri(self):
        return nn.BCELoss()
    
    @property
    def _bcecri_perex(self):
        return nn.BCELoss(reduction="none")
    
    @property
    def _msecri(self):
        return nn.MSELoss()

    @property
    def _msecri_perex(self):
        return nn.MSELoss(reduction="none")

# %% Abstract methods delegated to my children

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

    @abstractmethod
    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, data_subset=constants.FULL_DATA, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader, optional): [description]. Defaults to None.
            trn_wts ([type], optional): [description]. Defaults to None.
            epochs ([type], optional): [description]. Defaults to None.
            steps ([type], optional): [description]. Defaults to None.

        Raises:
            NotImplementedError: [description]
        """
        raise NotImplementedError()

    @abstractmethod
    def init_model(self):
        """Initialize trhe model.
        Foir resnet and the link, this should initialize with the pretrained weights
        For LR and stuff it should do random weights.


        Raises:
            NotImplementedError: [description]
            NotImplementedError: [description]

        Returns:
            [type]: [description]
        """
        raise NotImplementedError()


# %% some utilities

    def copy_model(self, *args, **kwargs):
        """Stores a copy of the model
        """
        assert self.__model_copy is None, "Why are you copying an alreasy copied model?"
        self.__model_copy = deepcopy(self._model.state_dict())
    
    def apply_copied_model(self, *args, **kwargs):
        """Loads the weights of deep copied model to the origibal model
        """
        assert self.__model_copy != None
        self.model.load_state_dict(self.__model_copy)

    def clear_copied_model(self, *args, **kwargs):
        """[summary]
        """
        assert self.__model_copy is not None, "Why are you clearing an already cleared copy?"
        self.__model_copy = None

    def predict_labels(self, loader) -> torch.Tensor:
        self._model.eval()
        pred_labels = []
        with torch.no_grad():
            for id, x, y in loader:
                x = x.to(cu.get_device())
                pred_labels.append(self._model.forward_labels(x).squeeze().cpu().detach())
        return torch.cat(pred_labels)

    def predict_proba(self, loader):
        self._model.eval()
        pred_probs = []
        with torch.no_grad():
           for id, x, y in loader:
               x = x.to(cu.get_device())
               pred_probs.append(self._model.forward_proba(x).cpu().detach())
        return torch.cat(pred_probs)

    def accuracy(self, loader=None, *args, **kwargs) -> float:
        self._model.eval()
        if loader is None:
            loader = self._tst_loader
        
        correct  = torch.scalar_tensor(0).to(cu.get_device())
        with torch.no_grad():
            for ids, x, y in loader:
                x, y = x.to(cu.get_device()), y.to(cu.get_device())
                correct += torch.sum(self._model.forward_labels(x) == y)
        return correct.item() / len(loader.dataset)


    def grp_accuracy(self, loader:data_utils.DataLoader=None, *args, **kwargs) -> dict:
        """Computes the accuracy on a per group basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if loader is None:
            loader = self._tst_loader
            y = self._dh._test._y
            Beta_test = self._dh._test._Beta
        else:
            y = self._dh._train._y
            Beta_test = self._dh._train._Beta

        res_dict = {}
        beta_dim = self._tst_data._Betadim
        res_dict = {}

        y_preds = self.predict_labels(loader)

        for beta_id in range(beta_dim):
            beta_values = set(Beta_test[:, beta_id])
            for beta_v in beta_values:
                beta_samples = torch.where(Beta_test[:, beta_id] == beta_v)[0]
                beta_val_acc = torch.sum(y[beta_samples] == y_preds[beta_samples]) / len(beta_samples)
                res_dict[f"id-{beta_id}:val-{beta_v}"] = beta_val_acc.item()
        return res_dict

    def beta_accuracy(self, loader:data_utils.DataLoader=None,*args, **kwargs) -> dict:
        """Computes the accuracy on a per beta basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        y_test = self._dh._test._y
        Beta_test = self._dh._test._Beta

        if loader is None:
            loader = self._tst_loader
        else:
            y_test = self._dh._train._y
            Beta_test = self._dh._train._Beta
        
        if "subset_idxs" in kwargs:
            subset_idxs = kwargs["subset_idxs"]
            y_test = self._dh._train._y[subset_idxs]
            Beta_test = self._dh._train._Beta[subset_idxs]
        
        res_dict = defaultdict(int)
        unq_beta = self._dh._test._unq_beta

        y_preds = self.predict_labels(loader)
        for beta in unq_beta:
            beta_ids = torch.where( torch.sum(Beta_test == beta, dim=1) == Beta_test.shape[1] )[0]
            acc = torch.sum(y_test[beta_ids] == y_preds[beta_ids]) / len(beta_ids)
            res_dict[str(beta.tolist())] = acc.item()
        return res_dict

    def get_conf_ybeta_prior(self, loader=None, *args, **kwargs) -> dict:
        """Computes the prior on P(Y=y | beta)
        """
        if loader is None:
            loader = self._tst_loader
            y_test = self._dh._test._y
            Beta_test = self._dh._test._Beta
        else:
            y_test = self._dh._train._y
            Beta_test = self._dh._train._Beta
        
        probs = self.predict_proba(loader)
        
        res_dict = {}
        unq_beta = self._dh._test._unq_beta

        for y in self.dh._test._classes:
            res_dict[y] = {}
            cls_idxs = torch.where(y_test == y)[0]
            for beta in unq_beta:
                beta_ids = torch.where( torch.sum(Beta_test[cls_idxs] == beta, dim=1) == Beta_test.shape[1] )[0]
                ybeta_ids = cls_idxs[beta_ids]
                conf = torch.mean(probs[ybeta_ids], dim=0)[y]
                res_dict[y][str(beta.tolist())] = conf
        return res_dict


    def get_conf_matrix(self, loader:data_utils.DataLoader=None,*args, **kwargs) -> dict:
        """Computes the accuracy on a per beta basis

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]
            Beta_test ([type]): [description]

        Returns:
            dict: [description]
        """
        if loader is None:
            loader = self._tst_loader
            y_test = self._dh._test._y
            Beta_test = self._dh._test._Beta
        else:
            y_test = self._dh._train._y
            Beta_test = self._dh._train._Beta

        unq_beta = self._dh._test._unq_beta
        unq_label = torch.unique(self._dh._test._y, dim=0)
        confusion_matrix = dict()

        y_preds = self.predict_labels(loader)
        print(f"y_preds - {len(y_preds)}, y_test - {len(y_test)}")

        for beta in unq_beta:
            confusion_matrix[beta] = []
            for label in unq_label:
                beta_ids = torch.where( (y_test==label) & (torch.sum(Beta_test == beta, dim=1) == Beta_test.shape[1]) )[0]
                acc = torch.sum(y_test[beta_ids] == y_preds[beta_ids]) / len(beta_ids)
                # res_dict[f"{label}_{beta}"] = acc.item()
                confusion_matrix[beta].append(round(acc.item(), 4))
        return confusion_matrix

    def get_batchloss_perex(self, X_test, y_test) -> torch.Tensor:
        """Gets the cross entropy loss per example

        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            np.array of losses
        """
        T = torch.Tensor
        if not isinstance(X_test, T):
            X_test, y_test = T(X_test), T(y_test)
        probs = self._model.forward_proba(X_test)
        return self._xecri_perex(
            T(probs).to(cu.get_device()), T(y_test).to(cu.get_device(), dtype=torch.int64)
        ).detach()

    def get_loaderlosses_perex(self, loader=None) -> torch.Tensor:
        self._model.eval()
        if loader is None:
            loader = self._dh._train_test.get_theta_loader(shuffle=False, batch_size=self._batch_size)
        losses = []
        with torch.no_grad():
            for dataids, x, y in loader:
                x, y = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64)
                y_preds = self._model.forward(x)
                losses.append(self._xecri_perex(y_preds, y).cpu().detach())
            return torch.cat(losses)


    def save_model_defname(self, suffix="", logger=None):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        if logger is not None:
            logger.info(f"Saved nnth model at {str(fname)}")
        torch.save(self._model.state_dict(), fname)

    def save_optim_defname(self, suffix="", logger=None):
        dir = self._def_dir
        fname = dir / f"{self._def_name}{suffix}-optim.pt"
        if logger is not None:
            logger.info(f"Saved nnth optim at {str(fname)}")
        torch.save(self._optimizer.state_dict(), fname)
    
    def load_model_defname(self, suffix="", logger=None):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded NN theta model from {str(fname)}")
        if logger is not None:
            logger.info(f"Loaded nnth model from {str(fname)}")
        self._model.load_state_dict(torch.load(fname, map_location=cu.get_device()))

    def load_optim_defname(self, suffix="", logger=None):
        dir = self._def_dir
        fname = dir / f"{self._def_name}{suffix}-optim.pt"
        print(f"Loaded NN theta Optimizer from {str(fname)}")
        if logger is not None:
            logger.info(f"Loaded nnth optim from {str(fname)}")
        self._optimizer.load_state_dict(torch.load(fname, map_location=cu.get_device()))
# %%
