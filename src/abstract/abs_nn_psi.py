from abc import ABC, abstractmethod, abstractproperty
from cProfile import label
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
import utils.common_utils as cu
import utils.torch_data_utils as tdu
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter
from src.abstract.abs_data import Data, DataHelper
from src.abstract.abs_greedy_rec import RecourseHelper
import constants as constants


class NNPsiHelper(ABC):
    def __init__(self, psimodel:nn.Module, rechlpr:RecourseHelper, psi_tgts:str, dh:DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.psimodel = psimodel
        self.rechlpr = rechlpr
        self.psi_tgts = psi_tgts
        self.dh = dh
        
        self.lr = 1e-3
        self.sw = None
        self.batch_size = 16
        self.lr_scheduler = None

        self.__init_kwargs(kwargs)
        self.__init_loaders()
    
    def __init_kwargs(self, kwargs:dict):
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]

    def __init_loaders(self):
        # This is very simple. We just need x, y, R in each batch. Thats all
        # here care needs to be taken so that each batch has 50% +ve samples

        # Make R = 1 depending on the R taret strategy
        self.trn_tgts = torch.zeros(self._dh._train._num_data, dtype=int)

        if self.psi_tgts == constants.ONLYR:
            self.trn_tgts[self._R] = 1
        elif self.psi_tgts == constants.R_WRONG:
            self.trn_tgts[self._R] = 1 # This is for R
            pred_labels = self.rechlpr._nnth.predict_labels(loader=self.dh._train_test.get_theta_loader(batch_size=64, shuffle=False))
            labels = self.dh._train._y
            inc_idxs = torch.where(pred_labels != labels)[0]
            self.trn_tgts[inc_idxs] = 1 # This is for R UNION misclassified examples
            print(f"For strategy {constants.R_WRONG}, number of 1's in targets is: {sum(self.trn_tgts)}")
        elif self.psi_tgts == constants.ONLY_WRONG:
            pred_labels = self.rechlpr._nnth.predict_labels(loader=self.dh._train_test.get_theta_loader(batch_size=64, shuffle=False))
            labels = self.dh._train._y
            inc_idxs = torch.where(pred_labels != labels)[0]
            self.trn_tgts[inc_idxs] = 1 # This is for R UNION misclassified examples
            print(f"For strategy {constants.ONLY_WRONG}, number of 1's in targets is: {sum(self.trn_tgts)}")
        else:
            raise ValueError("Pass the right target strategy for psi")

        trn_tgts_oh = torch.vstack([1-self.trn_tgts, self.trn_tgts]).T.numpy() # This is needed only for creatig the batch sampler
        batch_sampler = tdu.MultilabelBalancedRandomSampler(trn_tgts_oh) 

        loader_args = {
            constants.SAMPLER: batch_sampler
        }
        self.trn_loader = self._dh._train.get_psi_loader(R_tgts=self.trn_tgts, batch_size=self.batch_size, shuffle=True, **loader_args)
        self.trn_xbeta_loader = self._dh._train.get_xbeta_loader(batch_size=128, shuffle=False)

        self.tst_loader = self._dh._test.get_xbeta_loader(batch_size=128, shuffle=False)
        self.val_loader = self._dh._val.get_xbeta_loader(batch_size=128, shuffle=False)

# %% Properties
    @property
    def _psimodel(self) -> nn.Module:
        return self.psimodel
    @_psimodel.setter
    def _psimodel(self, value):
        self.psimodel = value

    @property
    def _rechlpr(self) -> RecourseHelper:
        return self.rechlpr
    @_rechlpr.setter
    def _rechlpr(self, value):
        self.rechlpr = value

    @property
    def _dh(self) -> DataHelper:
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _R(self):
        return self.rechlpr._R.to(dtype=torch.int64)
    
    @property
    def _batch_size(self):
        return self.batch_size

    @property
    def _trn_loader(self) -> data_utils.DataLoader:
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
    def _lr(self) -> float:
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
    def _lr_scheduler(self):
        return self.lr_scheduler
    @_lr_scheduler.setter
    def _lr_scheduler(self, value):
        self.lr_scheduler = value

    @property
    def _def_dir(self):
        return Path("./image_recourse/results/models/nnpsi")

    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

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

# %% Abstract methods to be delegated to my children
    @abstractmethod
    def fit_rec_r(self, epochs, loader:data_utils.DataLoader, *args, **kwargs):
        """Fits the Recourse beta that was learned during recourse R, Sij generation
        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type]): [description]
        Raises:
        Returns:
            [type]: [description]
        """
        raise NotImplementedError()
    
    @abstractmethod
    def init_model(self):
        """Initiaslize the model
        """
        raise NotImplementedError()
       
# %% Some utilities
    def r_acc(self, loader=None, class_wise=False, *args, **kwargs) -> float:
        """Returns the Accuracy of predictions
        Also return the class wise performance as a dictionary
        Works only for the train data
        """
        pred_r = self.predict_r(self.trn_xbeta_loader)
        correct = pred_r == self.trn_tgts

        if class_wise == True:
            ones = torch.where(self.trn_tgts == 1)[0]
            zeros = torch.where(self.trn_tgts == 0)[0]
            res_dict = {
                "ones_acc": (torch.sum(correct[ones])/len(ones)).item(),
                "zero_acc": (torch.sum(correct[zeros])/len(zeros)).item()
            }
            return (torch.sum(correct) / len(correct)).item(), res_dict
        else:
            return (torch.sum(correct) / len(correct)).item()

        
    def predict_r(self, XBetaloader:data_utils.DataLoader=None, **kwargs):
        assert isinstance(XBetaloader.dataset, tdu.CustomXBetaDataset) == True, "Pass only XBeta loader to me"

        pred_r = []
        self._psimodel.eval()
        with torch.no_grad():
            for data_id, x, beta, y in XBetaloader:
                x, beta = x.to(cu.get_device()), beta.to(cu.get_device(), dtype=torch.int64)
                r = self._psimodel.forward_r(x, beta).cpu().detach()
                r = r > 0.5
                pred_r.append(r)
        return torch.cat(pred_r) * 1

    def predict_proba(self, XBetaloader:data_utils.DataLoader=None, **kwargs):
        assert isinstance(XBetaloader.dataset, tdu.CustomXBetaDataset) == True, "Pass only XBeta loader to me"
        pred_r = []
        self._psimodel.eval()
        with torch.no_grad():
            for data_id, x, beta, y in XBetaloader:
                x, beta = x.to(cu.get_device()), beta.to(cu.get_device(), dtype=torch.int64)
                r = self._psimodel.forward_proba(x, beta).cpu().detach()
                pred_r.append(r)
        return torch.cat(pred_r)

    def save_model_defname(self, suffix="", logger=None):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        if logger is not None:
            logger.info("NNPhi model saved at: ", fname)
        torch.save(self._psimodel.state_dict(), fname)
    
    def load_model_defname(self, suffix="", logger=None):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded model from {str(fname)}")
        if logger is not None:
            logger.info("Loaded nnphi model from: ", fname)
        self._psimodel.load_state_dict(torch.load(fname, map_location=cu.get_device()))