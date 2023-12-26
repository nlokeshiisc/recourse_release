from abc import ABC, abstractmethod, abstractproperty
import constants as constants
from pathlib import Path

from torch.utils.tensorboard.writer import SummaryWriter
from src.abstract.abs_data import Data, DataHelper
from src.abstract.abs_nn_phi import NNPhiHelper
from src.abstract.abs_nn_theta import NNthHelper
from src.abstract.abs_greedy_rec import RecourseHelper

import torch
import utils.common_utils as cu
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from copy import deepcopy

class MethodsHelper(ABC):
    """
    This is a base class for all models that provide recourse to our Synthetic dataset.
    Donot instantiate objects out of tjis class.
    """
    def __init__(self, dh:DataHelper, nnth:NNthHelper, nnphi:NNPhiHelper, 
                        rechlpr:RecourseHelper, *args, **kwargs) -> None:
        
        self.dh = dh
        self.nnth = nnth
        self.nnphi = nnphi
        self.rech = rechlpr

        self.rech.set_Sij(margin=0, pooling_type=constants.ALL_POOL)

        self.optim = None
        self.phi_optim = None
        self.th_optim = None
        self.psi_optim = None

        self.lr = 1e-4
        self.sw = None
        self.batch_size = 16 * self.dh._train._B_per_i
        self.pretrn_models = {
            "th": False,
            "phi": False,
            "psi": False
        }
        self.__init_kwargs(kwargs)
        self.__init_loader()

        self._thmodel.to(cu.get_device())
        self._phimodel.to(cu.get_device())
        self._psimodel.to(cu.get_device())

        self.prior = None
        self.compute_prior()

        self.raw_losses = None
        self.prior_losses = None
        self.cls_prior_losses = None
        self.pos_samples = None
        self.neg_samples = None
        self.compute_prior_losses()


# %% inits
    def __init_kwargs(self, kwargs:dict):
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.PRETRN_THPHI in kwargs.keys():
            assert isinstance(kwargs[constants.PRETRN_THPHI], dict), "Please pass a dictionary to me"
            self.pretrn_models = kwargs[constants.PRETRN_THPHI]
            if not self.pretrn_models[constants.THETA]:
                self.nnth.init_model()
            if not self.pretrn_models[constants.PHI]:
                self.nnphi.init_model()

    def __init_loader(self):
        """We uill use the dh loaders only here
        """
        self.trn_loader = self._dh._train_test.get_xbeta_loader(batch_size=self.batch_size, shuffle=True)
        self.trn_grp_loader = self.dh._train_test.get_XBetagrp_loader(batch_size=self.batch_size, shuffle=True)
        self.val_loader = self._dh._val.get_xbeta_loader(batch_size=self.batch_size, shuffle=False)
        self.tst_loader = self._dh._test.get_xbeta_loader(batch_size=self.batch_size, shuffle=False)


# %% Properties
    @property
    def _thmodel(self):
        return self.nnth._model
    @_thmodel.setter
    def _thmodel(self, state_dict):
        self.nnth._model.load_state_dict(state_dict) 

    @property
    def _phimodel(self):
        return self.nnphi._phimodel
    @_phimodel.setter
    def _phimodel(self, state_dict):
        self.nnphi._phimodel.load_state_dict(state_dict) 

    @property
    def _psimodel(self):
        return self.nnpsi._psimodel
    @_psimodel.setter
    def _psimodel(self, state_dict):
        self.nnpsi._psimodel.load_state_dict(state_dict) 

    @property
    def _nnth(self) -> NNthHelper:
        return self.nnth
    
    @property
    def _nnphi(self) -> NNPhiHelper:
        return self.nnphi

    @property
    def _R(self):
        return self.rech._R

    @property
    def _Sij(self):
        return self.rech._Sij

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _batch_size(self):
        return self.batch_size
    @_batch_size.setter
    def _batch_size(self, value):
        self.batch_size = value

    @property
    def _trn_loader(self):
        return self.trn_loader
    
    @property
    def _trngrp_loader(self):
        return self.trn_grp_loader

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
    def _Xdim(self):
        return self._trn_data._Xdim
    @property
    def _Betadim(self):
        return self._trn_data._Betadim
    @property
    def _num_classes(self):
        return self._trn_data._num_classes

    @property
    def _optim(self) -> optim.Optimizer:
        return self.optim
    @_optim.setter
    def _optim(self, value):
        self.optim = value

    @property
    def _thoptim(self) -> optim.Optimizer:
        return self.th_optim
    @_thoptim.setter
    def _thoptim(self, value):
        self.th_optim = value

    @property
    def _phioptim(self) -> optim.Optimizer:
        return self.phi_optim
    @_phioptim.setter
    def _phioptim(self, value):
        self.phi_optim = value

    @property
    def _psioptim(self) -> optim.Optimizer:
        return self.psi_optim
    @_psioptim.setter
    def _psioptim(self, value):
        self.psi_optim = value

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
    def _xecri(self):
        return nn.CrossEntropyLoss()

    @property
    def _rankingloss(self):
        return nn.MarginRankingLoss(margin=0.0)

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

    @property
    def _KLCriterion_rednone(self):
        return nn.KLDivLoss(reduction="none")


# %% abstract methods

    @abstractmethod
    def fit_epoch(self, epoch, loader=None, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractproperty
    def _def_name(self):
        pretrn_sfx = "pretrn="
        if self.pretrn_models["th"]:
            pretrn_sfx += "Th"
        if self.pretrn_models["phi"]:
            pretrn_sfx += "Phi"
        if self.pretrn_models["psi"]:
            pretrn_sfx += "Psi"
        return pretrn_sfx + "-"
    
    @abstractproperty
    def _def_dir(self):
        return Path("./results/ourm/models")

# %% some utilities
    def compute_prior(self):
        """Computes the counterfactual function $f^{\text{CF}}$
        """
        loader = self._dh._train_test.get_theta_loader(batch_size=128, shuffle=False)
        y = self._dh._train_test._y
        self.prior = {}
        self.prior["beta_acc"] = {}
        beta_acc = self._nnth.get_conf_ybeta_prior(loader=loader)
        for cls_idx in range(self._dh._train._num_classes):
            beta_acc_cls = beta_acc[cls_idx]
            Keymax = max(beta_acc_cls, key= lambda entry: beta_acc_cls[entry])
            self.prior[cls_idx] = torch.Tensor(eval(Keymax)).view(1,-1).to(device=cu.get_device(), dtype=torch.int64)
            self.prior["beta_acc"][cls_idx] = beta_acc_cls

    def compute_prior_losses(self):
        """Coputes a prior on losses. The losses are computed on X|y,beta on te train_test dataset.
        Train test is nothing but the train data without data augmentations.
        This function also pre computes the positive and negative samples to be used by Ranking Prior in Our Method later.
        """
        self.raw_losses = self._nnth.get_loaderlosses_perex(loader=self.dh._train_test.get_theta_loader(batch_size=self.batch_size, shuffle=False))
        Beta = self._dh._train_test._Beta
        y = self.dh._train_test._y
        self.cls_prior_losses = []

        beta_to_idx = self._dh._test.beta_to_idx

        for cls_idx in self.dh._train_test._classes:
            cls_beta_losses = torch.zeros(len(self._dh._test._unq_beta))
            cls_idxs = torch.where(y == cls_idx)[0]
            for beta in self.dh._train_test.unq_beta:
                beta_cls_idxs = torch.where( torch.sum(Beta[cls_idxs] == beta, dim=1) == Beta.shape[1] )[0]
                beta_cls_idxs = cls_idxs[beta_cls_idxs]
                cls_beta_losses[beta_to_idx[str(beta)]] = torch.mean(self.raw_losses[beta_cls_idxs])
            self.cls_prior_losses.append(cls_beta_losses)
        
        self.cls_prior_losses = torch.stack(self.cls_prior_losses)

        B_per_i = self.dh._train_test._B_per_i
        get_grp = lambda idx: torch.arange(int(idx/B_per_i)*B_per_i, ((int(idx/B_per_i)+1)*B_per_i)).to(dtype=torch.int64)

        self.prior_losses = []
        self.pos_samples = []
        self.neg_samples = []

        for idx, (l, b, y_idx) in enumerate(zip(self.raw_losses, Beta, y)):
            idx_losses = deepcopy(self.cls_prior_losses[y_idx])
            grp_idxs = get_grp(idx)
            for i in grp_idxs:
                idx_losses[beta_to_idx[str(Beta[i])]] = self.raw_losses[i]
            self.prior_losses.append(idx_losses)

            pvec = torch.zeros(len(self._dh._test._unq_beta))
            nvec = torch.zeros(len(self._dh._test._unq_beta))

            p = torch.where(idx_losses < self.raw_losses[idx])[0]
            if len(p) == 0:
                p = torch.Tensor([beta_to_idx[str(b)]]).to(dtype=torch.int64)
            pvec[p] = torch.softmax(-idx_losses[p], dim=0)
            self.pos_samples.append(pvec)

            n = torch.where(idx_losses > self.raw_losses[idx])[0]
            if len(n) == 0:
                n = torch.Tensor([beta_to_idx[str(b)]]).to(dtype=torch.int64)
            nvec[n] = torch.softmax(idx_losses[n], dim=0)
            self.neg_samples.append(nvec)


    def accuracy(self, loader=None, *args, **kwargs) -> float:
        return self._nnth.accuracy(loader=loader, *args, **kwargs)

    def grp_accuracy(self, loader, *args, **kwargs) -> dict:
        return self._nnth.grp_accuracy(loader, *args, **kwargs)
    
    def predict_labels(self, loader, *args, **kwargs) -> torch.Tensor:
        return self._nnth.predict_labels(loader)

    def predict_proba(self, loader, *args, **kwargs) -> torch.Tensor:
        return self._nnth.predict_proba(loader)

    def predict_betas(self, loader, *args, **kwargs) -> torch.Tensor:
        return self._nnphi.predict_beta(loader=loader)

    def predict_r(self, loader, *args, **kwargs) -> torch.Tensor:
        return self._nnpsi.predict_r(loader=loader)
    
    def save_model_defname(self, suffix = "", logger=None):
        state_dict = {
            "phi": self._phimodel.state_dict(),
            "psi": self._psimodel.state_dict(),
            "th": self._thmodel.state_dict()
        }
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = self._def_name + suffix + ".pt"
        if logger is not None:
            logger.info(f"Dumping our method at: {fname}")
        torch.save(state_dict, self._def_dir / fname)
    
    def load_model_defname(self, suffix="", logger=None):
        fname = self._def_name + suffix + ".pt"
        if logger is not None:
            logger.info("loading our method from: ", fname)
        state_dict = torch.load(self._def_dir / (self._def_name + suffix + '.pt'), map_location=cu.get_device())

        self._thmodel.load_state_dict(state_dict["th"])
        self._phimodel.load_state_dict(state_dict["phi"])
        # self._psimodel.load_state_dict(state_dict["psi"])
        print(f"Models loaded from {str(self._def_dir / (self._def_name + suffix + '.pt'))}")


