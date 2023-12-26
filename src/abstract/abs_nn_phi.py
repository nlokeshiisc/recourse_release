from abc import ABC, abstractmethod, abstractproperty
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import utils.common_utils as cu
from torch.utils.tensorboard import SummaryWriter

from src.abstract.abs_data import DataHelper, Data
from src.abstract.abs_greedy_rec import RecourseHelper
import constants as constants
import utils.torch_data_utils as tdu


class NNPhiHelper(ABC):
    def __init__(self, phimodel:nn.Module, rechlpr:RecourseHelper, dh:DataHelper, *args, **kwargs) -> None:
        super().__init__()
        self.phimodel = phimodel
        self.rechlpr = rechlpr
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
        # Initializing train loader is fairly complicated
        # tst_loader and val_loader behave as always

        # For training, we need X, Beta, Sib_beta, Sij, losses of siblings (to implement many strategies)
        # To avoid complications, we only pass Beta ans tgt beta shall be computed runtime based on Sij
        trn_X = self._trn_data._X
        trn_Beta = self._trn_data._Beta
        trn_Sij = self._rechlpr._Sij
        trn_sibs = self._trn_data._Siblings

        trn_losses = self._rechlpr._nnth.get_loaderlosses_perex()
        sib_losses = torch.vstack([trn_losses[sib_ids] for sib_ids in trn_sibs])
        # now get the target beta. We only compute the sibnling betas here. Create the target as please later
        trn_sib_beta = ([
            trn_Beta[sib_ids] for sib_ids in trn_sibs
        ])
        trn_sib_beta = torch.vstack(trn_sib_beta).view(len(trn_sib_beta), *trn_sib_beta[0].shape)
        
        R_ids = self._rechlpr._R

        T = torch.Tensor
        self.phi_ds = tdu.CustomPhiGenDataset(R_ids=R_ids, X=trn_X[R_ids], Beta=trn_Beta[R_ids],
                    Sib_beta=trn_sib_beta[R_ids], Sij=trn_Sij[R_ids], Sib_losses=sib_losses[R_ids], transform=self._dh._train.transform)
        self.trn_gen_loader = data_utils.DataLoader(self.phi_ds, batch_size=self._batch_size, shuffle=False)

        self.tgt_betas = self.collect_tgt_betas(loader=self.trn_gen_loader)
        self.tgt_beta_unq, self.tgt_beta_cnts = torch.unique(self.tgt_betas, dim=0, return_counts=True)
        self.trn_loader = self._dh._train.get_phi_loader(R_ids=R_ids, tgt_Beta=self.tgt_betas, batch_size=self.batch_size,
                                                            shuffle=True)

        self.tst_loader = self._dh._test.get_xbeta_loader(shuffle=False, batch_size=128)
        self.val_loader = self._dh._val.get_xbeta_loader(shuffle=False, batch_size=128)

# %% Properties
    @property
    def _phimodel(self):
        return self.phimodel
    @_phimodel.setter
    def _phimodel(self, value):
        self.phimodel = value

    @property
    def _rechlpr(self) -> RecourseHelper:
        return self.rechlpr
    @_rechlpr.setter
    def _rechlpr(self, value):
        self.rechlpr = value

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
    def _optimizer(self):
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
    def _lr_scheduler(self):
        return self.lr_scheduler
    @_lr_scheduler.setter
    def _lr_scheduler(self, value):
        self.lr_scheduler = value

    @property
    def _sw(self) -> SummaryWriter:
        return self.sw
    @_sw.setter
    def _sw(self, value):
        self.sw = value

    @property
    def _tgt_betas(self):
        return self.tgt_betas
    @_tgt_betas.setter
    def _tgt_betas(self, value):
        self.tgt_betas = value

    @property
    def _def_dir(self):
        return Path("./results/models/nnphi")

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
    def _msecri(self):
        return nn.MSELoss()

    @property
    def _msecri_perex(self):
        return nn.MSELoss(reduction="none")

# %% Abstract methods to be delegated to my children
    @abstractmethod
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader, *args, **kwargs):
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
        """Initialize the model
        """
        raise NotImplementedError()
       


# %% Some utilities

    def trn_fit_accuracy(self):
        """Computes how good the nnphi model has fit to the trarget betas
        """
        loader = self._dh._train.get_phi_loader(R_ids=self.rechlpr._R, tgt_Beta=self.tgt_betas, batch_size=self.batch_size, shuffle=False)
        self._phimodel.eval()
       
        fit_dict = {}
        for b in self._dh._train._unq_beta:
            fit_dict[f"crct_{str(b.tolist())}"] = 0
            fit_dict[f"incrct_{str(b.tolist())}"] = 0
        acc = torch.scalar_tensor(0)
        with torch.no_grad():
            for rid, x, beta, tgt_beta in loader:
                x, beta, tgt_beta = x.to(cu.get_device()), beta.to(cu.get_device(), dtype=torch.int64), tgt_beta.to(cu.get_device(), dtype=torch.int64)
                pred_beta = self._phimodel.forward_labels(x, beta)
                correct = torch.sum(pred_beta == tgt_beta, dim=1) == tgt_beta.shape[1]
                correct = correct.cpu()
                for tb, c in zip(tgt_beta.cpu(), correct):
                    if c == True:
                        fit_dict[f"crct_{str(tb.tolist())}"] += 1
                    else:
                        fit_dict[f"incrct_{str(tb.tolist())}"] += 1
                acc += torch.sum(correct)
        return fit_dict, (acc/len(loader.dataset)).item()


    def predict_beta(self, loader=None):
        """This function predicts beta | x, beta. This is the output of g_phi. We clamp the predictions
        to \beta available in the training data only. Because it make no sense of the network
        to predict something it has never seen as it can not hallucinate goodness of betas outside
        training dataset.
        """

        if loader is None:
            loader = self._trn_loader
        
        self._phimodel.eval()
        pred_betas = []

        unq_beta = self.dh._test._unq_beta

        with torch.no_grad():
            for r_id, x, beta, _ in loader:
                x, beta = x.to(cu.get_device()), beta.to(cu.get_device())        
                pred_beta_probs = self._phimodel.forward_proba(x, beta)
                pred_beta_probs = [entry.cpu()  for entry in pred_beta_probs]

                pred_beta = []
                for idx in range(len(x)):
                    max_prob = 0
                    sel_beta = None
                    for beta_entry in unq_beta:
                        beta_etry_probs = torch.Tensor([pred_beta_probs[entry][idx][beta_entry[entry]] for entry in range(len(beta_entry))])            
                        beta_entry_prob = torch.prod(beta_etry_probs)
                        if beta_entry_prob > max_prob:
                            sel_beta = beta_entry
                            max_prob = beta_entry_prob
                    assert sel_beta is not None, "Why is sel beta none? We should have atleast one positive prob beta"
                    pred_beta.append(sel_beta)
                pred_betas.append(torch.stack(pred_beta))
        return torch.cat(pred_betas)


    def get_loss_perex(self, X_test, y_test):
        """Gets the cross entropy loss per example
        Args:
            X_test ([type]): [description]
            y_test ([type]): [description]

        Returns:
            torch.Tensor of losses
        """
        T = torch.Tensor
        X_test, y_test = T(X_test), T(y_test).to(cu.get_device(), dtype=torch.int64)
        probs = self.predict_proba(X_test)
        raise NotImplementedError("Decide the correct loss here") # TODO
        return self._xecri_perex(T(probs).to(cu.get_device()), y_test).cpu().numpy()

    
    def save_model_defname(self, suffix="", logger=None):
        dir = self._def_dir
        dir.mkdir(exist_ok=True, parents=True)
        fname = dir / f"{self._def_name}{suffix}.pt"
        
        if logger is not None:
            logger.info(f"Saving phi model at {fname}")
        torch.save(self._phimodel.state_dict(), fname)
    
    def load_model_defname(self, suffix="", logger=None):
        fname = self._def_dir / f"{self._def_name}{suffix}.pt"
        print(f"Loaded model from {str(fname)}")
        if logger is not None:
            logger.info(f"Loaded model from {str(fname)}")
        self._phimodel.load_state_dict(torch.load(fname, map_location=cu.get_device()))

    def collect_rec_betas(self):
        loader = self._trn_loader
        beta_preds = []
        for epoch_step, (rids, X, Beta, SibBeta, Sij, Siblosses) in enumerate(loader):
            X, Beta = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64)
            beta_pred = self._phimodel.forward_labels(X, Beta).cpu().detach()
            beta_preds.append(beta_pred)
        return torch.cat(beta_preds, dim=0)


    def collect_tgt_betas(self, loader=None):
        if loader is None:
            loader = self._trn_loader
        tgt_betas = []
        for epoch_step, (rids, X, Beta, SibBeta, Sij, Siblosses) in enumerate(loader):

            sel_min = lambda t, losses_i : torch.squeeze(t[torch.argmin(losses_i)])

            tgt_beta = torch.vstack([
                sel_min(SibBeta[entry], Siblosses[entry]) for entry in range(X.size(0)) 
            ])
            tgt_betas.append(tgt_beta)
        return torch.cat(tgt_betas, dim=0)