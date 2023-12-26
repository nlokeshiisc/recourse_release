import pickle as pkl
import warnings
from abc import ABC, abstractmethod, abstractproperty
from copy import deepcopy
from pathlib import Path
import numpy as np
import torch
import utils.common_utils as cu
from torch.optim.sgd import SGD
from src.abstract.abs_data import DataHelper
from src.abstract.abs_nn_theta import NNthHelper
import constants as constants
from torch.utils.tensorboard.writer import SummaryWriter
import copy

class RecourseHelper(ABC):
    def __init__(self, nnth:NNthHelper, dh:DataHelper, budget, num_badex, R_per_iter, logger=None, *args, **kwargs) -> None:
        super().__init__()
        self.nnth = nnth
        self.dh = dh
        self.budget = budget
        self.num_r_per_iter = R_per_iter
        self.num_badex = num_badex
        self.sgd_optim = SGD([
            {"params": self._nnth._model.parameters()},
        ], lr=1e-3, momentum=0, nesterov=False)

        self.batch_size = 32
        self.lr = 1e-3

        self.R = torch.Tensor([])
        self.Sij = None
        self.trn_wts = torch.ones(self.dh._train._num_data)
        self.all_losses_cache = None

        self.__init_kwargs(kwargs)
        if logger is not None:
            logger.info("Ïnitializing Sij based on min pooling")
        self.set_Sij(margin=0)
        if logger is not None:
            logger.info("Ïnitializing Sij Complete")

    def __init_kwargs(self, kwargs):
        if constants.BATCH_SIZE in kwargs.keys():
            self.batch_size = kwargs[constants.BATCH_SIZE]
        if constants.LRN_RATTE in kwargs.keys():
            self.lr = kwargs[constants.LRN_RATTE]
        if constants.SW in kwargs.keys():
            self.sw = kwargs[constants.SW]
        if constants.NUMR_PERITER in kwargs.keys():
            self.num_r_per_iter = kwargs[constants.NUMR_PERITER]

    def init_trn_wts(self):
        for rid in self.R:
            self.trn_wts = self.simulate_addr(trn_wts=self.trn_wts, R=self.R, rid=rid)

# %% Some properties      
    @property
    def _nnth(self):
        return self.nnth
    @_nnth.setter
    def _nnth(self, value):
        self.nnth = value

    @property
    def _dh(self):
        return self.dh
    @_dh.setter
    def _dh(self, value):
        self.dh = value

    @property
    def _budget(self):
        return self.budget
    @_budget.setter
    def _budget(self, value):
        self.budget = value

    @property
    def _num_badex(self):
        return self.num_badex
    @_num_badex.setter
    def _num_badex(self, value):
        self.num_badex = value

    @property
    def _Sij(self):
        return self.Sij
    @_Sij.setter
    def _Sij(self, value):
        self.Sij = value

    @property
    def _R(self):
        return self.R
    @_R.setter
    def _R(self, value):
        if not isinstance(value, torch.Tensor):
            value = torch.Tensor(value)
        value = value.to(dtype=torch.int64)
        self.R = value

    @property
    def _sw(self) -> SummaryWriter:
        return self.sw
    @_sw.setter
    def _sw(self, value):
        self.sw = value

    @property
    def _trn_wts(self):
        return self.trn_wts
    @_trn_wts.setter
    def _trn_wts(self, value):
        self.trn_wts = value

    @property
    def _SGD_optim(self):
        return self.sgd_optim

    @property
    def _lr(self):
        return self.lr
    @property
    def _batch_size(self):
        return self.batch_size

    @property
    def _def_dir(self):
        return Path("./results/models/greedy_rec")
    

# %% some utility functions

    def get_trnloss_perex(self) -> torch.Tensor:
        """This is an utility to get the loss of all training examples in order.
        """
        if self.all_losses_cache is not None:
            return self.all_losses_cache
        loader = self._dh._train_test.get_theta_loader(shuffle=False, batch_size=128) # Have large batch size here for faster parallelism
        self.all_losses_cache = self._nnth.get_loaderlosses_perex(loader)
        return self.all_losses_cache

    def sample_bad_ex(self, R:list, num_ex=None):
        """Returns the examples that have the highest loss.

        Note: This only works with training data in dh. I dont know if we will ever need to sample bad guys in val/test data?

        Args:
            R ([type]): [description]
            model ([type], optional): [description]. Defaults to None.
            num_ex ([type], optional): [description]. Defaults to None.
        """ 
        if num_ex is None:
            num_ex = self._num_badex
        # We definitely cannot sample elements in R
        num_ex = min(num_ex, self._dh._train._num_data - len(R))

        losses = copy.deepcopy(self.get_trnloss_perex())
        if len(R) > 0:
            losses[R] = -1e10
        return torch.topk(losses, num_ex)[1]

    def minze_theta(self, loader, ex_trnwts, logger=None):
        """Minimizes Theta on the specified data loader
        This does a weighted ERM

        Args:
            rbg_loader ([type]): [description]
            ex_trnwts ([type]): [description]
        """
        self._nnth._model.train()
        for data_ids, x, y in loader:
            data_ids, x, y = data_ids.to(cu.get_device(), dtype=torch.int64), x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64)

            if torch.sum(ex_trnwts[data_ids]) == 0:
                if logger is not None:
                    logger.warning(f"extrn_wts for data_ids {data_ids.cpu()} came out to be all 0s")
                continue

            self._SGD_optim.zero_grad()

            pred_probs = self._nnth._model.forward(x)
            loss = self._nnth._xecri_perex(pred_probs, y)
            loss = torch.dot(loss, ex_trnwts[data_ids]) / (torch.sum(ex_trnwts[data_ids]))

            loss.backward()
            self._SGD_optim.step()        

    def set_Sij(self, margin, loader=None, pooling_type=constants.MIN_POOL):
        """This method find the set Sij for every ij present in the dataset.
        The recourse algorithm is heavily dependent on the margin. The margin is instrumental in putting only high likelihiood recourse elements into Sij
        You must pass loader with shuffle = False (or) You will hit an run time error.

        We compute losses using the currentr theta of NN_theta and appropriately determine Sij as thoise examples who have
        loss(it) < loss(ij) - margin

        We resorted to keeping margin=0 as we anyways plan to do min-pooling later.

        Args:
            margin ([type]): [description]
            loader ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        self.Sij = []
        losses = self.get_trnloss_perex()
        for data_id in self._dh._train._data_ids:
            sib_ids = self._dh._train._Siblings[data_id]

            arr = torch.zeros(len(sib_ids))

            if pooling_type == constants.MIN_POOL:
                arr[torch.argmin(losses[sib_ids])] = 1 # argmin may be rid itself and I dont care at this point
            elif pooling_type == constants.ALL_POOL:
                lessloss_ids = torch.where(losses[sib_ids] < losses[data_id])[0]
                if len(lessloss_ids) == 0:
                    lessloss_ids = torch.where(losses[sib_ids] <= losses[data_id])[0] # There is no recourse for this element
                arr[lessloss_ids] = 1
            
            self.Sij.append(arr)
    
        self.Sij = torch.vstack(self.Sij)
        return self.Sij

    def assess_R_candidates(self, trn_wts, R, bad_exs, rbg_loader):
        
        bad_losses = []

        for bad_ex in bad_exs:
            # save the model state
            self._nnth.copy_model()

            # Here we only do SGD Optimizer. Do not use momentum and stuff
            ex_trnwts = self.simulate_addr(trn_wts=trn_wts, R=R, rid=bad_ex)

            self.minze_theta(rbg_loader, torch.Tensor(ex_trnwts).to(cu.get_device()))
            bad_losses.append(np.mean(self.get_trnloss_perex(loader=rbg_loader)))

            self._nnth.apply_copied_model()
            self._nnth.clear_copied_model()
        
        return bad_losses

    def simulate_addr(self, trn_wts, R, rid):
        new_wts = deepcopy(trn_wts)
        sib_ids = self._dh._train._Siblings[rid]
        if sum(self.Sij[rid]) == 0.:
            # If there are no sij, recourse is hopeless
            warnings.warn("Why will Sij be empty when we attempt to add a bad example in R?")
            pass
        else:
            # This is for average pooling

            # numer = []
            # for idx, sid in enumerate(sib_ids):
            #     if  self.Sij[rid][idx] == 1 and sid not in R:
            #         numer.append(1)
            #     else:
            #         numer.append(0)

            # if sum(numer) != 0:
            #     new_wts[sib_ids] += np.array(numer)/sum(numer)
            #     new_wts[rid] -= 1
            # else:
            #     pass

            # This is for min pooling
            new_wts[sib_ids] += self.Sij[rid]
            new_wts[rid] -= 1 # Note if rid happens to be the least in the group, the net effect is no change so that gain=0
            if new_wts[rid] < 0:
                new_wts[rid] = 0
                warnings.warn("Can u justify in what cases, trn wts can become negative?")
        return new_wts
    
    def dump_recourse_state_defname(self, suffix="", model=False, logger=None):
        dir = self._def_dir
        dir.mkdir(parents=True, exist_ok=True)
        if model:
            self._nnth.save_model_defname(suffix=f"greedy-{suffix}")
        fname = f"{self._def_name}{suffix}-R-Sij-wts.pkl"
        with open(dir/fname, "wb") as file:
            pkl.dump({"R": self._R, "Sij": self._Sij, "trn_wts": self._trn_wts}, file)
        logger.info(f"Dumped greedy recourse results at {fname}")

    def load_recourse_state_defname(self, suffix="", model=False, logger=None):
        dir = self._def_dir
        if model:
            self._nnth.load_model_defname(suffix=f"greedy-{suffix}")
        fname = f"{self._def_name}{suffix}-R-Sij-wts.pkl"
        with open(dir/fname, "rb") as file:
            rsij_dict = pkl.load(file)
        self._R, self._Sij = rsij_dict["R"], rsij_dict["Sij"] # check if we need to load the trn wts also?
        print(f"Loaded Recourse from {dir}/{self._def_name}{suffix}")
        self.init_trn_wts()
        if logger is not None:
            logger.info(f"Loaded greedy recourse from {fname}")



# %% Abstract methods delegated to my children
    @abstractmethod
    def recourse_theta(self, *args, **kwargs):
        raise NotImplementedError()
    
    @abstractproperty
    def _def_name(self):
        raise NotImplementedError()

