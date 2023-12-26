import abc
from abc import ABC, abstractproperty
from multiprocessing.sharedctypes import Value
import constants as constants
import torch
import utils.torch_data_utils as tdu

class Data(ABC):
    """This is an abstract class for Dataset
    For us dataset is a tuple (x, y, z, beta, siblings, Z_id, ideal_betas)
    This is a facade for all the data related activities in code.
    """
    def __init__(self, X, y, Z, Beta, B_per_i, Siblings, Z_ids, ideal_betas, *args, **kwargs) -> None:
        super().__init__()
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y).to(dtype=torch.int64)
        self.Z = torch.Tensor(Z)

        assert len(Z) == len(y), "If the length of Z and y is not the same, we may have to repeat_interleave somewhere to make the rest of code easy to access."

        self.Beta = torch.Tensor(Beta).to(torch.int64)

        self.unq_beta = torch.unique(self.Beta, dim=0)
        self.idx_to_beta = []
        self.beta_to_idx = {}
        for idx, beta in enumerate(self.unq_beta):
            self.beta_to_idx[str(beta)] = idx
            self.idx_to_beta.append(beta)

        self.B_per_i = B_per_i
        
        if Siblings is not None:
            self.Siblings = torch.Tensor(Siblings).to(torch.int64)
        else:
            self.Siblings = None

        self.Z_ids = torch.Tensor(Z_ids).to(dtype=torch.int64)

        self.num_classes = len(set(y))
        self.classes = set(y)

        if ideal_betas is not None:
            self.ideal_betas = torch.Tensor(ideal_betas).to(torch.int64)
        else:
            self.ideal_betas = None
        
        self.data_ids = torch.arange(len(X)).to(dtype=torch.int64)
        self.num_Z = len(set(self.Z_ids.tolist()))

        self.transform = None
        self.__init_kwargs(kwargs)

    def __init_kwargs(self, kwargs):
         if constants.TRANSFORM in kwargs.keys():
            self.transform = kwargs[constants.TRANSFORM]

    @property
    def _data_ids(self) -> torch.Tensor:
        return self.data_ids

    @property
    def _Z_ids(self) -> torch.Tensor:
        return self.Z_ids

    @property
    def _X(self) -> torch.Tensor:
        return self.X
    @_X.setter
    def _X(self, value):
        self.X = value
    
    @property
    def _y(self) -> torch.Tensor:
       return self.y
    @_y.setter
    def _y(self, value):
        self.y = value
    
    @property
    def _Z(self) -> torch.Tensor:
        return self.Z
    @_Z.setter
    def _Z(self, value):
        self.Z = value
    
    @property
    def _Beta(self) -> torch.Tensor:
        return self.Beta
    @_Beta.setter
    def _Beta(self, value):
        self.Beta = value

    @property
    def _num_beta(self):
        return self.Beta.shape[1]

    @property
    def _unq_beta(self):
        return self.unq_beta

    @property
    def _B_per_i(self):
        if self.B_per_i is None:
            raise ValueError("You are perhaps trying to get this variable for test data which is illegal")
        return self.B_per_i

    @property
    def _Siblings(self) -> torch.Tensor:
        if self.Siblings is None:
            raise ValueError("Why are u calling siblings on the test/val data?")
        return self.Siblings
    
    @property
    def _ideal_betas(self) -> torch.Tensor:
        return self.ideal_betas
    @_ideal_betas.setter
    def _ideal_betas(self, value):
        raise ValueError("Pass it once in constructor. Why are u settig it again?")

# %% some useful functions
    
    def get_instances(self, data_ids:torch.Tensor):
        """Returns ij data ids in order:
            x, y, z, Beta

        Args:
            data_ids (Tensor): [description]

        Returns:
            X, y, Z, Beta in order
        """
        if not isinstance(data_ids, torch.Tensor):
            data_ids = torch.Tensor(data_ids)
        return self._X[data_ids], self._y[data_ids], self._Beta[data_ids]
    
    def get_Zgrp_instances(self, zids:torch.Tensor):
        """Finds z id of all the ij instances given in the data_ids
        Then returns all the items in the Z group in order
            x, y, z, Beta

        Args:
            data_ids (np.array): [description]

        Returns:
            X, y, Z, Beta
        """
        if isinstance(zids, int):
            zids = torch.Tensor([zids]).to(torch.int64)
        if not isinstance(zids, torch.Tensor):
            zids = torch.Tensor(zids).to(torch.int64)
        zids = [torch.where(self._Z_ids == entry)[0] for entry in zids]
        zids = torch.stack(zids).flatten()
        return zids, self._X[zids], self._y[zids], self._Beta[zids]
    
    def get_siblings_intances(self, data_ids):
        if not isinstance(data_ids, torch.Tensor):
            data_ids = torch.Tensor(data_ids)
        return self._Siblings[data_ids]

    def get_theta_loader(self, batch_size, shuffle, **kwargs):
        theta_ds = tdu.CustomThetaDataset(self.data_ids, self.X, self.y, self.transform)
        th_loader = tdu.init_loader(theta_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return th_loader

    def get_xbeta_loader(self, batch_size, shuffle, **kwargs):
        xbeta_ds = tdu.CustomXBetaDataset(self.data_ids, self.X, self.Beta, self.y, self.transform)
        xb_loader = tdu.init_loader(xbeta_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return xb_loader

    def get_greedy_loader(self, batch_size, shuffle, **kwargs):
        greedy_ds = tdu.CustomGreedyDataset(self.data_ids, self.X, self.y, self.transform)
        greedy_loader = tdu.init_loader(greedy_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return greedy_loader

    def get_phi_loader(self, R_ids:torch.Tensor, tgt_Beta:torch.Tensor, batch_size, shuffle, **kwargs):
        if not isinstance(R_ids, torch.Tensor):
            R_ids = torch.Tensor(R_ids)
        phi_ds = tdu.CustomPhiDataset(R_ids=R_ids, X=self.X[R_ids], Beta=self.Beta[R_ids], tgt_Beta=tgt_Beta, transform=self.transform)
        phi_loader = tdu.init_loader(phi_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return phi_loader

    def get_psi_loader(self, R_tgts, batch_size, shuffle, **kwargs):
        if not isinstance(R_tgts, torch.Tensor):
            R_tgts = torch.Tensor(R_tgts)
        psi_ds = tdu.CustomPsiDataset(data_ids=self.data_ids, X=self.X, Beta=self.Beta, R_tgts=R_tgts, transform=self.transform)
        psi_loader = tdu.init_loader(psi_ds, batch_size=batch_size, shuffle=shuffle, **kwargs)
        return psi_loader

    def get_XBetagrp_loader(self, batch_size, shuffle, **kwargs):
        grp_xbeta_ds = tdu.init_grpXBeta_dataset(data_ids=self.data_ids, X=self.X, y=self.y,
                                                Beta=self.Beta, B_per_i=self.B_per_i, transform=self.transform)
        grp_xbeta_loader = tdu.init_loader(grp_xbeta_ds, shuffle=shuffle, batch_size=int(batch_size/self.B_per_i))
        return grp_xbeta_loader

    @property
    def _num_data(self):
        return len(self.data_ids)

    @property
    def _num_Z(self):
        return self.num_Z

    @property
    def _Xdim(self):
        return self._X.shape[1]
    
    @property
    def _num_classes(self):
        return self.num_classes
    @property
    def _classes(self):
        return self.classes

    @abc.abstractmethod
    def apply_recourse(self, data_id, betas):
        raise NotImplementedError()

    @abstractproperty
    def _BetaShape(self):
        raise NotImplementedError()

    @abstractproperty
    def _Betadim(self):
        return self._Beta.shape[1]

class DataHelper(ABC):
    def __init__(self, train, test, val, train_test=None) -> None:
        super().__init__()
        self.train = train
        self.test = test
        self.val = val
        self.train_test = train_test
        self.trn_subset = constants.FULL_DATA
    
    @property
    def _train(self) -> Data:
        return self.train
    @_train.setter
    def _train(self, value):
        self.train = value

    @property
    def _train_test(self) -> Data:
        return self.train_test
    @_train_test.setter
    def _train_test(self, value):
        self.train_test = value
    
    @property
    def _test(self) -> Data:
        return self.test
    @_test.setter
    def _test(self, value):
        self.test = value

    @property
    def _val(self) -> Data:
        return self.val
    @_val.setter
    def _val(self, value):
        self.val = value
