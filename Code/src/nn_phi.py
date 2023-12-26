from numpy import dtype
from src.abstract.abs_nn_phi import NNPhiHelper
from src.abstract.abs_greedy_rec import RecourseHelper
from src.abstract.abs_data import DataHelper
import torch.utils.data as data_utils
import warnings
from tqdm import tqdm
import utils.common_utils as cu
import utils.torch_data_utils as tdu
import utils.torch_utils as tu
import torch
from torch.optim import SGD, AdamW
from src.models import ResNETRecourse, FNNXBeta
import constants

class SynNNPhiMinHelper(NNPhiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, nn_arch:list, rechlpr:RecourseHelper, dh:DataHelper, *args, **kwargs) -> None:
        self.nn_arch = nn_arch
        # if u need dropouts, pass it in kwargs
        phimodel = FNNXBeta(in_dim=dh._train._Xdim+dh._train._Betadim, out_dims=dh._train._BetaShape, 
                            nn_arch=nn_arch, beta_dims=dh._train._BetaShape, prefix="SynNNPhi", *args, **kwargs)
        super(SynNNPhiMinHelper, self).__init__(phimodel, rechlpr, dh, args, **kwargs)

        self.init_model()
        

    def init_model(self):
        tu.init_weights(self._phimodel)
        self._phimodel.to(cu.get_device())
        self._optimizer = AdamW([
            {'params': self._phimodel.parameters()},
        ], lr=self._lr)
       
        
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader=None, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._phimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")

        if constants.SW in kwargs.keys():
            self._sw = kwargs[constants.SW]
        if constants.SCHEDULER in kwargs.keys():
            if constants.SCHEDULER_TYPE in kwargs.keys():
                raise NotImplementedError()
            self._lr_scheduler = tu.get_lr_scheduler(self._optimizer, scheduler_name="linear", n_rounds=epochs)

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (rids, X, Beta, tgt_beta) in enumerate(loader):
                global_step += 1

                X, Beta, tgt_beta = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), tgt_beta.to(cu.get_device(), dtype=torch.int64)

                self._optimizer.zero_grad()
                beta_preds = self._phimodel.forward(X, Beta)
                
                loss = torch.Tensor([self._xecri(beta_preds[idx], tgt_beta[:, idx]) for idx in range(self._dh._train._Betadim)])
                loss = torch.sum(loss)
                loss /= self._dh._train._Betadim
                loss.backward()

                self._optimizer.step()
                if self._sw is not None:
                    self._sw.add_scalar("phi_loss", loss.item(), global_step)

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)       

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            
            fit_dict, fit_acc = self.trn_fit_accuracy()
            fit_dict_str = cu.dict_print(fit_dict)
            if logger is not None:
                logger.info(f"Epoch: {epoch}, phi fit accuracy:")
                logger.info(fit_dict_str)
                logger.info(f"Fit Epoch Accuracy: {fit_acc}")
            if self._sw is not None:
                self._sw.add_scalar("phi_epoch_acc", fit_acc)
            print("")
        

    @property
    def _def_name(self):
        return f"syn_phi"

class ShapenetNNPhiMinHelper(NNPhiHelper):  
    """This is the default class for BBPhi.
    This des mean Recourse of Betas

    Args:
        NNPhiHelper ([type]): [description]
    """
    def __init__(self, nn_arch:list, rechlpr:RecourseHelper, dh:DataHelper, *args, **kwargs) -> None:
        self.nn_arch = nn_arch
        # if u need dropouts, pass it in kwargs
        phimodel = ResNETRecourse(out_dim=sum(dh._train._BetaShape), nn_arch=nn_arch, beta_dims=dh._train._BetaShape, prefix="shapenetnnphi", *args, **kwargs)
        super(ShapenetNNPhiMinHelper, self).__init__(phimodel, rechlpr, dh, args, **kwargs)

        self.init_model()
        

    def init_model(self):
        self._phimodel.init_model()
        self._phimodel.to(cu.get_device())
        self._optimizer = AdamW([
            {'params': self._phimodel.parameters()},
        ], lr=self._lr)
       
        
    def fit_rec_beta(self, epochs, loader:data_utils.DataLoader=None, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._phimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            warnings.warn("Are u sure u want to pass a custom loader?")

        if constants.SW in kwargs.keys():
            self._sw = kwargs[constants.SW]
        if constants.SCHEDULER in kwargs.keys():
            if constants.SCHEDULER_TYPE in kwargs.keys():
                raise NotImplementedError()
            self._lr_scheduler = tu.get_lr_scheduler(self._optimizer, scheduler_name="linear", n_rounds=epochs)

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (rids, X, Beta, tgt_beta) in enumerate(loader):
                global_step += 1

                X, Beta, tgt_beta = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), tgt_beta.to(cu.get_device(), dtype=torch.int64)

                self._optimizer.zero_grad()
                beta_preds = self._phimodel.forward(X, Beta)
                
                loss = self._xecri(beta_preds[0], tgt_beta[:, 0]) + self._xecri(beta_preds[1], tgt_beta[:, 1]) +  \
                        self._xecri(beta_preds[2], tgt_beta[:, 2])
                loss /= 3
                loss.backward()

                self._optimizer.step()
                if self._sw is not None:
                    self._sw.add_scalar("phi_loss", loss.item(), global_step)

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)       

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            
            fit_dict, fit_acc = self.trn_fit_accuracy()
            fit_dict_str = cu.dict_print(fit_dict)
            if logger is not None:
                logger.info(f"Epoch: {epoch}, phi fit accuracy:")
                logger.info(fit_dict_str)
                logger.info(f"Fit Epoch Accuracy: {fit_acc}")
            if self._sw is not None:
                self._sw.add_scalar("phi_epoch_acc", fit_acc)
            print("")
        

    @property
    def _def_name(self):
        return f"resnet_phi"