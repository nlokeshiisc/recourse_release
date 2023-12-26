import logging
import warnings

import constants
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import utils.common_utils as cu
import utils.torch_utils as tu
from tqdm import tqdm

from src.abstract.abs_data import DataHelper
from src.abstract.abs_greedy_rec import RecourseHelper
from src.abstract.abs_nn_psi import NNPsiHelper
from src.models import FNNPsi, ResNETPsi


class SynNNPsiHelper(NNPsiHelper):
    """This is the default class for BBPhi.
    This des mean Recourse of Betas
    Args:
        NNPhiHelper ([type]): [description]
    """

    def __init__(self, out_dim, nn_arch: list, rechlpr: RecourseHelper, psi_tgts, dh: DataHelper, *args, **kwargs) -> None:
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        # if u need dropouts, pass it in kwargs
        psimodel = FNNPsi(in_dim=dh._train._Xdim+dh._train._Betadim, out_dim=out_dim,
                            nn_arch=nn_arch, prefix="psi")
        super(SynNNPsiHelper, self).__init__(
            psimodel, rechlpr, psi_tgts, dh, args, kwargs)

        self.init_model()
       

    def init_model(self):
        tu.init_weights(self._psimodel)
        self._psimodel.to(cu.get_device())
        self._optimizer = optim.AdamW([
            {'params': self._psimodel.parameters()},
        ], lr=self._lr)

    def fit_rec_r(self, epochs, loader: data_utils.DataLoader = None, logger: logging.Logger = None, *args, **kwargs):
        """fits the data on the Dataloader that is passed
        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._psimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            if logger is not None:
                logger.warn("Are u sure u want to pass a custom loader?")
            warnings.warn("Are u sure u want to pass a custom loader?")

        if constants.SCHEDULER in kwargs.keys():
            self._lr_scheduler = tu.get_lr_scheduler(
                self._optimizer, scheduler_name="linear", n_rounds=epochs)

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (data_ids, X, Beta, R) in enumerate(loader):
                global_step += 1

                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), \
                            R.to(cu.get_device(), dtype=torch.int64)

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward(X, Beta)
                loss = self._xecri(rpreds, R)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})

                if self._sw is not None:
                    self._sw.add_scalar("Psi_Loss", loss.item(), global_step)

                tq.update(1)

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            r_acc, r_cls_acc = self.r_acc(class_wise=True)
            if self._sw is not None:
                self._sw.add_scalar("Psi_Epoch_Acc", r_acc, epoch)
            if logger is not None:
                logger.info(f"Epoch {epoch}, R pred accuracy: {r_acc}")
                logger.info(cu.dict_print(r_cls_acc))

    @property
    def _def_name(self):
        return f"synnnpsi"


class ShapenetNNPsiHelper(NNPsiHelper):
    """This is the default class for BBPhi.
    This des mean Recourse of Betas
    Args:
        NNPhiHelper ([type]): [description]
    """

    def __init__(self, out_dim, nn_arch: list, rechlpr: RecourseHelper, psi_tgts:str, dh: DataHelper, *args, **kwargs) -> None:
        self.out_dim = out_dim
        self.nn_arch = nn_arch

        # assert out_dim == 1, "R network shoud predict onl one value for (x, beta)"

        # if u need dropouts, pass it in kwargs
        psimodel = ResNETPsi(out_dim=out_dim, nn_arch=nn_arch,
                             beta_dims=dh._train._BetaShape, prefix="psi", *args, **kwargs)
        super().__init__(
            psimodel, rechlpr, psi_tgts, dh, *args, **kwargs)

        self.init_model()

    def init_model(self):
        self._psimodel.init_model()
        self._psimodel.to(cu.get_device())
        self._optimizer = optim.AdamW([
            {'params': self._psimodel.parameters()},
        ], lr=self._lr)

    def fit_rec_r(self, epochs, loader: data_utils.DataLoader = None, logger: logging.Logger = None, *args, **kwargs):
        """fits the data on the Dataloader that is passed
        Args:
            loader (data_utils.DataLoader): [description]
            epochs ([type], optional): [description]. Defaults to None. 
        """
        global_step = 0
        self._psimodel.train()
        if loader is None:
            loader = self._trn_loader
        else:
            if logger is not None:
                logger.warn("Are u sure u want to pass a custom loader?")
            warnings.warn("Are u sure u want to pass a custom loader?")

        if constants.SCHEDULER in kwargs.keys():
            self._lr_scheduler = tu.get_lr_scheduler(
                self._optimizer, scheduler_name="linear", n_rounds=epochs)

        for epoch in range(epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (data_ids, X, Beta, R) in enumerate(loader):
                global_step += 1

                X, Beta, R = X.to(cu.get_device()), Beta.to(cu.get_device(), dtype=torch.int64), \
                            R.to(cu.get_device(), dtype=torch.int64)

                self._optimizer.zero_grad()
                rpreds = self._psimodel.forward(X, Beta)
                loss = self._xecri(rpreds, R)

                loss.backward()
                self._optimizer.step()
                tq.set_postfix({"Loss": loss.item()})

                if self._sw is not None:
                    self._sw.add_scalar("Psi_Loss", loss.item(), global_step)

                tq.update(1)

            if self._lr_scheduler is not None:
                self._lr_scheduler.step()

            r_acc, r_cls_acc = self.r_acc(class_wise=True)
            if self._sw is not None:
                self._sw.add_scalar("Psi_Epoch_Acc", r_acc, epoch)
            if logger is not None:
                logger.info(f"Epoch {epoch}, R pred accuracy: {r_acc}")
                logger.info(cu.dict_print(r_cls_acc))

    @property
    def _def_name(self):
        return "resnetpsi"