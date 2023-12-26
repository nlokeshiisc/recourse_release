import torch
from src.abstract import abs_nn_theta
from src.abstract import abs_data
from src.models import LRModel, ResNET
from utils import torch_utils as tu
from utils import common_utils as cu
from utils import torch_data_utils as tdu
import torch.optim as optim
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm
import constants as constants

class LRNNthHepler(abs_nn_theta.NNthHelper):  
    def __init__(self, dh:abs_data.DataHelper, *args, **kwargs) -> None:
        self.in_dim = dh._train._Xdim
        self.n_classes = dh._train._num_classes
        model = LRModel(in_dim=self.in_dim, n_classes=self.n_classes, *args, **kwargs)
        super(LRNNthHepler, self).__init__(model, dh, *args, **kwargs)

        self.init_model()
    
    def init_model(self):
        tu.init_weights(self._model)
        self._model.to(cu.get_device())
        self._optimizer = optim.AdamW([
            {'params': self._model.parameters()},
        ], lr=self._lr)

    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, data_subset=constants.FULL_DATA, logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            trn_wts ([type], optional): [description]. Defaults to None. weights to be associated with each traning sample.
            epochs ([type], optional): [description]. Defaults to None. 
            steps ([type], optional): [description]. Defaults to None.
        """
        assert not(epochs is not None and steps is not None), "We will run either the specified SGD steps or specified epochs over data. We cannot run both"
        assert not(epochs is None and steps is None), "We need atleast one of steps or epochs to be specified"

        global_step = 0
        total_sgd_steps = np.inf
        total_epochs = 10
        if steps is not None:
            total_sgd_steps = steps
        if epochs is not None:
            total_epochs = epochs

        self._model.train()

        if loader is None:
            loader = self._trn_loader
        
        if data_subset == constants.IDEAL_SUB:
            ideal_idxs = torch.where(self._trn_data._ideal_betas == 1)[0]
            loader = tdu.get_loader_subset(loader, subset_idxs=ideal_idxs, batch_size=self._batch_size, shuffle=True)
            if logger is not None:
                logger.info(f"Fitting nntheta on the ideal subset. After subsetting the Training data size is {len(loader.dataset)}")

        if trn_wts is None:
            trn_wts = torch.ones(len(loader.dataset)).to(cu.get_device())
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"
    
        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, x, y) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    if logger is not None:
                        logger.info("Aborting fit as we matched the number of SGD steps required!")
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                cls_out = self._model.forward(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"Loss": loss.item()})
                tq.update(1)
            
            epoch_acc = self.accuracy()
            print(f"Epoch: {epoch} accuracy: {epoch_acc}")
            if self._sw is not None:
                self._sw.add_scalar("nnth_Epoch_Acc", epoch_acc, epoch)
            if logger is not None:
                logger.info(f"Epoch: {epoch} accuracy: {epoch_acc}")

    def recourse_accuracy(self, rec_betas, rec_idxs=None):
        if rec_idxs is None:
            rec_idxs = torch.arange(self._dh._test._num_data)
        loader = self._tst_loader
        accs = 0
        self._model.eval()
        for dataid, x, y in loader:
            x_rec = x.clone()
            for idx, did in enumerate(dataid):
                if did in rec_idxs:
                    x_rec[idx] = torch.multiply(self._dh._test._Z[did], rec_betas[idx])
            pred_y = self._model.forward_labels(x_rec)
            accs += torch.sum(pred_y == y)
            mispred = torch.where(pred_y != y)[0]
            # print(rec_betas[mispred])

        return accs.cpu().item()/self._dh._test._num_data

    @property
    def _def_name(self):
        return "logreg"

class ResNETNNthHepler(abs_nn_theta.NNthHelper):  
    def __init__(self, dh:abs_data.DataHelper, *args, **kwargs) -> None:
        self.n_classes = dh._train._num_classes
        model = ResNET(out_dim=self.n_classes, *args, **kwargs)
        super(ResNETNNthHepler, self).__init__(model, dh, *args, **kwargs)

        self.init_model()
        # For Resnet, we should never initialize weights
       

    def init_model(self):
        self._model.init_model()
        self._model.to(cu.get_device())
        self._optimizer = optim.SGD(
            self._model.parameters(), lr=self._lr, momentum=self._momentum
            )

    def fit_data(self, loader:data_utils.DataLoader=None, trn_wts=None,
                        epochs=None, steps=None, data_subset=constants.FULL_DATA ,logger=None, *args, **kwargs):
        """fits the data on the Dataloader that is passed

        Args:
            loader (data_utils.DataLoader): [description]
            trn_wts ([type], optional): [description]. Defaults to None. weights to be associated with each traning sample.
            epochs ([type], optional): [description]. Defaults to None. 
            steps ([type], optional): [description]. Defaults to None.
        """
        assert not(epochs is not None and steps is not None), "We will run either the specified SGD steps or specified epochs over data. We cannot run both"
        assert not(epochs is None and steps is None), "We need atleast one of steps or epochs to be specified"

        global_step = 0
        total_sgd_steps = np.inf
        total_epochs = 10
        if steps is not None:
            total_sgd_steps = steps
        if epochs is not None:
            total_epochs = epochs

        self._model.train()

        if loader is None:
            loader = self._trn_loader

        """This should come here so thta trn_wts[batch_ids] is consistent with the global ids later
        """
        # Initialize weights to perform average loss
        if trn_wts is None:
            trn_wts = torch.ones(len(loader.dataset)).to(cu.get_device())
        assert len(trn_wts) == len(loader.dataset), "Pass all weights. If you intend not to train on an example, then pass the weight as 0"

        if data_subset == constants.IDEAL_SUB:
            ideal_idxs = torch.where(self._trn_data._ideal_betas == 1)[0]
            loader = tdu.get_loader_subset(loader, subset_idxs=ideal_idxs, batch_size=self._batch_size, shuffle=True)
            if logger is not None:
                logger.info(f"Fitting nntheta on the ideal subset. After subsetting the Training data size is {len(loader.dataset)}")

        if constants.SCHEDULER in kwargs.keys():
            if constants.SCHEDULER_TYPE in kwargs.keys():
                raise NotImplementedError()
            self._lr_scheduler = tu.get_lr_scheduler(self._optimizer, scheduler_name="linear", n_rounds=epochs)
        if constants.OPTIMIZER in kwargs.keys():
            self._optimizer = kwargs[constants.OPTIMIZER]
        if constants.SW in kwargs.keys():
            self._sw = kwargs[constants.SW]

        for epoch in range(total_epochs):
            tq = tqdm(total=len(loader))
            for epoch_step, (batch_ids, x, y) in enumerate(loader):
                global_step += 1
                if global_step == total_sgd_steps:
                    return

                x, y, batch_ids = x.to(cu.get_device()), y.to(cu.get_device(), dtype=torch.int64), batch_ids.to(cu.get_device(), dtype=torch.int64)
                self._optimizer.zero_grad()
                
                # For xent loss, we need only pass unnormalized logits. https://stackoverflow.com/questions/49390842/cross-entropy-in-pytorch
                cls_out = self._model.forward(x)
                loss = self._xecri_perex(cls_out, y)
                loss = torch.dot(loss, trn_wts[batch_ids]) / torch.sum(trn_wts[batch_ids])
                
                loss.backward()
                self._optimizer.step()

                tq.set_postfix({"nnth_Loss": loss.item()})
                tq.update(1)

                if self._sw is not None:
                    self._sw.add_scalar("nnth_Loss", loss.item(), global_step)
            
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
        
            epoch_acc = self.accuracy()
            print(f"Epoch: {epoch} accuracy: {epoch_acc}")
            if self._sw is not None:
                self._sw.add_scalar("nnth_Epoch_Acc", epoch_acc, epoch)
            if logger is not None:
                logger.info(f"Epoch: {epoch} accuracy: {epoch_acc}")

    def grp_accuracy(self, loader=None, *args, **kwargs) -> dict:
        """Adding some more functionality to our Shapenet dataset
        """
        res_dict = super().grp_accuracy(loader=loader)
        
        if loader is None:
            loader = self._tst_loader
            ideal_beta = self._tst_data._ideal_betas
        else:
            ideal_beta = self._trn_data._ideal_betas
        
        ideal_idxs = torch.where(ideal_beta == 1)[0]
        non_ideal_idxs = torch.where(ideal_beta == 0)[0]
        
        res_dict["ideal_accuracy"] = self.accuracy(loader=tdu.get_loader_subset(loader, ideal_idxs))
        res_dict["non-ideal_accuracy"] = self.accuracy(loader=tdu.get_loader_subset(loader, non_ideal_idxs))
        return res_dict

    def recourse_accuracy(self, data_id, beta):
        pass

    @property
    def _def_name(self):
        return "resnet"
