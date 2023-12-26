from src.abstract.abs_data import  DataHelper
from src.abstract.abs_nn_phi import NNPhiHelper
from src.abstract.abs_nn_theta import NNthHelper
from src.abstract.abs_greedy_rec import RecourseHelper
from src.abstract.abs_our_method import MethodsHelper
import constants as constants
import torch.optim as optim
import warnings
from tqdm import tqdm
import torch
import utils.common_utils as cu

class BaselineHelper(MethodsHelper):
    def __init__(self, dh: DataHelper, nnth:NNthHelper, nnphi:NNPhiHelper, rechlpr: RecourseHelper, filter_grps:int, *args, **kwargs) -> None:
        super().__init__(dh, nnth, nnphi, rechlpr, *args, **kwargs)
        if filter_grps < 0:
            filter_grps = dh._train._num_Z
        self.filter_grps = filter_grps
        pred_losses = self._nnth.get_loaderlosses_perex(loader=self._dh._train_test.get_theta_loader(batch_size=128, shuffle=False))
        pred_losses = pred_losses.view(-1, dh._train._B_per_i)
        min_losses, _ = torch.min(pred_losses, dim=1)
        sorted_idxs = torch.argsort(min_losses)
        self.selected_grps = sorted_idxs[0:self.filter_grps] * dh._train._B_per_i
        self.tst_pred_labels = self._nnth.predict_labels(loader=self._nnth._tst_loader)

        # Set all the optimizers
        self._optim = optim.AdamW([
            {'params': self._phimodel.parameters()},
            {'params': self._thmodel.parameters()},
            {'params': self._psimodel.parameters()}
        ], lr=self._lr)

        if self.pretrn_models[constants.THETA] == True:
            self._thoptim = optim.AdamW([
                {'params': self._thmodel.parameters()}
            ], lr = 1e-6)
        else:
            self._thoptim = optim.AdamW([
                {'params': self._thmodel.parameters()}
            ], lr = self._lr)
        
        self._phioptim = optim.AdamW([
            {'params': self._phimodel.parameters()}
        ], lr = self._lr)

    @property
    def _def_name(self):
        return super()._def_name + constants.SEQUENTIAL

    @property
    def _def_dir(self):
        return super()._def_dir / "baseline"

    def fit_epoch(self, epoch, grp_loader=None, logger=None, enforce_beta_prior=False, *args, **kwargs):

        inter_iters = None
        inter_epochs = None

        self._thmodel.train()
        self._phimodel.train()
        self._psimodel.eval() # For baseline we dont need to involve psi anyways

        if constants.INTERLEAVE_ITERS in kwargs.keys():
            inter_iters = kwargs[constants.INTERLEAVE_ITERS]
            raise ValueError("This strategy works bur inreleave epochs is more graceful.")
        elif constants.INTERLEAVE_EPOCHS in kwargs.keys():
            inter_epochs = kwargs[constants.INTERLEAVE_EPOCHS]
            # Dont flow gradients to the models that we dont need training
            if inter_epochs[constants.THETA] < 0:
                self._thmodel.eval()
            if inter_epochs[constants.PHI] < 0:
                self._phimodel.eval()

        pred_betas = [] # This is also purely for logging only
        ys = []

        ideal_betas = self._dh._train._ideal_betas
        ideal_betas = torch.where(ideal_betas == 1)[0]

        if grp_loader is None:
            grp_loader = self._trngrp_loader
        else:
            logger.warn("Are u sure that u are passing the right group loader?")
            warnings.warn("Are u sure that u are passing the right group loader?")
        
        global_step = epoch * len(grp_loader)
        tq = tqdm(total=len(grp_loader), desc="Loss")

        num_grp = self._trn_data.B_per_i
        num_beta = self._trn_data._num_beta

        get_grp = lambda idx: torch.arange(int(idx)*num_grp, (idx+1)*num_grp).to(dtype=torch.int64, device=cu.get_device())

        one_vec = torch.tensor(1).view(-1).to(cu.get_device(), dtype=torch.int64)

        for local_step, (dataid_grp, X_grp, Beta_grp, y_grp) in enumerate(grp_loader):
            util_grp = 0

            self._thoptim.zero_grad()
            self._phioptim.zero_grad()

            X_grp, Beta_grp, y_grp = X_grp.to(cu.get_device()), Beta_grp.to(cu.get_device()), y_grp.to(cu.get_device(), dtype=torch.int64)
            X_flat, Beta_flat = X_grp.view(-1, *X_grp.shape[2:]), Beta_grp.view(-1, *Beta_grp.shape[2:])

            beta_phi_flat = self._phimodel.forward_proba(X_flat, Beta_flat)
            y_theta_flat = self._thmodel.forward_proba(X_flat)

            for local_grp_idx, (dataid, x, beta, y)  in enumerate(zip(dataid_grp, X_grp, Beta_grp, y_grp)):
                
                prior_beta = None
                do_cls, do_phi = cu.get_do_cls_phi(epoch=epoch, local_step=local_step, inter_epochs=inter_epochs, inter_iters=inter_iters)
                if not (dataid[0] in self.selected_grps):
                    if do_cls == True:
                        continue
                    elif do_phi == True:
                        prior_beta = self.prior[y[0].item()]
                    else:
                        raise ValueError("phi and cls should be interleaving!")


                grp_idxs = get_grp(local_grp_idx)
                beta_phi = [entry[grp_idxs] for entry in beta_phi_flat]
                ypred = y_theta_flat[grp_idxs]
                
                label = lambda b : torch.argmax(b, dim=1).view(-1, 1)
                beta_phi_labels = torch.hstack([label(beta_phi[0]), label(beta_phi[1]), label(beta_phi[2])])

                # rpt stands for repeated matrices and rptI stands for repeat interleave matrices
                beta_phi_rptI = [entry.repeat_interleave(num_grp, dim=0) for entry in beta_phi]
                
                if do_phi and prior_beta is not None:
                    beta_rpt = prior_beta.repeat(num_grp*len(beta), 1)
                else:
                    beta_rpt = beta.repeat(num_grp, 1)
                
                beta_phi_gthr = [torch.gather(input=beta_phi_rptI[beta_idx], dim=1, index=beta_rpt[:, beta_idx].view(-1,1)).squeeze() \
                                    for beta_idx in range(self._trn_data._num_beta)]
                beta_phi_gthr = [torch.log(entry) for entry in beta_phi_gthr]
                beta_phi_gthr = torch.sum(torch.stack(beta_phi_gthr, dim=1), dim=1)
                
                ypred = torch.gather(input=ypred, dim=1, index=y.view(-1,1))
                ypred_rpt = ypred.repeat(num_grp, 1)
                ypred_rpt = torch.log(ypred_rpt+1e-5)

                likelihood = beta_phi_gthr.squeeze() + ypred_rpt.squeeze()
                likelihood = likelihood.view(-1, num_grp)
                likelihood, sel_beta_idxs = torch.max(likelihood, dim=1)

                # This is for loggig purposes
                pred_betas.append(beta_phi_labels.cpu())
                ys.append(y.cpu())
                
                util_grp += torch.sum(likelihood)
              
            if util_grp == 0:
                continue

            do_cls, do_phi = cu.get_do_cls_phi(epoch=epoch, local_step=local_step, inter_epochs=inter_epochs, inter_iters=inter_iters)
            loss = -util_grp/len(X_grp)
            cls_loss, phi_loss = torch.scalar_tensor(0), torch.scalar_tensor(0)

            # minimize theta
            if do_cls > 0:
                cls_loss = do_cls * loss
                cls_loss.backward()
                self._thoptim.step()
            if do_phi > 0:
                phi_loss = do_phi * loss
                phi_loss.backward()
                self._phioptim.step()
            
            self._sw.add_scalar("baseline_thLoss", cls_loss.item(), global_step+local_step)
            self._sw.add_scalar("baseline_phiLoss", phi_loss.item(), global_step+local_step)

            tq.set_description(f"thLoss: {cls_loss.item()}, phiLoss: {phi_loss.item()}")
            tq.update(1)
        
        if logger is not None:
            logger.info(f"groups cut at {len(self.selected_grps)}")
            pred_betas = torch.cat(pred_betas).view(-1, num_beta)
            ys = torch.cat(ys)

            y_test = self._dh._test._y
            
            def cls_str(beta):
                cnts = torch.unique(beta, dim=0, return_counts=True)
                cls_str = ""
                for i, j in zip(cnts[0], cnts[1]):
                    cls_str += f" :: {i} -> {j}"
                return cls_str

            for cls in self._dh._test._classes:
                logger.info(f"***************Class {cls}: prior {self.prior[cls]}*******************")
                cls_idxs = torch.where(ys == cls)[0]
                logger.info(f"For trn data: {cls_str(pred_betas[cls_idxs])}")
                cls_idxs = torch.where(y_test == cls)[0]
            logger.info(f"*********************Epoch {epoch} completed***********************")

    def rec_acc(self, pred_betas):
        tst_predlabels = self.tst_pred_labels
        tst_betas = self._dh._test._Beta
        beta_to_idx = {}
        tst_labels = self._dh._test._y
        for i, b in enumerate(tst_betas[0:9]):
            beta_to_idx[str(b.tolist())] = i

        corrects = torch.zeros(self._dh._test._num_classes)
        counts = torch.zeros(self._dh._test._num_classes)
        for i in range(len(tst_predlabels)):
            beta_pred = pred_betas[i]
            try:
                idx = int(i/9)*9 + beta_to_idx[str(beta_pred.tolist())]
                label_pred = tst_predlabels[idx]
            except:
                label_pred = tst_predlabels[i]
            if label_pred == tst_labels[i]:
                corrects[tst_labels[i]] = corrects[tst_labels[i]] + 1
            counts[tst_labels[i]] = counts[tst_labels[i]] + 1
        for i in range(self._dh._test._num_classes):
            corrects[i] = corrects[i] / counts[i]
        return torch.mean(corrects)