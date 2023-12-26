from src.abstract.abs_greedy_rec import RecourseHelper
from src.abstract.abs_data import Data, DataHelper
from src.abstract.abs_nn_theta import NNthHelper
import torch
import time
import utils.common_utils as cu
from src.models import ResNET
import constants as constants
from torch.optim import AdamW, SGD

class SynRecourseHelper(RecourseHelper):
    def __init__(self, nnth: NNthHelper, dh: DataHelper, budget, num_badex=100, R_per_iter=10, logger=None, *args, **kwargs) -> None:
        super(SynRecourseHelper, self).__init__(nnth, dh, budget, num_badex, R_per_iter=R_per_iter, logger=logger, *args, **kwargs)

    @property
    def _def_name(self):
        return "syn_greedy"

    def compute_gain(self, bad_exs, trn_wts) -> torch.Tensor:
        """Computes the gain of examples passed in bad_exs

        Args:
            bad_exs ([type]): [description]
        """
        gain = torch.ones(len(bad_exs)) * -1e10
        losses = self.get_trnloss_perex()
        ref_loss = torch.dot(trn_wts, losses)

        for idx, rid in enumerate(bad_exs):
            rid_wts = self.simulate_addr(trn_wts=trn_wts, R=self._R, rid=rid)
            recourse_loss = torch.dot(rid_wts, losses)
            gain[idx] = ref_loss - recourse_loss

        return gain
    
    def recourse_theta(self, logger=None, *args, **kwargs):

        # mimic argmin and get gain
        # select the one with highest gain
        # Adjust the trn wts
        # Perform one epoch with full min (better to have some tolerance)

        self.all_losses_cache = self.get_trnloss_perex()

        for r_iter in range(int(self._budget/self.num_r_per_iter)):
            start = time.time()
            self.set_Sij(margin=0)
            bad_exs = self.sample_bad_ex(self.R)
            gain = self.compute_gain(bad_exs=bad_exs, trn_wts=self._trn_wts)
            # sel_r = bad_exs[np.argmax(gain)]
            if self._sw is not None:
                self._sw.add_scalar("greedy_gain", torch.max(gain), r_iter)

            _, sel_r =  torch.topk(gain, self.num_r_per_iter)
            print(f"Gain = {torch.mean(gain)}")

            for sel_ridx in sel_r:
                self._R = torch.cat([self._R, bad_exs[sel_ridx].view(-1)])
                self._trn_wts = self.simulate_addr(self._trn_wts, self._R, bad_exs[sel_ridx])
            
            self.all_losses_cache = None
            self.minze_theta(self._nnth._trn_loader, torch.Tensor(self._trn_wts).to(cu.get_device()))
            self.all_losses_cache = self.get_trnloss_perex()
            rec_loss = torch.dot(self.get_trnloss_perex(), self._trn_wts)
            print(f"Inside R iteration {r_iter}; Loss after minimizing on adding {len(sel_r)} indices is {rec_loss}", flush=True)

            if logger is not None:
                logger.info(f"r_iter: {r_iter}, avg gain: {torch.mean(gain)}, greedy_loss: {rec_loss}")
            if self._sw is not None:
                self._sw.add_scalar("greedy_Loss", rec_loss, r_iter)

            print(f"Time taken = {time.time() - start}")

            if logger is not None and r_iter % 10 == 0:
                logger.info(f"After {r_iter} R iterations, accuracy is {self._nnth.accuracy()}")
                logger.info("Grp accuracy is:")
                logger.info(cu.dict_print(self._nnth.grp_accuracy()))

        self.all_losses_cache = None
        return self.R, self._Sij, self._trn_wts

    def apply_recourse(self, data_id, beta):
        """Applies recourse to a test image. We take in the data id and return the x corresponding to asked beta

        Args:
            data_id ([type]): [description]
            beta ([type]): [description]
        """
        pass



class ShapenetRecourseHelper(RecourseHelper):
    def __init__(self, nnth: NNthHelper, dh: DataHelper, budget, num_badex=100, R_per_iter=10, logger=None, *args, **kwargs) -> None:
        super(ShapenetRecourseHelper, self).__init__(nnth, dh, budget, num_badex, R_per_iter=R_per_iter, logger=logger, *args, **kwargs)

    @property
    def _def_name(self):
        return "shapenetgreedy"

    def compute_gain(self, bad_exs, trn_wts) -> torch.Tensor:
        """Computes the gain of examples passed in bad_exs

        Args:
            bad_exs ([type]): [description]
        """
        gain = torch.ones(len(bad_exs)) * -1e10
        losses = self.get_trnloss_perex()
        ref_loss = torch.dot(trn_wts, losses)

        for idx, rid in enumerate(bad_exs):
            rid_wts = self.simulate_addr(trn_wts=trn_wts, R=self._R, rid=rid)
            recourse_loss = torch.dot(rid_wts, losses)
            gain[idx] = ref_loss - recourse_loss

        return gain
    
    def recourse_theta(self, logger=None, *args, **kwargs):

        # mimic argmin and get gain

        # select the one with highest gain

        # Adjust the trn wts

        # Perform one epoch with full min (better to have some tolerance)

        self.all_losses_cache = self.get_trnloss_perex()

        for r_iter in range(int(self._budget/self.num_r_per_iter)):

            start = time.time()

            self.set_Sij(margin=0)

            bad_exs = self.sample_bad_ex(self.R)

            gain = self.compute_gain(bad_exs=bad_exs, trn_wts=self._trn_wts)

            # sel_r = bad_exs[np.argmax(gain)]

            if self._sw is not None:
                self._sw.add_scalar("greedy_gain", torch.max(gain), r_iter)

            _, sel_r =  torch.topk(gain, self.num_r_per_iter)
            print(f"Gain = {torch.mean(gain)}")

            for sel_ridx in sel_r:
                self._R = torch.cat([self._R, bad_exs[sel_ridx].view(-1)])
                self._trn_wts = self.simulate_addr(self._trn_wts, self._R, bad_exs[sel_ridx])
            
            self.all_losses_cache = None

            self.minze_theta(self._nnth._trn_loader, torch.Tensor(self._trn_wts).to(cu.get_device()))
            
            self.all_losses_cache = self.get_trnloss_perex()

            rec_loss = torch.dot(self.get_trnloss_perex(), self._trn_wts)

            print(f"Inside R iteration {r_iter}; Loss after minimizing on adding {len(sel_r)} indices is {rec_loss}", flush=True)
            
            if logger is not None:
                logger.info(f"r_iter: {r_iter}, avg gain: {torch.mean(gain)}, greedy_loss: {rec_loss}")
            if self._sw is not None:
                self._sw.add_scalar("greedy_Loss", rec_loss, r_iter)

            print(f"Time taken = {time.time() - start}")

            if logger is not None and r_iter % 10 == 0:
                logger.info(f"After {r_iter} R iterations, accuracy is {self._nnth.accuracy()}")
                logger.info("Grp accuracy is:")
                logger.info(cu.dict_print(self._nnth.grp_accuracy()))

        self.all_losses_cache = None

        return self.R, self._Sij, self._trn_wts

    def apply_recourse(self, data_id, beta):
        """Applies recourse to a test image. We take in the data id and return the x corresponding to asked beta

        Args:
            data_id ([type]): [description]
            beta ([type]): [description]
        """
        pass