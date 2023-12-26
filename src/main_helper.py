from calendar import EPOCH
import pickle as pkl

from scipy.fft import dst
from src.abstract.abs_our_method import MethodsHelper

import constants as constants
import numpy as np
import utils.common_utils as cu
import utils.torch_data_utils as tdu
from src.data_helper import SyntheticData, SyntheticDataHelper, ShapenetData, ShapenetDataHelper,GoogleTTSData,GoogleTTSDataHelper
from src.abstract.abs_data import DataHelper, Data
from src.abstract.abs_nn_theta import NNthHelper
from src.abstract.abs_greedy_rec import RecourseHelper
from src.nn_phi import ShapenetNNPhiMinHelper, SynNNPhiMinHelper
from src.nn_theta import LRNNthHepler, ResNETNNthHepler
import logging
from src.greedy_rec import ShapenetRecourseHelper, SynRecourseHelper
import torch
from src.methods import BaselineHelper
from src.nn_psi import ShapenetNNPsiHelper, SynNNPsiHelper

def get_data_helper(dataset_name, logger:logging.Logger = None):
    if logger is not None:
        logger.info(f"Loading dataset: {dataset_name}")
    if dataset_name == constants.SYNTHETIC:
        data_dir = constants.SYN_DIR
        print(f"Loading the dataset: {data_dir}")
        def process_data(fname):
            print(f"Loading: {fname}")
            if logger is not None:
                logger.info(f"Loading: {fname}")
            if "/" in fname: 
                with open(fname, "rb") as file:
                    shapenet_full = pkl.load(file)
            else:
                with open(data_dir / fname, "rb") as file:
                    shapenet_full = pkl.load(file)
            
            data_tuple = []
            for idx in range(7):
                X = np.array([shapenet_full[entry][idx] for entry in range(len(shapenet_full))])
                X = np.squeeze(X)
                data_tuple.append(X)
            return data_tuple
        train, test, val = process_data("training_synthetic-new1.pkl"), process_data("testing_synthetic-new1.pkl"), process_data("testing_synthetic.pkl")
        
        # fmt_data_train = lambda ds : (ds[0], ds[1], ds[2], ds[3], ds[4], ds[5], ds[6])
        # fmt_data_test = lambda ds : (ds[0], ds[1], ds[2], ds[3], None, None, ds[6])
        # train, test, val = fmt_data_train(train), fmt_data_test(test), fmt_data_test(val) 

    elif dataset_name == constants.SHAPENET_SAMPLE:
        data_dir = constants.SHAPENET_DIR
        with open(data_dir / "data_sample.pkl", "rb") as file:
            shapenet_sample = pkl.load(file)
        data_tuple = []
        for idx in range(7):
            X = np.array([shapenet_sample[entry][idx] for entry in range(900)])
            if idx == 3: # This is to make labels 0-indexed
                X = X-1
            X = np.squeeze(X)
            data_tuple.append(X)
        train, test, val = data_tuple, data_tuple, data_tuple

    elif dataset_name in [constants.GOOGLETTS,constants.SHAPENET, constants.SHAPENET_SMALL, constants.SHAPENET_NOISE_FULL, constants.SHAPENET_NOISE_SMALL]:
        data_dir = constants.SHAPENET_DIR_SAI
        def process_data(fname):
            print(f"Loading: {fname}")
            if logger is not None:
                logger.info(f"Loading: {fname}")
            if "/" in fname: 
                with open(fname, "rb") as file:
                    shapenet_full = pkl.load(file)
            else:
                with open(data_dir / fname, "rb") as file:
                    shapenet_full = pkl.load(file)
            
            data_tuple = []
            for idx in range(7):
                X = np.array([shapenet_full[entry][idx] for entry in range(len(shapenet_full))])
                X = np.squeeze(X)
                data_tuple.append(X)
            return data_tuple
        if dataset_name == constants.GOOGLETTS:
            train, test, val = process_data(constants.AUDIO_TRN_PATH), process_data(constants.AUDIO_TEST_PATH), process_data(constants.AUDIO_TEST_PATH)
        elif dataset_name == constants.SHAPENET_SMALL:
            train, test, val = process_data("training_shapenet_data_2_0.4.pkl"), process_data("testing_all_shapenet_data.pkl"), process_data("validation_shapenet_data.pkl")
        elif dataset_name == constants.SHAPENET_NOISE_FULL:
            train, test, val = process_data(constants.SHAPENET_FULL_NOISE_TRN_PATH), process_data(constants.SHAPENET_NOISE_TEST_PATH), process_data("validation_shapenet_data.pkl")
        elif dataset_name == constants.SHAPENET_NOISE_SMALL:
            train, test, val = process_data(constants.SHAPENET_SMALL_NOISE_TRN_PATH), process_data(constants.SHAPENET_NOISE_TEST_PATH), process_data("validation_shapenet_data.pkl")

    A = np.array
    
    if dataset_name == constants.SYNTHETIC:
        X, Z, Beta, Y, Ins, Sib, ideal_betas = train
        B_per_i = len(Sib[0])
        train_data = SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins, _, _ = test
        Ins = np.arange(len(X))
        test_data = SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        X, Z,  Beta, Y, Ins, _, _ = val
        Ins = np.arange(len(X))
        val_data = SyntheticData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas))

        dh = SyntheticDataHelper(train_data, test_data, val_data, train_test_data=train_data)
    
    elif dataset_name == constants.SHAPENET or \
        dataset_name == constants.SHAPENET_SAMPLE \
        or dataset_name == constants.SHAPENET_SMALL or \
        dataset_name == constants.SHAPENET_NOISE_FULL or \
        dataset_name == constants.SHAPENET_NOISE_SMALL:

        X, Z, Beta, Y, Ins, Sib, ideal_betas = train
        B_per_i = len(Sib[0])
        train_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["train"]
        }

        test_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["test"]
        }


        train_data = ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas), **train_args)

        train_test_data = ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas), **test_args)

        X, Z,  Beta, Y, Ins, _, ideal_betas = test
        test_data = ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **test_args)

        val_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["val"]
        }
        X, Z,  Beta, Y, Ins, _, ideal_betas = val
        val_data = ShapenetData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **val_args)
        dh = ShapenetDataHelper(train_data, test_data, val_data, train_test_data)
    elif dataset_name == constants.GOOGLETTS:
        X, Z, Beta, Y, Ins, Sib, ideal_betas = train
        B_per_i = len(Sib[0])
        train_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["train"]
        }

        test_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["test"]
        }


        train_data = GoogleTTSData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas), **train_args)

        train_test_data = GoogleTTSData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                            Siblings=A(Sib), 
                                            Z_ids=A(Ins),
                                            ideal_betas=A(ideal_betas), **test_args)

        X, Z,  Beta, Y, Ins, _, ideal_betas = test
        test_data = GoogleTTSData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **test_args)

        val_args = {
            constants.TRANSFORM: constants.RESNET_TRANSFORMS["val"]
        }
        X, Z,  Beta, Y, Ins, _, ideal_betas = val
        val_data = GoogleTTSData(A(X), A(Y), A(Z), A(Beta), B_per_i=B_per_i, 
                                        Siblings=None, Z_ids=A(Ins),
                                        ideal_betas=A(ideal_betas), **val_args)
        dh = GoogleTTSDataHelper(train_data, test_data, val_data, train_test_data)
    
    else:
        raise ValueError("Pass supported datasets only")
    
    if logger is not None:
        logger.info("Dataset loading completed")
    return dh


def fit_theta(nn_theta_type, models_defname, dh:DataHelper, fit, nnth_epochs, data_subset, logger:logging.Logger=None, *args, **kwargs):

    if nn_theta_type == constants.LOGREG:
        nnth_mh = LRNNthHepler(dh = dh, **kwargs)

    elif nn_theta_type == constants.RESNET:
        nnth_mh = ResNETNNthHepler(dh=dh, **kwargs)
        print("nn_theta is Resnet Model")
    else:
        raise NotImplementedError()

    if constants.GOOD_TRAINING in kwargs and kwargs[constants.GOOD_TRAINING] == True:
        logger.info("performing good training of NN theta on good samples obtained by dropping points with bad losses!")
        assert fit == True, "Fit better be true for Good training case"
        
        nnth_mh.load_model_defname(suffix=kwargs[constants.GOOD_INIT_MODEL])
        losses = nnth_mh.get_loaderlosses_perex(dh._train_test.get_theta_loader(batch_size=64, shuffle=False))
        srtd_idxs = torch.argsort(losses)
        good_pc = kwargs[constants.BTM_LOSSES]

        if logger is not None:
            logger.warn(f"Attempting to fit theta on a good subset with pc: {good_pc}")

        sel_ex = srtd_idxs[:int(len(srtd_idxs)*good_pc)]

        old_train = dh._train
        new_train = ShapenetData(X=old_train._X[sel_ex].numpy(), y=old_train._y[sel_ex].numpy(), Z=old_train._Z[sel_ex].numpy(),
                                    Beta=old_train._Beta[sel_ex].numpy(), B_per_i=old_train._B_per_i,
                                    Siblings=torch.empty_like(old_train._Siblings[sel_ex]).numpy(), Z_ids=old_train._Z_ids[sel_ex].numpy(),
                                    ideal_betas=old_train._ideal_betas[sel_ex].numpy(), transform=old_train.transform)

        dh._train = new_train
        logger.info("initializing the weights of nnth")
        if nn_theta_type == constants.LOGREG:
            nnth_mh = LRNNthHepler(dh = dh, **kwargs)

        elif nn_theta_type == constants.RESNET:
            nnth_mh = ResNETNNthHepler(dh=dh, **kwargs)
            print("nn_theta is Resnet Model")

        if logger is not None:
            logger.info(f"Changing the train dataset and nes DS has {dh._train._num_data} samples")
        dh._train = new_train

    if logger is not None:
        logger.info(f"nn_theta is: {nn_theta_type}")

    # fit
    if fit == True:
        print("Fitting nn_theta")
        if logger is not None:
            logger.info(f"Fittig nntheta for {nnth_epochs} Epochs")

        if data_subset != constants.FULL_DATA:
            print(f"Fitting on partial Data: {data_subset}")
            if logger is not None:
                logger.warning(f"Fitting on partial Data: {data_subset}")

        nnth_mh.fit_data(epochs=nnth_epochs, logger=logger, data_subset=data_subset, **kwargs)

        test_acc = nnth_mh.accuracy()
        print(f"Test Accuracy after fitting nn_theta: {test_acc}")
        if logger is not None:
            logger.info(f"Test Accuracy after fitting nn_theta: {test_acc}")

        nnth_mh.save_model_defname(suffix=models_defname)

        print(f"Grp Accuracy of {nn_theta_type} the ERM model is ")
        grp_acc = cu.dict_print(nnth_mh.grp_accuracy())
        if logger is not None:
            logger.info("Test Group Accuray is: ")
            logger.info(grp_acc)
            
            logger.info(f"Train Grp accuracy is: \
                {cu.dict_print(nnth_mh.grp_accuracy(loader=nnth_mh._dh._train.get_theta_loader(batch_size=128, shuffle=False)))}")
        
            # logger.info("Confusion matrix for Train data is:")
            # logger.info(cu.dict_print(nnth_mh.get_conf_matrix(loader=nnth_mh._dh._train.get_theta_loader(batch_size=128, shuffle=False))))
            # logger.info("Confusion matrix for test data is:")
            # logger.info(cu.dict_print(nnth_mh.get_conf_matrix()))
    # load
    else:
        if logger is not None:
            logger.info("Loading nth model as fit = False")
        
        if models_defname is not None: # Donot load the model is we dont pass a models_defname
            nnth_mh.load_model_defname(suffix=models_defname, logger=logger)

    print("nnth helper Ready!")
    if logger is not None:
        logger.info("nnth Ready")
    return nnth_mh


# def fit_R_theta(synR:ourr.RecourseHelper, scratch, models_defname, epochs=1, *args, **kwargs):
#     # rfit
#     synR.nnth_rfit(epochs=epochs, scratch=scratch, *args, **kwargs)
#     print(f"Accuracy after finetuning nntheta on Recourse set with weighted ERM is {synR._nnth.accuracy()}")
#     print(f"Grp Accuracy of the rfit finetuned model is ")
#     cu.dict_print(synR._nnth.grp_accuracy())
#     synR._nnth.save_model_defname(suffix=models_defname)



def fit_greedy(dataset_name, nnth:NNthHelper, load_th:bool, dh:DataHelper, budget, num_badex, R_per_iter, models_defname, 
                        fit, init_theta, logger:logging.Logger=None, *args, **kwargs):
    if logger is not None:
        logger.info(f"started fitting greedy recourse. Budget = {budget}")
        logger.info(f"Are we loading th100? - {load_th}")

    cu.set_seed()

    if init_theta == True:
        # initialize the model
        logger.info("Initializing the theta model as init_theta = True")
        nnth = fit_theta(nn_theta_type=kwargs["nnth_type"], models_defname=None, dh=dh, 
                                    nnth_epochs=0, data_subset=constants.FULL_DATA, fit=False, 
                                    logger=logger, **kwargs["nnth_args"])
        """
        This code is for warm start
        """
        logger.warn("To have a better theta on the onset of greedy algo, fitting theta for 5 epochs to give it a warmstart")
        nnth.fit_data(epochs=5)
        logger.info("Warm start accuracy and grp accuracy is:")
        logger.info(f"Accuracy: {nnth.accuracy()}")
        logger.info(f"Grp Accuracy:")
        logger.info(cu.dict_print(nnth.grp_accuracy()))

    if dataset_name == constants.SYNTHETIC:
        rechlpr = SynRecourseHelper(nnth=nnth, dh=dh, budget=budget, num_badex=num_badex, R_per_iter=R_per_iter,
                                        logger=logger, *args, **kwargs)
    elif dataset_name == constants.SHAPENET or \
        dataset_name == constants.SHAPENET_SAMPLE or \
        dataset_name == constants.SHAPENET_SMALL or \
        dataset_name == constants.SHAPENET_NOISE_SMALL or\
        dataset_name == constants.SHAPENET_NOISE_FULL:
        rechlpr = ShapenetRecourseHelper(nnth=nnth, dh=dh, budget=budget, num_badex=num_badex, R_per_iter=R_per_iter,
                                        logger=logger, *args, **kwargs)

    elif dataset_name == constants.GOOGLETTS:
        rechlpr = ShapenetRecourseHelper(nnth=nnth, dh=dh, budget=budget, num_badex=num_badex, R_per_iter=R_per_iter,
                                        logger=logger, *args, **kwargs)


    def load_recourse(models_defname):
        if models_defname is None:
            if logger is not None:
                logger.info("Not loading greedy recourse")
            return
        if logger is not None:
            logger.info("loading greedy recourse")
        rechlpr.load_recourse_state_defname(
                suffix=models_defname, model=load_th, logger=logger)
        rid_nosij = torch.sum(rechlpr.trn_wts[rechlpr._R] != 0)
        if logger is not None:
            logger.warn(
                    f"There are a total of {rid_nosij} objects without Sij. Removing all such examples")
        rechlpr._trn_wts[rechlpr._R] = 0
        assert torch.sum(
                rechlpr.trn_wts[rechlpr._R] != 0) == 0, "Now all R should have been removed."


    def fit_recourse(models_defname):
        print("Fitting Recourse")
        if logger is not None:
            logger.info("Fitting greedy recourse")
        rechlpr.recourse_theta(logger=logger)
        print(f"Accuracy on last step of Recourse: {rechlpr._nnth.accuracy()}")
        if logger is not None:
            logger.info(f"Accuracy on last step of Recourse: {rechlpr._nnth.accuracy()}")
        rechlpr.dump_recourse_state_defname(suffix=models_defname, model=True, logger=logger)
        if logger is not None:
            print(f"Grp Accuracy of last step of rec theta the ERM model is ")
            logger.info(f"Grp Accuracy of last step of rec theta the ERM model is ")
            dict_str = cu.dict_print(rechlpr._nnth.grp_accuracy())
            logger.info(dict_str)


    if constants.GOOD_RECOURSE in kwargs.keys() and kwargs[constants.GOOD_RECOURSE] == True:
        if logger is not None:
            logger.info("Performing good recourse training")
            logger.info("Providing additional recourse to: ", kwargs[constants.GOOD_INIT_RECOURSE])
        load_recourse(models_defname=kwargs[constants.GOOD_INIT_RECOURSE])
        if logger is not None:
            logger.info(f"The old recourse trained on old theta already has {len(rechlpr._R)} examples and we add: {budget} more examples")

    # fit
    if fit == True:
        fit_recourse(models_defname)

    # load
    else:
        load_recourse(models_defname)


    if logger is not None:
        print("Greedy Recourse Ready!")
        logger.info("Greedy recourse Ready")

    return rechlpr


def fit_nnphi(dataset_name, dh:DataHelper, epochs, greedy_rec:RecourseHelper, models_defname, fit, logger=None, *args, **kwargs):

    if dataset_name == constants.SYNTHETIC:
        nnpihHelper = SynNNPhiMinHelper(nn_arch=[10, 6], 
                                            rechlpr=greedy_rec, dh=dh, *args, **kwargs)
                                            
    elif dataset_name == constants.SHAPENET or dataset_name == constants.SHAPENET_SAMPLE or\
        dataset_name == constants.SHAPENET_SMALL or dataset_name==constants.SHAPENET_NOISE_FULL or\
        dataset_name == constants.SHAPENET_NOISE_SMALL or dataset_name == constants.GOOGLETTS:
        nnpihHelper = ShapenetNNPhiMinHelper(nn_arch=[128, 64, 16], 
                                            rechlpr=greedy_rec, dh=dh, *args, **kwargs)

    if logger is not None:
        logger.info(f"target betas are: {nnpihHelper.tgt_beta_unq}")
        logger.info(f"target beta counts are {nnpihHelper.tgt_beta_cnts}")

    # fit
    if fit == True:
        print("fitting NPhi")
        if logger is not None:
            logger.info("Fitting NNPhi")
        fit_args = {
            constants.SCHEDULER: True
        }
        nnpihHelper.fit_rec_beta(epochs=epochs, logger=logger, **fit_args)
        nnpihHelper.save_model_defname(suffix=models_defname, logger=logger)
    # load
    else:
        if models_defname is not None:
            nnpihHelper.load_model_defname(suffix=models_defname, logger=logger)
        else:
            print("nnphi is not loaded")
            if logger is not None:
                logger.info("nnphi is set to resnet init weigts and is not loaded")

    print("nnpi Ready!")
    return nnpihHelper


def fit_nnpsi(dataset_name,  dh:DataHelper, nn_arch, synR:RecourseHelper, epochs, models_defname, fit, psi_tgts, logger=None, *args, **kwargs):

    if dataset_name == constants.SYNTHETIC:
         nnpsiHelper = SynNNPsiHelper(out_dim=2, nn_arch=nn_arch, 
                                rechlpr=synR, psi_tgts = psi_tgts, dh=dh)
                                
    elif dataset_name == constants.SHAPENET or dataset_name == constants.SHAPENET_SMALL or \
        dataset_name==constants.SHAPENET_NOISE_FULL or\
        dataset_name == constants.SHAPENET_NOISE_SMALL or dataset_name == constants.GOOGLETTS:
        nnpsiHelper = ShapenetNNPsiHelper(out_dim=2, nn_arch=nn_arch, 
                                rechlpr=synR, psi_tgts = psi_tgts, dh=dh, **kwargs)

    # fit
    if fit == True:
        print("Fitting NNPsi")
        if logger is not None:
            logger.info("Fitting NNPsi")

            logger.info(f"Initial Accuracy is: {nnpsiHelper._rechlpr._nnth.accuracy()}")
            logger.info(f"Number of 1 targets = {sum(nnpsiHelper.trn_tgts)}")
        
        nnpsiHelper.fit_rec_r(epochs=epochs, logger=logger, **kwargs)
        nnpsiHelper.save_model_defname(suffix=models_defname, logger=logger)

    # load
    else:
        if models_defname is not None:
            nnpsiHelper.load_model_defname(suffix=models_defname, logger=logger)
        else:
            print("not loading nnpsi")
            if logger is not None:
                logger.info("not loading nnpsi")

    if logger is not None:
        logger.info("NNPsi is ready!")
    return nnpsiHelper


# # # %% Assessing three models
# def assess_thphipsi(dh, nnth, nnphi, nnpsi, pipeline=True):
#     """Assess the three models together

#     Args:
#         sdh ([type]): [description]
#         nnth ([type]): [description]
#         nnphi ([type]): [description]
#         nnpsi ([type]): [description]
#         pipeline (bool, optional): [description]. Defaults to True. This says if we should include recourse only for recourse needed examples.
#     """
#     if pipeline:
#         raw_acc, rec_acc, rs, pred_betas = tstm.assess_th_phi_psi(dh = dh, nnth=nnth, nnphi=nnphi, 
#                                                                     nnpsi=nnpsi)
#         print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
#         print(f"Asked recourse for {np.sum(rs)} and predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")
#     else:
#         raw_acc, rec_acc, pred_betas = tstm.assess_th_phi(dh = dh, nnth=nnth, nnphi=nnphi)
#         print(f"Raw acc={raw_acc}, Rec acc={rec_acc}")
#         print(f"predicted betas stats are {np.sum(pred_betas > 0.5, axis=0)}")


def get_ourm_hlpr(ourm_type, dh, nnth, nnphi, nnpsi, greedy_r, filter_grps, logger=None, **kwargs) -> MethodsHelper:

    if ourm_type == constants.SEQUENTIAL:
        mh = BaselineHelper(dh=dh, nnth=nnth, nnphi=nnphi, nnpsi=nnpsi, rechlpr=greedy_r, filter_grps=filter_grps, **kwargs)
    else:
        raise ValueError("Please pass a valid our method.")
    if logger is not None:
        logger.info(f"After making the weights to default, Theta accuracy = {nnth.accuracy()}")
        logger.warning(f"Filtering groups at: {filter_grps}")
    return mh