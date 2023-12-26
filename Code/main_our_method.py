import utils.common_utils as cu
import sys
from config import config_shapenet_large,config_shapenet_small,config_synthetic,config_audio
import argparse
import logging
import torch.utils.tensorboard
from torch.utils.tensorboard.writer import SummaryWriter
import constants as constants
import src.main_helper as main_helper


def parseDataset(str):
    if str not in ['shapenet-large','shapenet-small','synthetic','audio']:
        raise argparse.ArgumentTypeError('dataset has to be one among shapenet-large,shapenet-small,synthetic')
    return str


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",type = parseDataset,help="Please specify one of the following datasets (i) shapenet-large (ii) shapenet-small (iii) synthetic")
    args = parser.parse_args()
    print(args.dataset)
    if args.dataset == 'shapenet-large':
        config = config_shapenet_large
    elif args.dataset == 'shapenet-small':
        config = config_shapenet_small
    elif args.dataset == 'synthetic':
        config = config_synthetic
    elif args.dataset == 'audio':
        config = config_audio
    else:
        raise argparse.ArgumentError("Please specify one of the following datasets (i) shapenet-large (ii) shapenet-small (iii) synthetic")

    cu.set_cuda_device(config["gpu_id"])
    cu.set_seed(42)


    # %% Parse Config
    dataset_name = config[constants.GENERAL_SPECS][constants.DATASET]
    data_subset = config[constants.GENERAL_SPECS][constants.TRNDATA_SUBSET]

    fit_th = config[constants.TRAIN_ARGS][constants.THETA]
    fit_greedy = config[constants.TRAIN_ARGS][constants.GREEDY]
    fit_phi = config[constants.TRAIN_ARGS][constants.PHI]
    fit_ourm = config[constants.TRAIN_ARGS][constants.OUR_METHOD]

    nnth_type = config[constants.THETA_SPECS][constants.MODEL_TYPE]
    nnth_name = config[constants.THETA_SPECS][constants.MODEL_NAME]
    nnth_epochs = config[constants.THETA_SPECS][constants.EPOCHS]
    nnth_lr = config[constants.THETA_SPECS][constants.LRN_RATTE]
    nnth_lrs = config[constants.THETA_SPECS][constants.SCHEDULER]

    greedy_name = config[constants.GREEDY_SPECS][constants.MODEL_NAME]
    greedy_load_theta = config[constants.GREEDY_SPECS][constants.LOAD_THETA]
    greedy_budget = config[constants.GREEDY_SPECS][constants.BUDGET]
    greedy_badex_iter = config[constants.GREEDY_SPECS][constants.BADEX_PERITER]
    greedy_r_iter = config[constants.GREEDY_SPECS][constants.NUMR_PERITER]

    phi_name = config[constants.PHI_SPECS][constants.MODEL_NAME]
    phi_epochs = config[constants.PHI_SPECS][constants.EPOCHS]
    phi_lr = config[constants.PHI_SPECS][constants.LRN_RATTE]
    phi_lrs = config[constants.PHI_SPECS][constants.SCHEDULER]
    phi_batchnorm = config[constants.PHI_SPECS][constants.BATCH_NORM]

    ourm_type = config[constants.OURM_SPECS][constants.OURM_TYPE]
    ourm_name = config[constants.OURM_SPECS][constants.MODEL_NAME]
    ourm_args = config[constants.OURM_SPECS][constants.OURM_ARGS]
    ourm_epochs = config[constants.OURM_SPECS][constants.EPOCHS]
    ourm_interleave = config[constants.OURM_SPECS][constants.INTERLEAVE]
    ourm_interleave_type = config[constants.OURM_SPECS][constants.INTERLEAVE_TYPE]
    ourm_inter_iter_dict = config[constants.OURM_SPECS][constants.INTERLEAVE_ITERS]
    ourm_inter_epochs_dict = config[constants.OURM_SPECS][constants.INTERLEAVE_EPOCHS]
    ourm_beta_prior = config[constants.OURM_SPECS][constants.BETA_PRIOR]
    ourm_filter_grps = config[constants.OURM_SPECS][constants.FILTER_GRPS]

    sw = SummaryWriter(config[constants.SW])
    # Create and configure logger
    logging.basicConfig(filename=str(config["logger"].absolute()),
                        format='%(asctime)s :: %(filename)s:%(funcName)s :: %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.info(cu.dict_print(config))

# %% Create all the needed objects

    dh = main_helper.get_data_helper(dataset_name=dataset_name, logger=logger)
    logger.info(
        f"num_train: {dh._train._num_data}, num_test: {dh._test._num_data}")

    nnth_args = {
        constants.SW: sw,
        constants.BATCH_SIZE: 32,
        constants.LRN_RATTE: nnth_lr,
        constants.SCHEDULER: nnth_lrs,
        constants.MOMENTUM: 0.9,
    }
    nnth_mh = main_helper.fit_theta(nn_theta_type=nnth_type, models_defname=nnth_name,
                                    dh=dh, nnth_epochs=nnth_epochs,
                                    fit=fit_th, data_subset=data_subset, logger=logger, **nnth_args)


# %% Greedy Recourse

    greedy_args = {
        constants.SW: sw,
        constants.BATCH_SIZE: 128,
        constants.GOOD_RECOURSE: greedy_good_incr,
        "nnth_args": nnth_args,
        "nnth_type": nnth_type
    }
    greedy_r = main_helper.fit_greedy(dataset_name=dataset_name, nnth=nnth_mh, load_th=greedy_load_theta, dh=dh, budget=greedy_budget, 
                                      num_badex=greedy_badex_iter, R_per_iter=greedy_r_iter, models_defname=greedy_name,
                                      fit=fit_greedy, logger=logger, **greedy_args)


# # %% NNPhi
    nnphi_args = {
        constants.SW: sw,
        constants.SCHEDULER: phi_lrs,
        constants.BATCH_NORM: phi_batchnorm
    }
    nnphi = main_helper.fit_nnphi(dataset_name=dataset_name, dh=dh, epochs=phi_epochs, greedy_rec=greedy_r, models_defname=phi_name,
                                  fit=fit_phi, logger=logger, **nnphi_args)


# # # # %% Kick starts our method training
    ourm_args = cu.insert_kwargs(ourm_args, {
        constants.SW: sw,
        })

    ourm_hlpr = main_helper.get_ourm_hlpr(ourm_type=ourm_type, dh=dh, nnth=nnth_mh,
                                            nnphi=nnphi, greedy_r=greedy_r, 
                                            filter_grps = ourm_filter_grps, logger=logger, **ourm_args)

    logger.info(f"starting our method: {ourm_type}")
    logger.info(f"Initial Accuracy is: {ourm_hlpr._nnth.accuracy()}")

    logger.info("Prior on beta at the initial theta is:")
    logger.info(cu.dict_print(ourm_hlpr.prior))

    if fit_ourm == False:
        ourm_hlpr.load_model_defname(suffix=f"-{ourm_name}")
        print("Done loading recourse and the predicted betas are is:")
        pred_betas = ourm_hlpr._nnphi.predict_beta(loader=ourm_hlpr._nnphi.tst_loader)
        print(torch.unique(pred_betas, dim=0, return_counts=True))

    fit_args = {}
    if ourm_interleave == True:
        if ourm_interleave_type == constants.INTERLEAVE_ITERS:
            fit_args[constants.INTERLEAVE_ITERS] = ourm_inter_iter_dict
            logger.info("Interleaved training with SGD iterations")
        elif ourm_interleave_type == constants.INTERLEAVE_EPOCHS:
            fit_args[constants.INTERLEAVE_EPOCHS] = ourm_inter_epochs_dict
            logger.info("interleaved training with epochs")
        logger.info(cu.dict_print(fit_args))

    # fit and test
    for epoch in range(ourm_epochs):
        ourm_hlpr.fit_epoch(epoch=epoch, logger=logger, enforce_beta_prior=ourm_beta_prior, enforce_rank_prior=ourm_ranking_prior, **fit_args)

        if epoch >= 50 and (epoch%10) == 0:
            ourm_hlpr.save_model_defname(suffix=f"-{ourm_name}-epoch-{epoch}")
            logger.info(f"$$$$$$$$$$$$ Saving model at epoch {epoch} $$$$$$$$$$$$$$$$$$")
    
    logger.info(cu.dict_print(ourm_hlpr._nnth.grp_accuracy()))
    logger.info(cu.dict_print(ourm_hlpr._nnth.beta_accuracy()))
    ourm_hlpr.save_model_defname(suffix=f"-{ourm_name}")

# %%