import constants as constants


config_audio = {
    constants.GENERAL_SPECS: {
        constants.DATASET: constants.GOOGLETTS,
        constants.TRNDATA_SUBSET: constants.FULL_DATA # Important
    },

    constants.THETA_SPECS: {
        constants.MODEL_TYPE: constants.RESNET,
        constants.MODEL_NAME:"final-theta-baseline-audio", # important
        constants.EPOCHS: 5,
        constants.LRN_RATTE: 1e-3,
        constants.SCHEDULER: True,
    },
    
    constants.GREEDY_SPECS: {
        constants.MODEL_NAME: "final-greedy-audio", #Important
        constants.LOAD_THETA: True,
        constants.BUDGET: 30,
        constants.BADEX_PERITER: 100,
        constants.NUMR_PERITER: 10
    },

    # We will learn phi using the joint objective.
    constants.PHI_SPECS: {
        constants.MODEL_NAME: None, # important
        constants.EPOCHS: 30,
        constants.BATCH_NORM: True,
        constants.LRN_RATTE: 1e-2,
        constants.SCHEDULER: True
    },

    constants.OURM_SPECS: {
        constants.OURM_TYPE: constants.SEQUENTIAL,
        constants.MODEL_NAME: "final-phi-audio",
        constants.OURM_ARGS:  {
                        constants.PRETRN_THPHI: {
                        constants.THETA: True, # We will initialize theta with trained weights
                        constants.PHI: False,
                        }
                    },
        constants.EPOCHS: 5,
        constants.BETA_PRIOR: True,
        constants.FILTER_GRPS: 1000,
        constants.INTERLEAVE: True,
        constants.INTERLEAVE_TYPE: constants.INTERLEAVE_EPOCHS,
        constants.INTERLEAVE_EPOCHS: { 
            constants.PHI: 3,
            constants.THETA: 2,
        },
        constants.INTERLEAVE_ITERS: {
            constants.THETA: 20,
            constants.PHI: 20,
        }
    },
   
    constants.TRAIN_ARGS: {  # important
        constants.THETA: True,
        constants.GREEDY: True,
        constants.PHI: False,
        constants.OUR_METHOD: True
    },

    constants.EXPT_NAME: "Audio",
    "logger": constants.LOG_DIR / "Audio.log", # important
    "gpu_id": 0,
    constants.SW: constants.TB_DIR / "Audio" # important
}



config_shapenet_large = {
    constants.GENERAL_SPECS: {
        constants.DATASET: constants.SHAPENET_NOISE_FULL,
        constants.TRNDATA_SUBSET: constants.FULL_DATA # Important
    },

    constants.THETA_SPECS: {
        constants.MODEL_TYPE: constants.RESNET,
        constants.MODEL_NAME:"final-theta-baseline-large", # important
        constants.EPOCHS: 5,
        constants.LRN_RATTE: 1e-3,
        constants.SCHEDULER: True,
    },
    
    constants.GREEDY_SPECS: {
        constants.MODEL_NAME: "final-greedy-large", # Important
        constants.LOAD_THETA: True,
        constants.BUDGET: 30,
        constants.BADEX_PERITER: 100,
        constants.NUMR_PERITER: 10
    },

    # We will learn phi using the joint objective.
    constants.PHI_SPECS: {
        constants.MODEL_NAME: None, # important
        constants.EPOCHS: 30,
        constants.BATCH_NORM: True,
        constants.LRN_RATTE: 1e-2,
        constants.SCHEDULER: True
    },

    constants.OURM_SPECS: {
        constants.OURM_TYPE: constants.SEQUENTIAL,
        constants.MODEL_NAME: "final-phi-large",
        constants.OURM_ARGS:  {
                        constants.PRETRN_THPHI: {
                        constants.THETA: True, # We will initialize theta with trained weights
                        constants.PHI: False,
                        }
                    },
        constants.EPOCHS: 5,
        constants.BETA_PRIOR: True,
        constants.FILTER_GRPS: 1000,
        constants.INTERLEAVE: True,
        constants.INTERLEAVE_TYPE: constants.INTERLEAVE_EPOCHS,
        constants.INTERLEAVE_EPOCHS: { 
            constants.PHI: 3,
            constants.THETA: 2,
        },
        constants.INTERLEAVE_ITERS: {
            constants.THETA: 20,
            constants.PHI: 20,
        }
    },
   
    constants.TRAIN_ARGS: {  # important
        constants.THETA: True,
        constants.GREEDY: True,
        constants.PHI: False,
        constants.OUR_METHOD: True
    },

    constants.EXPT_NAME: "Shapenet_large",
    "logger": constants.LOG_DIR / "Shapenet_large.log", # important
    "gpu_id": 0,
    constants.SW: constants.TB_DIR / "Shapenet_large" # important
}


config_shapenet_small = {
    constants.GENERAL_SPECS: {
        constants.DATASET: constants.SHAPENET_NOISE_SMALL,
        constants.TRNDATA_SUBSET: constants.FULL_DATA # Important
    },

    constants.THETA_SPECS: {
        constants.MODEL_TYPE: constants.RESNET,
        constants.MODEL_NAME:"final_theta_baseline_small", # important
        constants.EPOCHS: 5,
        constants.LRN_RATTE: 1e-3,
        constants.SCHEDULER: True,
    },
    
    constants.GREEDY_SPECS: {
        constants.MODEL_NAME: "final-greedy-small", # Important
        constants.LOAD_THETA: True,
        constants.BUDGET: 30,
        constants.BADEX_PERITER: 100,
        constants.NUMR_PERITER: 10
    },

    # We will learn phi using the joint objective.
    constants.PHI_SPECS: {
        constants.MODEL_NAME: None, # important
        constants.EPOCHS: 30,
        constants.BATCH_NORM: True,
        constants.LRN_RATTE: 1e-2,
        constants.SCHEDULER: True
    },

    constants.OURM_SPECS: {
        constants.OURM_TYPE: constants.SEQUENTIAL,
        constants.MODEL_NAME: "final-phi-small",
        constants.OURM_ARGS:  {
                        constants.PRETRN_THPHI: {
                        constants.THETA: True, # We will initialize theta with trained weights
                        constants.PHI: False,
                        }
                    },
        constants.EPOCHS: 5,
        constants.BETA_PRIOR: True,
        constants.FILTER_GRPS: 1000,
        constants.INTERLEAVE: True,
        constants.INTERLEAVE_TYPE: constants.INTERLEAVE_EPOCHS,
        constants.INTERLEAVE_EPOCHS: { 
            constants.PHI: 3,
            constants.THETA: 2,
        },
        constants.INTERLEAVE_ITERS: {
            constants.THETA: 20,
            constants.PHI: 20,
        }
    },
   
    constants.TRAIN_ARGS: {  # important
        constants.THETA: True,
        constants.GREEDY: True,
        constants.PHI: False,
        constants.OUR_METHOD: True
    },

    constants.EXPT_NAME: "Shapenet_small_training",
    "logger": constants.LOG_DIR / "Shapenet_small_training.log", # important
    "gpu_id": 0,
    constants.SW: constants.TB_DIR / "Shapenet_small_training" # important
}


config_synthetic = {
    constants.GENERAL_SPECS: {
        constants.DATASET: constants.SYNTHETIC,
        constants.TRNDATA_SUBSET: constants.FULL_DATA # Important
    },

    constants.THETA_SPECS: {
        constants.MODEL_TYPE: constants.LOGREG,
        constants.MODEL_NAME:"synthetic", # important
        constants.EPOCHS: 5,
        constants.LRN_RATTE: 1e-3,
        constants.SCHEDULER: True,
    },
    
    constants.GREEDY_SPECS: {
        constants.MODEL_NAME: "greedy-synthetic", # Important
        constants.LOAD_THETA: True,
        constants.BUDGET: 30,
        constants.BADEX_PERITER: 100,
        constants.NUMR_PERITER: 10
    },

    # We will learn phi using the joint objective.
    constants.PHI_SPECS: {
        constants.MODEL_NAME: None, # important
        constants.EPOCHS: 30,
        constants.BATCH_NORM: True,
        constants.LRN_RATTE: 1e-2,
        constants.SCHEDULER: True
    },

    constants.OURM_SPECS: {
        constants.OURM_TYPE: constants.SEQUENTIAL,
        constants.MODEL_NAME: "synthetic-phi",
        constants.OURM_ARGS:  {
                        constants.PRETRN_THPHI: {
                        constants.THETA: True, # We will initialize theta with trained weights
                        constants.PHI: False,
                        }
                    },
        constants.EPOCHS: 5,
        constants.BETA_PRIOR: True,
        constants.FILTER_GRPS: 1000,
        constants.INTERLEAVE: True,
        constants.INTERLEAVE_TYPE: constants.INTERLEAVE_EPOCHS,
        constants.INTERLEAVE_EPOCHS: { 
            constants.PHI: 3,
            constants.THETA: 2,
        },
        constants.INTERLEAVE_ITERS: {
            constants.THETA: 20,
            constants.PHI: 20,
        }
    },
   
    constants.TRAIN_ARGS: {  # important
        constants.THETA: True,
        constants.GREEDY: True,
        constants.PHI: False,
        constants.OUR_METHOD: True
    },

    constants.EXPT_NAME: "Synthetic training",
    "logger": constants.LOG_DIR / "Synthetic_training.log", # important
    "gpu_id": 0,
    constants.SW: constants.TB_DIR / "Syntetic_training" # important
}