from pathlib import Path
from torchvision import transforms

GENERAL_SPECS = "general_specs"
THETA_SPECS = "theta_specs"
GREEDY_SPECS = "greedy_specs"
PHI_SPECS = "phi_specs"
OURM_SPECS = "our_method_specs"

SYN_DIR = Path("./data")
SHAPENET_DIR = Path("./data")
SHAPENET_DIR_SAI = Path("./data")
TB_DIR = Path("tblogs/")
LOG_DIR = Path("results/logs/")

DATASET = "dataset"
MODEL_NAME = "model_name"
MODEL_TYPE = "model_type"
SYNTHETIC = "synthetic"
SHAPENET_SAMPLE = "shapenet_sample"
SHAPENET = "shapenet"
SHAPENET_SMALL = "shapenet_small"
SHAPENET_NOISE_FULL = "shapenet_full_with_Gausian_Noise"
SHAPENET_NOISE_SMALL = "shapenet_small_with_Gausian_Noise"
GOOGLETTS = "googeltts"


TRNDATA_SUBSET = "training_data_subset"
FULL_DATA = "full_data"
IDEAL_SUB = "ideal_subset"

BUDGET = "budget"
BADEX_PERITER = "num_badex"
NUMR_PERITER = "num_R_per_iter"

LOGREG = "LR"
RESNET = "resnet"
MOBNET_V1 = "mobilenet_v1"
MOBNET_V2 = "mobilenet_v2"

INIT_THETA = "initialize_theta_from_scratch"
LOAD_THETA = "load_theta"

TRAIN_ARGS = "train_args"
OURM_ARGS = "our_method_args"
OURM_TYPE = "our_method_type"

SEQUENTIAL = "our_method"

INTERLEAVE = "interleave"
INTERLEAVE_TYPE = "interleave_epochs_or_sgditers"
INTERLEAVE_ITERS = "interleave_sgd_iters"
INTERLEAVE_EPOCHS = "interleave _epochs"

THETA = "th"
GREEDY = "greedy"
TUNE_THETA = "fine_tune_greedy_theta"
PHI = "phi"
PRETRN = "pre_training"
PRETRN_THPHI = "pretrn_th_phi"
OUR_METHOD = "our_method"
BETA_PRIOR = "prior_on_beta_using_theta"
FILTER_GRPS = "filter_groups_at_idx"
RANKING_PRIOR = "ranking_prior_on_beta_using_theta"

LRN_RATTE = "lr"
MOMENTUM = "momentum"
OPTIMIZER = "optimizer"
BATCH_NORM = "batch_norm"
SAMPLER = "sampler"
SCRATCH = "scratch"
NNARCH = "nn_arch"

SW = "summarywriter"
BATCH_SIZE = "batch_size"

EXPT_NAME = "experiment_name"

RESNET_TRANSFORMS = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

TRANSFORM = "transform"

SCHEDULER = "scheduler"
SCHEDULER_TYPE = "scheduler_type"

EPOCHS = "epochs"

Y_DICT = {
        0:  "02691156",
        1:  "02828884",
        2:  "02924116",
        3:  "02933112",
        4:  "03001627",
        5:  "03211117",
        6:  "03624134",
        7:  "03636649",
        8:  "03691459", 
        9:  "04090263", 
}

NUM_DICT = {
        0 : 250,
        1 : 70,
        2 : 80  
}

DIST_DICT = {0 : 0.5,
            1 : 1.5,
            2 : 4,
            }
    
SHAPENET_FULL_NOISE_TRN_PATH = "./data/shapenet-large.pkl"
SHAPENET_SMALL_NOISE_TRN_PATH = "./data/shapenet-small.pkl"
SHAPENET_NOISE_TEST_PATH = "./data/shapenet-test.pkl"

AUDIO_TRN_PATH = "./data/training_audio_data_4_60.pkl"
AUDIO_TEST_PATH = "./data/testing_audio_data_60.pkl"

MIN_POOL = 'minpool'