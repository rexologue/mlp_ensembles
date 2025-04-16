from easydict import EasyDict
from utils.enums import WeightsInitType, LayerType, SamplerType, TransformType

#######################
# MODEL CONFIGURATION #
#######################

model_cfg = EasyDict()

# Layers configuration
model_cfg.layers = [
    {'type': LayerType.Linear, 'params': {'in_features': 140, 'out_features': 64}},
    {'type': LayerType.LeakyReLU, 'params': {'alpha': 0.2}},
    {'type': LayerType.Dropout, 'params': {'p': 0.35}},

    {'type': LayerType.Linear, 'params': {'in_features': 64, 'out_features': 7}},
]

# Weights and bias initialization
model_cfg.params = EasyDict()
model_cfg.params.zero_bias = True
model_cfg.params.init_type = WeightsInitType.he

######################
# DATA CONFIGURATION #
######################

data_cfg = EasyDict()

# Label mapping
data_cfg.label_mapping = {
    'angry': 0,
    'disgusted': 1,
    'fearful': 2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprised': 6
}

data_cfg.classes_num = 7

data_cfg.train_transforms = {}
data_cfg.validation_transforms = {}

# Training configuration
data_cfg.sampler_type = SamplerType.Upsampling


############################
# EXPERIMENT CONFIGURATION #
############################

exp_cfg = EasyDict()
exp_cfg.seed = 0
exp_cfg.num_epochs = 120


# Train parameters
exp_cfg.train = EasyDict()
exp_cfg.train.batch_size = 64
exp_cfg.train.weight_decay = 0.01
exp_cfg.train.learning_rate = 0.01


# Overfit parameters
exp_cfg.overfit = EasyDict()
exp_cfg.overfit.num_iterations = 1000


# Neptune parameters
exp_cfg.neptune = EasyDict()
exp_cfg.neptune.project = 'rexologue/MLP'
exp_cfg.neptune.experiment_name = 'MLP'
exp_cfg.neptune.run_id = None # For resuming (Paste from neptune.ai UI) 


# Checkpoints parameters
exp_cfg.checkpoints_dir = '/home/duka/job/mlp/ckps'
exp_cfg.checkpoint_save_frequency = 1
exp_cfg.continue_from_checkpoint = None
exp_cfg.keep_last_n_checkpoints = 10


# Data parameters
exp_cfg.data_cfg = data_cfg


# Model parameters
exp_cfg.model_cfg = model_cfg
