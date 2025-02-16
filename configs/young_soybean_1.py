# DATA
dataset='CULane'
data_root = '/home/mrsd_teamh/sush/11-785/project_train_data_1'


# TRAIN
epoch = 50
batch_size = 32
optimizer = 'SGD'  #['SGD','Adam']
learning_rate = 0.1
weight_decay = 1e-4
momentum = 0.9

scheduler = 'multi' #['multi', 'cos']
steps = [25,38]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 695

# NETWORK
use_aux = True
griding_num = 200
backbone = '18'

# LOSS
sim_loss_w = 0.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "/home/mrsd_teamh/sush/11-785/ufld_logging"

# FINETUNE or RESUME MODEL PATH
# finetune = None
finetune = "/home/mrsd_teamh/sush/11-785/Ultra-Fast-Lane-Detection/checkpoints/culane_18.pth"
resume = None

# TEST
test_model = "/home/mrsd_teamh/sush/11-785/Ultra-Fast-Lane-Detection/checkpoints/checkpoint.pth"
test_work_dir = None

num_lanes = 4
