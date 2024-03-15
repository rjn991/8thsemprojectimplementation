import sys
sys.path.append(".")
from train_net import *

cfg=Config('collective')
cfg.training_stage=2
cfg.train_backbone=False

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13

# START: Original code by Zijian and Xinran
cfg.num_actions=8
cfg.num_activities=7
# END: Original code by Zijian and Xinran

cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=16
cfg.test_batch_size=8 
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}

# START: Original code by Zijian and Xinran
cfg.max_epoch=100
cfg.exp_note='Collective_train_' + cfg.backbone
# END: Original code by Zijian and Xinran

train_net(cfg)