import sys
sys.path.append(".")
from train_net import *

cfg=Config('collective')
cfg.training_stage=1
cfg.train_backbone=True

cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13

cfg.num_actions=8
cfg.num_activities=7

cfg.num_frames=10

cfg.batch_size=16
cfg.test_batch_size=8 
cfg.train_learning_rate=1e-5
cfg.train_dropout_prob=0.5
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=100

cfg.exp_note='Collective_train_' + cfg.backbone

train_net(cfg)
