# START: Original code by Zijian and Xinran
import sys
sys.path.append(".")
from test_net import *

cfg=Config('collective')
cfg.training_stage=3
cfg.test_seqs=[9, 64]

cfg.train_backbone=False
cfg.test_before_train=True
cfg.image_size=480, 720
cfg.out_size=57,87
cfg.num_boxes=13
cfg.num_actions=8
cfg.num_activities=7
cfg.num_frames=10
cfg.num_graph=4
cfg.tau_sqrt=True

cfg.batch_size=16
cfg.test_batch_size=1
cfg.train_learning_rate=1e-4
cfg.train_dropout_prob=0.2
cfg.weight_decay=1e-2
cfg.lr_plan={}
cfg.max_epoch=50

cfg.exp_note='Collective_test_' + cfg.backbone
test_net(cfg)
# END: Original code by Zijian and Xinran