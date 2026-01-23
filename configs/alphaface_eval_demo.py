from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.network = "test1"
config.embedding_size = 512
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 4
config.init_lr_swapper = 1e-4
config.init_lr_dis = 1e-4
config.verbose = 2000
config.num_epoch = 100000
config.warmup_epoch = 0
config.savedir = 'Save2'
config.src_img_size = 112
config.tar_img_size = 256
config.visualize = True
config.from_scretch = False

config.optimizer = "adamw"
config.db_path = './dataset/ffhq_lpff/256_small'
config.tensorboard = False
config.tb_dir = './logs/swap1'
config.adv_sess = 150000
config.lr_schedule_step = 20000
config.exp_try_num = 100

#Config for Id encoder
config.id_network_path = './vit_b_fr_pgair.pt'
config.id_network = 'vit_b'


#Config for saving
config.model_path = './alphaface_demo.pt'
config.output = './output'

config.src_path = './dataset/source'
config.tar_path = './dataset/target'