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
config.savedir = 'Save4'
config.src_img_size = 112
config.tar_img_size = 256
config.visualize = True
config.from_scretch = False

config.optimizer = "adamw"
config.db_path = './dataset/ffhq_lpff/256'
config.tensorboard = False
config.tb_dir = './logs/swap7'
config.adv_sess = 200000
config.lr_schedule_step = 20000

#Config for Id encoder
config.new_id_model = False
config.id_network_path = './vit_b_fr_pgair.pt'
config.id_network = 'vit_b'


#Config for saving
config.resume = False
config.model_path = None
config.output = './Save/save7'
config.save_interval = 25000


#Config for balancing weights
config.w_id = 5.0
config.w_self_rec = 5.0
config.w_percept = 1.0
config.w_2cycle = 1.0
config.w_mask_rec = 1.0
config.w_t_ssim = 1.0
config.w_t_id = 5.0
config.w_gen_adv = 1.0
config.gp = 10.0