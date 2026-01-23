import argparse
import pdb

import matplotlib.pyplot as plt
import torch
from Models.Swapper_AlphaFace import build_AlphaFace
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
import os
from dataset.get_dataloader import get_dataloader_clip
import torchvision.transforms as transforms
from PIL import Image
from Objectives.Loss import *
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import clip 


def normalize_by_127_5(img):
    """
    Normalize a tensor image by dividing by its maximum value.

    Args:
        img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor with values in [0, 1].
    """
    img = (img * 255.0).int()
    return (img / 127.5) - 1.0  # Return unchanged if max is 0 (e.g., all-zero image)


# Return unchanged if max is 0 (e.g., all-zero image)


def tensor2img(tensor):
    """
    Normalize a tensor image by dividing by its maximum value.

    Args:
        img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: Normalized image tensor with values in [0, 1].
    """
    return (tensor * 255.0).int()  # Return unchanged if max is 0 (e.g., all-zero image)



def train_with_clip(config, fs_model,clip_model, opts, dataloader):
    start_epoch = 0
    global_step = 0
    writer = SummaryWriter(config.tb_dir)
    swapper_opt = opts[0]
    dis_opt = opts[1]

    #scheduler_swapper = CosineAnnealingLR(swapper_opt, T_max=50, eta_min=1e-6)  # Decays to 1e-6 over 50 epochs
    scheduler_swapper = StepLR(swapper_opt, step_size=config.lr_schedule_step, gamma=0.97)  # Reduce discriminator LR every 10 epochs  original step_size = 100k
    scheduler_discriminator = StepLR(dis_opt, step_size=config.lr_schedule_step, gamma=0.97)  # Reduce discriminator LR every 10 epochs original step_size = 100k
    #
    if not os.path.exists(config.output):
        os.makedirs(config.output)
        print(f"Directory '{config.output}' created.")
    else:
        print(f"Directory '{config.output}' already exists.")



    normalize_transform = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))
    
    if config.resume:
        print('Resuming from checkpoint...')
        if config.model_path==None:
            print('from '+os.path.join(config.output, f"model_last.pt"))
            dict_checkpoint = torch.load(os.path.join(config.output, f"model_last.pt"))
            start_epoch = dict_checkpoint["epoch"]
            global_step = dict_checkpoint["global_step"]
            fs_model.Swapper.load_state_dict(dict_checkpoint["swapper"])
            fs_model.dis.load_state_dict(dict_checkpoint["discriminator"])
            swapper_opt.load_state_dict(dict_checkpoint["swapper_optimizer"])
            dis_opt.load_state_dict(dict_checkpoint["discriminator_optimizer"])
            del dict_checkpoint
        else:
            dict_checkpoint = torch.load(config.model_path)
            print('from '+config.model_path)
            start_epoch = dict_checkpoint["epoch"]
            global_step = dict_checkpoint["global_step"]
            fs_model.Swapper.load_state_dict(dict_checkpoint["swapper"])
            fs_model.dis.load_state_dict(dict_checkpoint["discriminator"])
            swapper_opt.load_state_dict(dict_checkpoint["swapper_optimizer"])
            dis_opt.load_state_dict(dict_checkpoint["discriminator_optimizer"])
            del dict_checkpoint

    # Get source identity code for specific face identity swapping
    fs_model = fs_model.cuda()
    fs_model.Swapper.train()
    fs_model.Id_encoder.eval()
    fs_model.set_grads()

    clip_model.eval()

    # pdb.set_trace()
    for epoch in range(start_epoch, config.num_epoch):
        # print('epoch %d'%(epoch))
        for _, target_samples in enumerate(dataloader):
            
            img1_t, img2_t, mask1_t,mask2_t, txt1_t,txt2_t  = target_samples
            img1_t = img1_t.cuda()
            img1_s = ((F.interpolate(img1_t, (112, 112), mode='bilinear') * 255.0) / 127.5) - 1.0
            img2_t = img2_t.cuda()
            img2_s = ((F.interpolate(img2_t, (112, 112), mode='bilinear') * 255.0) / 127.5) - 1.0


            mask1_t = mask1_t.cuda()
            mask2_t = mask2_t.cuda()


            # print('global step %d'%(global_step))
            #swapped_1_2, latent_1_2 = fs_model(img1_t, img2_s,get_latent=True)
            #swapped_2_1, latent_2_1 = fs_model(img2_t, img1_s,get_latent=True)
            swapped_1_2 = fs_model(img1_t, img2_s)
            swapped_2_1 = fs_model(img2_t, img1_s)
            
            swapped112_1_2 = ((F.interpolate(swapped_1_2, (112, 112), mode='bilinear') * 255.0) / 127.5) - 1.0
            swapped112_2_1 = ((F.interpolate(swapped_2_1, (112, 112), mode='bilinear') * 255.0) / 127.5) - 1.0

            # For clip loss====================================================================================================
            
            tokenized1 = clip.tokenize(txt1_t, context_length=77, truncate=True).cuda()
            tokenized2 = clip.tokenize(txt2_t, context_length=77, truncate=True).cuda()
            #pdb.set_trace()

            img1_t_clip = (F.interpolate(img1_t, (224, 224), mode='bilinear') * 255.0) 
            img2_t_clip = (F.interpolate(img2_t, (224, 224), mode='bilinear') * 255.0)
            img1_t_clip  = normalize_transform(img1_t_clip)
            img2_t_clip  = normalize_transform(img2_t_clip)


            swapped_1_2_clip = (F.interpolate(swapped_1_2, (224, 224), mode='bilinear') * 255.0) 
            swapped_2_1_clip = (F.interpolate(swapped_2_1, (224, 224), mode='bilinear') * 255.0)
            swapped_1_2_clip  = normalize_transform(swapped_1_2_clip)
            swapped_2_1_clip  = normalize_transform(swapped_2_1_clip)
            
            
            #img1_features = clip_model.encode_image(img1_t_clip)   # [1, D]
            #img2_features = clip_model.encode_image(img2_t_clip)   # [1, D]

            img1_2_features = clip_model.encode_image(swapped_1_2_clip)   # [1, D]
            img2_1_features = clip_model.encode_image(swapped_2_1_clip)   # [1, D]
            
            
            
            
            with torch.no_grad():                
                identity_code_1 = fs_model.Id_encoder(img1_s)
                identity_code_2 = fs_model.Id_encoder(img2_s)

                img1_features = clip_model.encode_image(img1_t_clip)   # [1, D]
                img2_features = clip_model.encode_image(img2_t_clip)   # [1, D]

                text1_features  = clip_model.encode_text(tokenized1)  # [1, D]
                text2_features  = clip_model.encode_text(tokenized2)  # [1, D]

            swapped_code_1_2 = fs_model.Id_encoder(swapped112_1_2)
            swapped_code_2_1 = fs_model.Id_encoder(swapped112_2_1)

            # Compute identity loss
            loss_id_1 = identity_loss(swapped_code_2_1, identity_code_1)
            loss_id_2 = identity_loss(swapped_code_1_2, identity_code_2)
            loss_id = loss_id_1+loss_id_2
            # print(loss_id.grad_fn)
            
            
            # t2t s2s reconstruction loss
            swapped_face_1_1 = fs_model(img1_t, img1_s)
            swapped_face_2_2 = fs_model(img2_t, img2_s)
            loss_rec_1_1 = reconstruction_loss(img1_t, swapped_face_1_1)
            loss_rec_2_2 = reconstruction_loss(img2_t, swapped_face_2_2)
            loss_self_rec = loss_rec_1_1 + loss_rec_2_2
            
            
            
            #Weakly featrure matching loss (Perceptual loss) - VGG16 used
            loss_percept_1_2 = fs_model.feats_extractor(img1_t,swapped_1_2)
            loss_percept_2_1 = fs_model.feats_extractor(img2_t,swapped_2_1)
            loss_percept= loss_percept_1_2 + loss_percept_2_1
            
            
            # Second-round cyclic reconstruction loss (t > s > t   vs t)
            swapped_face_1_2_1 = fs_model(swapped_1_2, img1_s)
            swapped_face_2_1_2 = fs_model(swapped_2_1, img2_s)
            loss_2cycle_1_2_1 = reconstruction_loss(img1_t, swapped_face_1_2_1)
            loss_2cycle_2_1_2 = reconstruction_loss(img2_t, swapped_face_2_1_2)
            loss_2cycle_rec = loss_2cycle_1_2_1 + loss_2cycle_2_1_2

            #masked reconstruction loss
            loss_masked_recon_1_2 = masked_reconstruction_loss(img1_t, swapped_1_2,mask1_t)
            loss_masked_recon_2_1 = masked_reconstruction_loss(img2_t, swapped_2_1,mask2_t)
            loss_masked_recon = loss_masked_recon_1_2 + loss_masked_recon_2_1
            
            
            # Loss for clip loss================================================================================================================
            clip_t_loss_id_1 = identity_loss(img2_1_features, img1_features)
            clip_t_loss_id_2 = identity_loss(img1_2_features, img2_features)
            clip_t_loss_id = clip_t_loss_id_1+clip_t_loss_id_2

            clip_score_1 = identity_score(img1_features,text1_features)
            clip_score_2 = identity_score(img2_features,text2_features)

            clip_cl_text2img_id_1 = clip_text_loss(img2_1_features, text2_features, clip_score_2)
            clip_cl_text2img_id_2 = clip_text_loss(img1_2_features, text1_features, clip_score_1)   
            clip_text2img_loss = clip_cl_text2img_id_1+clip_cl_text2img_id_2
            #pdb.set_trace()

            

            
            #pdb.set_trace()
            if global_step <= config.adv_sess:
                total_gen_loss = (config.w_id*loss_id
                                  +config.w_self_rec*loss_self_rec
                                  +config.w_percept*loss_percept
                                  +config.w_2cycle*loss_2cycle_rec
                                  +config.w_mask_rec*loss_masked_recon
                                  +config.w_clip_id*clip_t_loss_id
                                  +config.w_clip_text*clip_text2img_loss
                                  )
                swapper_opt.zero_grad()
                total_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.Swapper.parameters(), max_norm=1.0)  # Add clipping
                swapper_opt.step()
                scheduler_swapper.step()
                
            else:
                # Compute adversarial loss for generator
                disc_gen_output_1_2 = fs_model.dis(swapped_1_2)
                disc_gen_output_2_1 = fs_model.dis(swapped_2_1)
                loss_adv_gen_1_2 = multi_scale_adversarial_loss(disc_gen_output_1_2, is_real=True)
                loss_adv_gen_2_1 = multi_scale_adversarial_loss(disc_gen_output_2_1, is_real=True)
                loss_adv_gen = loss_adv_gen_1_2+loss_adv_gen_2_1
                
                total_gen_loss = (config.w_id*loss_id
                                  +config.w_self_rec*loss_self_rec
                                  +config.w_percept*loss_percept
                                  +config.w_2cycle*loss_2cycle_rec
                                  +config.w_mask_rec*loss_masked_recon
                                  +config.w_clip_id*clip_t_loss_id
                                  +config.w_clip_text*clip_text2img_loss
                                  +config.w_gen_adv*loss_adv_gen
                                  )
                
                swapper_opt.zero_grad()
                total_gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.Swapper.parameters(), max_norm=1.0)  # Add clipping
                swapper_opt.step()

                # Update for discriminator
                # Real images
                disc_real_output_1 = fs_model.dis(img1_t)
                loss_disc_real_1 = multi_scale_adversarial_loss(disc_real_output_1, is_real=True)
                disc_real_output_2 = fs_model.dis(img2_t)
                loss_disc_real_2 = multi_scale_adversarial_loss(disc_real_output_2, is_real=True)
                loss_disc_real = loss_disc_real_1+loss_disc_real_2

                # Fake images (detach to prevent backprop through generator again)
                disc_fake_output_1_2 = fs_model.dis(swapped_1_2.detach())
                loss_disc_fake_1_2 = multi_scale_adversarial_loss(disc_fake_output_1_2, is_real=False)
                disc_fake_output_2_1 = fs_model.dis(swapped_2_1.detach())
                loss_disc_fake_2_1 = multi_scale_adversarial_loss(disc_fake_output_2_1, is_real=False)
                loss_disc_fake = loss_disc_fake_1_2+loss_disc_fake_2_1

                # Total discriminator loss
                total_disc_loss = (loss_disc_real + loss_disc_fake) / 2

                dis_opt.zero_grad()
                total_disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(fs_model.dis.parameters(), max_norm=1.0)  # Add clipping
                dis_opt.step()

                scheduler_swapper.step()
                scheduler_discriminator.step()
            
            

            
            if global_step % 100 == 0:
                dis_lr = dis_opt.param_groups[-1]['lr']
                gen_lr = swapper_opt.param_groups[-1]['lr']
                if global_step <= config.adv_sess:
                    print(
                        f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: Not applied")
                else:
                    #pdb.set_trace()
                    print(
                    f"global step {global_step}, Generator Loss: {total_gen_loss.item()}, Discriminator Loss: {total_disc_loss.item()}")
                writer.add_scalar("0.Total_loss_for_swapper/train", total_gen_loss, global_step)
                writer.add_scalar("1.Identity_loss/train", loss_id, global_step)
                writer.add_scalar("2.Self_recon_loss/train", loss_self_rec, global_step)
                writer.add_scalar("3.Perceptual_loss/train", loss_percept, global_step)
                writer.add_scalar("4.2Cyclic_recon_loss/train", loss_2cycle_rec, global_step)
                writer.add_scalar("5.Masked_recon_loss/train", loss_masked_recon, global_step)
                writer.add_scalar("8.[Clip-learning] CLIP_ID Similarity loss/train", clip_t_loss_id, global_step)
                writer.add_scalar("9.[Clip-learning] CLIP_img-text_contrastive_learning_loss/train", clip_text2img_loss, global_step)

                writer.add_scalar("[-].Learning_rate_swapper/train", gen_lr, global_step)
                if global_step > config.adv_sess:
                    writer.add_scalar("8.Adv_gen_Loss/train", loss_adv_gen, global_step)
                    writer.add_scalar("9.Adv_disc_Loss/train", total_disc_loss, global_step)
                    writer.add_scalar("[-].Discriminator_learning_rate/train", dis_lr, global_step)
                    
                if config.visualize:
                    target_image = img1_t
                    grid = make_grid(target_image)
                    writer.add_image('1_Target_attribute_image', grid, global_step)
                    
                    target_mask = mask1_t
                    grid = make_grid(target_mask)
                    writer.add_image('2_Target_mask', grid, global_step)

                    source_image = img2_t
                    grid = make_grid(source_image)
                    writer.add_image('3_Source_identity_image', grid, global_step)
                    
                    source_mask = mask2_t
                    grid = make_grid(source_mask)
                    writer.add_image('4_source_mask', grid, global_step)


                    swapped_face = swapped_1_2
                    grid = make_grid(swapped_face)
                    writer.add_image('5_Swapped_results', grid, global_step)
                    
            
            if global_step % config.save_interval == 0:
                # Save the model checkpoint
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "swapper_lr": gen_lr,
                    "dis_lr": dis_lr,
                    "swapper": fs_model.Swapper.state_dict(),
                    "discriminator": fs_model.dis.state_dict(),
                    "swapper_optimizer": swapper_opt.state_dict(),
                    "discriminator_optimizer": dis_opt.state_dict()
                }, os.path.join(config.output, "model_%d.pt"%(global_step)))
                
                # Save the model checkpoint
                torch.save({
                    "epoch": epoch,
                    "global_step": global_step,
                    "swapper_lr": gen_lr,
                    "dis_lr": dis_lr,
                    "swapper": fs_model.Swapper.state_dict(),
                    "discriminator": fs_model.dis.state_dict(),
                    "swapper_optimizer": swapper_opt.state_dict(),
                    "discriminator_optimizer": dis_opt.state_dict()
                }, os.path.join(config.output, "model_last.pt"))
            
            global_step += 1
            
                
        # Save generated images periodically for visual inspection


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    os.makedirs(cfg.log_dir, exist_ok=True)

    # Get dataloader for train
    print(cfg.num_workers)
    train_loader = get_dataloader_clip(
        cfg.db_path,
        cfg.batch_size,
        cfg.num_workers
    )

    
    # Model initialisation
    print('Preparing the AlphaFace model')
    AlphaFace = build_AlphaFace(config=cfg).to('cuda')
    #DPG_s.add_batch_instant_norm2swapper()
    AlphaFace = AlphaFace.cuda()


    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)

    # Optimiser initialisation
    if cfg.optimizer == "sgd":
        swapper_opt = torch.optim.SGD(params=[{"params": AlphaFace.Swapper.parameters()}], lr=cfg.init_lr_swapper, momentum=0.9,
                                      weight_decay=cfg.weight_decay)
        dis_opt = torch.optim.SGD(params=[{"params": AlphaFace.dis.parameters()}], lr=cfg.init_lr_dis, momentum=0.9,
                                  weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        swapper_opt = torch.optim.AdamW(params=[{"params": AlphaFace.Swapper.parameters()}], lr=cfg.init_lr_swapper,betas=(0.0, 0.99),
                                        weight_decay=cfg.weight_decay)
        dis_opt = torch.optim.AdamW(params=[{"params": AlphaFace.dis.parameters()}], lr=cfg.init_lr_dis,betas=(0.0, 0.99),
                                    weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adam":
        swapper_opt = torch.optim.Adam(params=[{"params": AlphaFace.Swapper.parameters()}], lr=cfg.init_lr_swapper,betas=(0.0, 0.99),
                                        weight_decay=cfg.weight_decay)
        dis_opt = torch.optim.Adam(params=[{"params": AlphaFace.dis.parameters()}], lr=cfg.init_lr_dis,betas=(0.0, 0.99),
                                    weight_decay=cfg.weight_decay)
    else:
        raise
    opts = [swapper_opt, dis_opt]
    train_with_clip(cfg, AlphaFace,clip_model, opts, train_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, default='./configs/test_config', help="py config file")
    main(parser.parse_args())






