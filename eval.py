import argparse
import pdb

import matplotlib.pyplot as plt
import torch
from Models.Swapper_AlphaFace import build_AlphaFace
from utils.utils_config import get_config
from utils.utils_distributed_sampler import setup_seed
import os
from dataset.get_dataloader import get_dataloader
import torchvision.transforms as transforms
from PIL import Image
from Objectives.Loss import *
from torchvision.utils import make_grid
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Iterable, List, Union


def list_images(
    directory: Union[str, Path],
    extensions: Union[Iterable[str], None] = None,
    recursive: bool = False
) -> List[Path]:
    """
    Return a sorted list of image files found in *directory*.

    Parameters
    ----------
    directory : str | pathlib.Path
        Folder to search.
    extensions : Iterable[str] | None, default None
        File-name extensions to keep (case-insensitive, with or without the dot).
        If None, a sensible default set is used.
    recursive : bool, default False
        If True, search all nested sub-directories; otherwise, look only at the top
        level of *directory*.

    Returns
    -------
    list[pathlib.Path]
        Absolute, sorted paths to the matching image files.
    """
    # Default, widely-used image extensions
    default_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff",
                    ".gif", ".webp", ".ppm", ".pgm"}
    exts = default_exts if extensions is None else {
        (e.lower() if e.startswith(".") else f".{e.lower()}") for e in extensions
    }

    directory = Path(directory).expanduser().resolve()

    if recursive:
        paths = (p for p in directory.rglob("*") if p.is_file())
    else:
        paths = (p for p in directory.iterdir() if p.is_file())

    return sorted(p for p in paths if p.suffix.lower() in exts)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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


def eval_alphaface(config, model):
    writer = SummaryWriter(config.tb_dir)

    if not os.path.exists(config.output):
        os.makedirs(config.output)
        print(f"Directory '{config.output}' created.")
    else:
        print(f"Directory '{config.output}' already exists.")
   
    print('Resuming from checkpoint...from %s'%(config.model_path))
    dict_checkpoint = torch.load(config.model_path)
    print('from '+config.model_path)
    model.Swapper.load_state_dict(dict_checkpoint["swapper"])

    # Get source identity code for specific face identity swapping
    model = model.cuda()
    print(f"Number of parameters in Swapper: {count_parameters(model.Swapper)}")
    model.Swapper.eval()
    model.Id_encoder.eval()


    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    s_transform =transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5)
    ])

    src_img_list = list_images(config.src_path)
    tar_img_list = list_images(config.tar_path)
    #pdb.set_trace()
    #pdb.set_trace()
    for src_img_file in src_img_list:
        img1_s = s_transform(Image.open(src_img_file)).unsqueeze(0)
        img1_s = img1_s.cuda()
        
        for tar_img_file in tar_img_list:
        
            img2_t = t_transform(Image.open(tar_img_file)).unsqueeze(0)
            img2_t = img2_t.cuda()
            
            swapped_2_1 = model(img2_t, img1_s)
                

            # If the tensor is batched, take the first item
            if swapped_2_1.dim() == 4:
                swapped_2_1 = swapped_2_1[0]
        
            # If the tensor is in [C, H, W] format, convert to [H, W, C]
            if swapped_2_1.shape[0] in {1, 3, 4}:
                swapped_2_1 = swapped_2_1.permute(1, 2, 0)
        
            # Clamp to valid range and convert to uint8
            if swapped_2_1.max() <= 1.0:
                swapped_2_1 = swapped_2_1 * 255.0
        
            swapped_2_1 = swapped_2_1.clamp(0, 255).byte()
        
            # Convert to numpy and PIL
            np_image = swapped_2_1.cpu().numpy()
            img = Image.fromarray(np_image)
        
            # If single channel, convert to 'L' mode explicitly
            if np_image.shape[2] == 1:
                img = img.convert("L")
            #pdb.set_trace()
            save_img_path = str(src_img_file).split('/')[-1].split('.')[0]+'_'+str(tar_img_file).split('/')[-1]
        
            img.save(os.path.join(config.output,save_img_path))
            print(f"Saved image to {os.path.join(config.output,save_img_path)}")
            
            #print('source: %s | target: %s > %s'%(src_img_file,tar_img_list,os.path.join(config.output,save_img_path)))
    
     


def main(args):
    cfg = get_config(args.config)
    setup_seed(seed=cfg.seed, cuda_deterministic=False)

    if cfg.tensorboard == True:
        writer = SummaryWriter(cfg.tb_path)
    else:
        writer = None
    os.makedirs(cfg.log_dir, exist_ok=True)

    # Get dataloader for train

    
    # Model initialisation
    print('Preparing the student model')
    alphaface = build_AlphaFace(config=cfg).to('cuda')


    eval_alphaface(cfg, alphaface)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Distributed Arcface Training in Pytorch")
    parser.add_argument("config", type=str, default='./configs/test_config', help="py config file")
    main(parser.parse_args())






