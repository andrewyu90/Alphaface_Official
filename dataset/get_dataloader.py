
import pdb
import queue as Queue
import threading
import torchvision.transforms as T
from typing import Iterable
import cv2
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
from glob import glob
import os
from PIL import Image
import random
import numpy as np

import random

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


def get_img_list(path):
    files = []
    for ext in ('*.gif', '*.png', '*.jpg'):
        files.extend(glob(os.path.join(path, ext)))
    return files

def synchronized_horizontal_flip_manual(image1, image2):
    """Applies synchronized horizontal flip using manual control."""
    if random.random() > 0.5:
        image1 = T.functional.hflip(image1)
        image2 = T.functional.hflip(image2)
    return image1, image2

def load_text_from_file(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return line
    raise ValueError(f"No non-empty line found in {txt_path}")



class FaceImageDataset_ImageOnly(Dataset):
    def __init__(self, db_path,t_transform=None,s_transform=None):
        #pdb.set_trace()
        self.img_list = get_img_list(os.path.join(db_path,'img'))
        np.random.shuffle(self.img_list)
        self.t_transform =t_transform
        self.s_transform =s_transform
        self.num_sample = len(self.img_list)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        #print('index %d'%(idx))
        src_idx = random.randint(0, self.num_sample-1)
        tar_idx = random.randint(0, self.num_sample-1)
        
        while src_idx==tar_idx:
            tar_idx = random.randint(0, self.num_sample-1)

        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]

        src_rgb_img = Image.open(src_img_path).convert('RGB')
        tar_rgb_img = Image.open(tar_img_path).convert('RGB')
        
        
        
        
        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)

        return img1_t, img2_t,



class FaceImageDataset_CLIP(Dataset):
    def __init__(self, db_path,t_transform=None,s_transform=None):
        #pdb.set_trace()
        self.img_list = get_img_list(os.path.join(db_path,'img'))
        self.mask_path = os.path.join(db_path,'mask')
        self.text_path = os.path.join(db_path,'txt')
        np.random.shuffle(self.img_list)
        self.t_transform =t_transform
        self.s_transform =s_transform
        self.num_sample = len(self.img_list)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        #print('index %d'%(idx))
        src_idx = random.randint(0, self.num_sample-1)
        tar_idx = random.randint(0, self.num_sample-1)
        
        while src_idx==tar_idx:
            tar_idx = random.randint(0, self.num_sample-1)

        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]

        src_rgb_img = Image.open(src_img_path).convert('RGB')
        tar_rgb_img = Image.open(tar_img_path).convert('RGB')
        

        # Need to parse the paths which are suitable for mask file paths.
        src_mask_path = os.path.join(self.mask_path,src_img_path.split('/')[-1])
        tar_mask_path = os.path.join(self.mask_path,tar_img_path.split('/')[-1])

        
        src_txt_path = os.path.join(self.text_path,src_img_path.split('/')[-1].split('.')[0]+'.txt')
        tar_txt_path = os.path.join(self.text_path,tar_img_path.split('/')[-1].split('.')[0]+'.txt')

        src_text_str = load_text_from_file(src_txt_path)
        tar_text_str = load_text_from_file(tar_txt_path)


        src_msk_img = Image.open(src_mask_path).convert('RGB')
        tar_msk_img = Image.open(tar_mask_path).convert('RGB')
        
        src_rgb_img, src_msk_img =  synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        tar_rgb_img, tar_msk_img=  synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)

        
        
        
        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
            mask1_t = self.t_transform(src_msk_img)
            mask2_t =  self.t_transform(tar_msk_img)
        #print('=================')
        #print(src_img_path)
        #print(tar_img_path)
        return img1_t, img2_t,1-mask1_t,1-mask2_t, src_text_str, tar_text_str


class FaceImageDataset(Dataset):
    def __init__(self, db_path,t_transform=None,s_transform=None):
        #pdb.set_trace()
        self.img_list = get_img_list(os.path.join(db_path,'img'))
        self.mask_path = os.path.join(db_path,'mask')
        np.random.shuffle(self.img_list)
        self.t_transform =t_transform
        self.s_transform =s_transform
        self.num_sample = len(self.img_list)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        #print('index %d'%(idx))
        src_idx = random.randint(0, self.num_sample-1)
        tar_idx = random.randint(0, self.num_sample-1)
        
        while src_idx==tar_idx:
            tar_idx = random.randint(0, self.num_sample-1)

        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]
        #print(src_img_path)
        #print(tar_img_path)

        #Change file titles to mask
        


        # #bgr_img = cv2.imread(img_path)
        # #rgb_img= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # #Title parsing
        # src_id = src_img_path.split('_')[0]
        # tar_id = tar_img_path.split('_')[0]
        # print(src_id)

        # if src_id!='nan' and tar_id!='nan':
        #     if src_id==tar_id:
        #         #pdb.set_trace()
        #         while src_id!=tar_id:
        #             tar_idx = random.randint(0, self.num_sample - 1)
        #             tar_img_path = self.img_list[tar_idx]
        #             tar_id = tar_img_path.split('_')[0]
        # else:
        #     if src_id==tar_id:
        #         #pdb.set_trace()
        #         while src_id!=tar_id:
        #             tar_idx = random.randint(0, self.num_sample - 1)
        #             tar_img_path = self.img_list[tar_idx]
        #             tar_id = tar_img_path.split('_')[0]

        src_rgb_img = Image.open(src_img_path).convert('RGB')
        tar_rgb_img = Image.open(tar_img_path).convert('RGB')
        

        # Need to parse the paths which are suitable for mask file paths.
        src_mask_path = os.path.join(self.mask_path,src_img_path.split('/')[-1])
        tar_mask_path = os.path.join(self.mask_path,tar_img_path.split('/')[-1])

        src_msk_img = Image.open(src_mask_path).convert('RGB')
        tar_msk_img = Image.open(tar_mask_path).convert('RGB')
        
        src_rgb_img, src_msk_img =  synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        
        tar_rgb_img, tar_msk_img=  synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)
        
        
        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
            mask1_t = self.t_transform(src_msk_img)
            mask2_t =  self.t_transform(tar_msk_img)
        #print('=================')
        #print(src_img_path)
        #print(tar_img_path)
        return img1_t, img2_t,1-mask1_t,1-mask2_t



class FaceImageDataset_SRC_TAR(Dataset):
    def __init__(self, db_path,t_transform=None,s_transform=None):
        #pdb.set_trace()
        self.img_list = get_img_list(os.path.join(db_path,'img'))
        self.mask_path = os.path.join(db_path,'mask')
        np.random.shuffle(self.img_list)
        self.t_transform =t_transform
        self.s_transform =s_transform
        self.num_sample = len(self.img_list)

    def __len__(self):
        return self.num_sample

    def __getitem__(self, idx):
        #print('index %d'%(idx))
        src_idx = random.randint(0, self.num_sample-1)
        tar_idx = random.randint(0, self.num_sample-1)
        
        while src_idx==tar_idx:
            tar_idx = random.randint(0, self.num_sample-1)

        src_img_path = self.img_list[src_idx]
        tar_img_path = self.img_list[tar_idx]
        #print(src_img_path)
        #print(tar_img_path)

        #Change file titles to mask
        


        # #bgr_img = cv2.imread(img_path)
        # #rgb_img= cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # #Title parsing
        # src_id = src_img_path.split('_')[0]
        # tar_id = tar_img_path.split('_')[0]
        # print(src_id)

        # if src_id!='nan' and tar_id!='nan':
        #     if src_id==tar_id:
        #         #pdb.set_trace()
        #         while src_id!=tar_id:
        #             tar_idx = random.randint(0, self.num_sample - 1)
        #             tar_img_path = self.img_list[tar_idx]
        #             tar_id = tar_img_path.split('_')[0]
        # else:
        #     if src_id==tar_id:
        #         #pdb.set_trace()
        #         while src_id!=tar_id:
        #             tar_idx = random.randint(0, self.num_sample - 1)
        #             tar_img_path = self.img_list[tar_idx]
        #             tar_id = tar_img_path.split('_')[0]

        src_rgb_img = Image.open(src_img_path).convert('RGB')
        tar_rgb_img = Image.open(tar_img_path).convert('RGB')
        

        # Need to parse the paths which are suitable for mask file paths.
        src_mask_path = os.path.join(self.mask_path,src_img_path.split('/')[-1])
        tar_mask_path = os.path.join(self.mask_path,tar_img_path.split('/')[-1])

        src_msk_img = Image.open(src_mask_path).convert('RGB')
        tar_msk_img = Image.open(tar_mask_path).convert('RGB')
        
        src_rgb_img, src_msk_img =  synchronized_horizontal_flip_manual(src_rgb_img, src_msk_img)
        
        tar_rgb_img, tar_msk_img=  synchronized_horizontal_flip_manual(tar_rgb_img, tar_msk_img)
        
        
        if self.t_transform is not None and self.s_transform is not None:
            img1_t = self.t_transform(src_rgb_img)
            img2_t = self.t_transform(tar_rgb_img)
            mask1_t = self.t_transform(src_msk_img)
            mask2_t =  self.t_transform(tar_msk_img)
        #print('=================')
        #print(src_img_path)
        #print(tar_img_path)
        return img1_t, img2_t,1-mask1_t,1-mask2_t




def get_dataloader(
    db_path,
    batch_size,
    num_workers = 4,
    ) -> Iterable:

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor()
    #     ])
    '''
    t_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    '''

    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    s_transform =transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5)
    ])

    train_set = FaceImageDataset(db_path, t_transform=t_transform,s_transform=s_transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader


def get_dataloader_img_only(
    db_path,
    batch_size,
    num_workers = 4,
    ) -> Iterable:

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor()
    #     ])
    '''
    t_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    '''

    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    s_transform =transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5)
    ])

    train_set = FaceImageDataset_ImageOnly(db_path, t_transform=t_transform,s_transform=s_transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader




def get_dataloader_clip(
    db_path,
    batch_size,
    num_workers = 4,
    ) -> Iterable:

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor()
    #     ])
    '''
    t_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    '''

    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    s_transform =transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5)
    ])

    train_set = FaceImageDataset_CLIP(db_path, t_transform=t_transform,s_transform=s_transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader





def get_dataloader_fixed_src_tar(
    db_path,
    batch_size,
    num_workers = 4,
    ) -> Iterable:

    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.Resize((256,256)),
    #     transforms.ToTensor()
    #     ])
    '''
    t_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        ])
    '''

    t_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    s_transform =transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Lambda(normalize_by_127_5)
    ])

    train_set = FaceImageDataset(db_path, t_transform=t_transform,s_transform=s_transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size = batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader





def get_dataloader_tmp(
    db_path,
    local_rank,
    batch_size,
    seed = 2048,
    num_workers = 2,
    ) -> Iterable:

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128,128)),
        transforms.ToTensor()
        ])

    train_set = FaceImageDataset(db_path, transform)

    rank, world_size = get_dist_info()
    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    if seed is None:
        init_fn = None
    else:
        init_fn = partial(worker_init_fn, num_workers=num_workers, rank=rank, seed=seed)

    train_loader = DataLoaderX(
        local_rank=local_rank,
        dataset=train_set,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=init_fn,
    )

    return train_loader

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        #pdb.set_trace()
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        #pdb.set_trace()
        #print('iter %d'%(self.iter))
        self.batch,_ = next(self.iter, None)
        #print(len(self.batch))
        #print(_)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

