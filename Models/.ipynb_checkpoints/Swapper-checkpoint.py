import torch
import torch.nn.functional as F
import pdb

import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch import nn
from Models.Swapper_Units import Encoder, Decoder, Discriminator, Decoder_Enlarge, Encoder_noBNIN, VGGPerceptualLoss, Decoder_Enlarge_for_onnx
from Models.arcface_resnet import resnet50, resnet_face18
# from Model.MultiScaleDiscriminator import MultiscaleDiscriminator
from Models.iresnet import iresnet100
from backbones import get_model
import numpy as np

# from Model.loss import GANLoss, AEI_Loss
batch_size = 1


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        x = x - self.mean
        x = x / self.std
        return x

class ResNet_ID_encoder(pl.LightningModule):
    def __init__(self, ema_path):
        super(ResNet_ID_encoder, self).__init__()
        self.resnet = resnet50()
        #pdb.set_trace()

        self.ema = nn.Linear(512, 512)
        self.ema.weight.data = torch.nn.Parameter(torch.from_numpy(np.load(ema_path)).permute(1,0)).cuda()
        self.ema.bias.data.fill_(0.0) 
        #self.ema = torch.from_numpy(np.load(ema_path)).cuda()

    def processing_ema_only(self,latent):
        x = self.ema(latent)
        # x = torch.matmul(x,self.ema)
        out = torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))
        return out

    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.ema(x)
        #x = torch.matmul(x,self.ema)
        out = torch.div(x, torch.linalg.norm(x, dim=1, keepdim=True))
        return out

class Swapper_Enlarge(nn.Module): #class Swapper_Enlarge(pl.LightningModule):
    def __init__(self, source_dim,exist_BN=True):
        super(Swapper_Enlarge, self).__init__()
        self.source_dim = source_dim
        if exist_BN:
            self.E = Encoder(self.source_dim)
        else:
            self.E = Encoder_noBNIN(self.source_dim)
        self.G = Decoder_Enlarge(1024, 3)
        self.dis = None
        self.train_adv = None


    def forward(self, target, source,get_latent=False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        if get_latent==True:
            output,latent = self.G(output,get_latent=get_latent)
            return output,latent 
        else:
            output = self.G(output)
        return output

    def forward(self, target, source,get_latent=False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        if get_latent==True:
            output,latent = self.G(output,get_latent=get_latent)
            return output,latent 
        else:
            output = self.G(output)
        return output
    

class Swapper_Enlarge_for_onnx(nn.Module): #class Swapper_Enlarge(pl.LightningModule):
    def __init__(self, source_dim,exist_BN=True):
        super(Swapper_Enlarge_for_onnx, self).__init__()
        self.source_dim = source_dim
        if exist_BN:
            self.E = Encoder(self.source_dim)
        else:
            self.E = Encoder_noBNIN(self.source_dim)
        self.G = Decoder_Enlarge_for_onnx(1024, 3)
        self.dis = None
        self.train_adv = None


    def forward(self, target, source):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        output = self.G(output)
        return output

    def forward(self, target, source):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        output = self.G(output)
        return output

class Swapper(nn.Module): #class Swapper(pl.LightningModule):
    def __init__(self, source_dim,exist_BN=True):
        super(Swapper, self).__init__()
        self.source_dim = source_dim
        if exist_BN:
            self.E = Encoder(self.source_dim)
        else:
            self.E = Encoder_noBNIN(self.source_dim)
        self.G = Decoder(1024, 3)
        self.dis = None
        self.train_adv = None


    def forward(self, target, source,get_latent=False):
        # source = np.dot(source, self.ema)
        # source /= np.linalg.norm(source)
        output = self.E(target, source)
        if get_latent==True:
            output,latent = self.G(output,get_latent=get_latent)
            return output,latent 
        else:
            output = self.G(output)
        return output

class Doppelganger(pl.LightningModule):
    def __init__(self, Swapper, id_encoder,fine_tune=False):
        super(Doppelganger, self).__init__()
        self.Swapper = Swapper.cuda()
        #self.Id_encoder = None
        self.fine_tune = fine_tune
        self.Id_encoder = id_encoder.cuda()

        self.dis = None
        self.train_adv = False
        
        self.feats_extractor = None

    def Prepareing_adversrial_learning(self,img_size,max_conv_size):
        self.dis = Discriminator(img_size=img_size, max_conv_dim=max_conv_size).cuda()
        self.train_adv = True
        
    def Preparing_VGG_percet_loss(self):
        self.feats_extractor = VGGPerceptualLoss(layer_ids=[3, 8, 15, 22]).cuda()
        

    def set_grads(self):
        swapper_param_list = list(self.Swapper.parameters())

        for param in swapper_param_list:
            param.requires_grad = True

        
        idc_param_list = list(self.Id_encoder.parameters())
        for param in idc_param_list:
            param.requires_grad = False

        # if self.fine_tune==False:
        #     idc_param_list = list(self.Id_encoder.parameters())
        #     for param in idc_param_list:
        #         param.requires_grad = False

        if self.train_adv==True:
            adv_param_list = list(self.dis.parameters())
            for param in adv_param_list:
                param.requires_grad = True

    def save(self, fname, step):
        print('Saving checkpoint into %s...' % fname)
        PATH_Swapper = fname + '_swapper_%d' % (step) + '.pt'
        PATH_IDC = fname + '_idc_%d' % (step) + '.pt'
        PATH_DIS = fname + '_dis_%d' % (step) + '.pt'

        PATH_Swapper_last = fname + '_swapper_last' + '.pt'
        PATH_IDC_last = fname + '_idc_last' + '.pt'
        PATH_DIS_last = fname + '_dis_last' + '.pt'

        # Save the mask module
        torch.save(self.Swapper.state_dict(), PATH_Swapper)
        torch.save(self.Swapper.state_dict(), PATH_Swapper_last)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC)
        torch.save(self.ID_encoder.state_dict(), PATH_IDC_last)

        # Save discriminator
        if self.train_adv==True:
            torch.save(self.discriminator.state_dict(), PATH_DIS)
            torch.save(self.discriminator.state_dict(), PATH_DIS_last)

        # Save the mask module


    def load(self, fname, step=None):
        print('Loading checkpoint from %s...' % fname)
        if step == None:
            PATH_Swapper = fname + '_swapper_last' + '.pt'
            PATH_IDC = fname + '_idc_last' + '.pt'
            PATH_DIS = fname + '_dis_last' + '.pt'
        else:
            PATH_Swapper = fname + '_swapper_%d' % (step) + '.pt'
            PATH_IDC = fname + '_idc_%d' % (step) + '.pt'
            PATH_DIS = fname + '_dis_%d' % (step) + '.pt'

        # Save the mask module
        self.Swapper.load_state_dict(torch.load(PATH_Swapper))
        # Save discriminator
        self.dis.load_state_dict(torch.load(PATH_DIS))
        if self.fine_tune==False:
            self.Id_encoder.load_state_dict(torch.load(PATH_IDC))

    def get_id_code(self,source):
        return self.Id_encoder(source).detach()


    def forward(self, target, source,get_latent=False):
        #if it's the fine-tunning process, only identity code (512D) will be provided, so, no need to use Id_encoder
        if self.fine_tune==False:
            source = self.Id_encoder(source)
        #Else
        output = self.Swapper(target, source,get_latent=get_latent)
        return output


    def add_batch_instant_norm2swapper(self):
        bn_layer = nn.BatchNorm2d(1024)  # BatchNorm after fusion
        in_layer = nn.InstanceNorm2d(1024, affine=True)  # InstanceNorm after fusion
        #name, module = self.Swapper.E.Encoder['layer_3'].items()
        self.Swapper.E.Encoder['layer_3']  = nn.Sequential(
                self.Swapper.E.Encoder['layer_3'] ,  # Original Feature_Fusion_Block
                bn_layer,  # BatchNorm
                in_layer,  # InstanceNorm
            )
        #pdb.set_trace()
        for i in range(5):  # Since we need norm layers between 6 fusion layers, we only need 5 sets of norms
            out_channels = 1024  # Assuming fusion layer output size is 2048
            # Directly add BatchNorm and InstanceNorm as attributes to the model
            setattr(self.Swapper.E, f'batch_norm_{i}', nn.BatchNorm2d(out_channels))
            setattr(self.Swapper.E, f'instance_norm_{i}', nn.InstanceNorm2d(out_channels, affine=True))

    # def enlarge_384(self):
    #     #pdb.set_trace()
    #     name_list = []
    #     for name, module in self.Swapper.G.named_children():
    #         print(name, module)
    #         name_list.append(name)
    #
    #     #print(self.Swapper.G)
    #     with torch.no_grad():
    #
    #         # For Conv1
    #         print(self.Swapper.G)
    #         gen_conv1_old_weight = self.Swapper.G.Conv1.weight.data
    #         _shape = np.shape(gen_conv1_old_weight)[0:2]
    #         #pdb.set_trace()
    #         new_conv = nn.Conv2d(in_channels=_shape[1], out_channels=_shape[0]*2, kernel_size=self.Swapper.G.Conv1.kernel_size[0], padding=self.Swapper.G.Conv1.padding[0])
    #         new_weight = new_conv.weight.data
    #         new_weight[:_shape[0], :, :, :] = gen_conv1_old_weight
    #         new_weight[_shape[0]:, :, :, :] = 0.0
    #         if self.Swapper.G.Conv1.bias is not None and new_conv.bias is not None:
    #             new_conv.bias.data[:_shape[0]] =  self.Swapper.G.Conv1.bias.data
    #             new_conv.bias.data[_shape[0]:] = 0.0
    #         self.Swapper.G.Conv1 = new_conv
    #
    #         #Conv2
    #         gen_conv2_old_weight = self.Swapper.G.Conv2.weight.data
    #         _shape = np.shape(gen_conv2_old_weight)[0:2]
    #         #pdb.set_trace()
    #         new_conv = nn.Conv2d(in_channels=_shape[1]*2, out_channels=_shape[0] * 2,
    #                              kernel_size=self.Swapper.G.Conv2.kernel_size[0],
    #                              padding=self.Swapper.G.Conv2.padding[0])
    #         new_weight = new_conv.weight.data
    #         new_weight[:_shape[0], :_shape[1], :, :] = gen_conv2_old_weight
    #         new_weight[_shape[0]:, _shape[1]:, :, :] = 0.0
    #         if self.Swapper.G.Conv2.bias is not None and new_conv.bias is not None:
    #             new_conv.bias.data[:_shape[0]] = self.Swapper.G.Conv2.bias.data
    #             new_conv.bias.data[_shape[0]:] = 0.0
    #         self.Swapper.G.Conv2 = new_conv
    #
    #         #Conv3
    #         gen_conv3_old_weight = self.Swapper.G.Conv3.weight.data
    #         _shape = np.shape(gen_conv3_old_weight)[0:2]
    #         #pdb.set_trace()
    #         new_conv = nn.Conv2d(in_channels=_shape[1]*2, out_channels=_shape[0] * 2,
    #                              kernel_size=self.Swapper.G.Conv2.kernel_size[0],
    #                              padding=self.Swapper.G.Conv2.padding[0])
    #         new_weight = new_conv.weight.data
    #         new_weight[:_shape[0], :_shape[1], :, :] = gen_conv3_old_weight
    #         new_weight[_shape[0]:, _shape[1]:, :, :] = 0.0
    #         if self.Swapper.G.Conv3.bias is not None and new_conv.bias is not None:
    #             new_conv.bias.data[:_shape[0]] = self.Swapper.G.Conv3.bias.data
    #             new_conv.bias.data[_shape[0]:] = 0.0
    #         self.Swapper.G.Conv3 = new_conv
    #
    #         #pdb.set_trace()
    #         #print(self.Swapper.G)
    #         layers = list(self.Swapper.G.children())
    #
    #         # Conv4
    #         gen_conv4_old_weight = self.Swapper.G.Conv4.weight.data
    #         _shape = np.shape(gen_conv4_old_weight)[0:2]
    #         #pdb.set_trace()
    #
    #         new_conv1 = nn.Conv2d(in_channels=_shape[1]*2, out_channels=_shape[1],
    #                              kernel_size=3,
    #                              padding=1)
    #         new_conv2 = nn.Conv2d(in_channels=_shape[1], out_channels=_shape[0],
    #                               kernel_size=self.Swapper.G.Conv4.kernel_size[0],
    #                               padding=self.Swapper.G.Conv4.padding[0])
    #
    #         layers.insert(4, new_conv1)
    #         new_weight = new_conv2.weight.data
    #         new_weight[:_shape[0], :_shape[1], :, :] = gen_conv4_old_weight
    #         new_weight[_shape[0]:, _shape[1]:, :, :] = 0.0
    #
    #         self.Swapper.G.Conv4 = new_conv2
    #         #pdb.set_trace()
    #         self.Swapper.G = nn.Sequential(*layers)
    #         print(self.Swapper.G)
    #         self.Swapperc
    #         .G = self.Swapper.G.cuda()



def build_arch(fine_tune=False,exist_BN=True,enlarge=False,new_id_model=False):
    #Swapper
    if enlarge==True:
        swapper = Swapper_Enlarge_for_onnx(512)
    else:   
        swapper = Swapper(512,exist_BN) 
    swapper = swapper.cuda()
    #pdb.set_trace()

    # swapper_enlarged = Swapper_Enlarge(512)
    # swapper_enlarged.E.load_state_dict(torch.load('./Models/MMNet.pt'), strict=False)
    # swapper_enlarged= swapper_enlarged.cuda()

    #ID encoder
    #pdb.set_trace()
    if new_id_model==False:
        ID_encoder = ResNet_ID_encoder(ema_path='./Models/emp.npy')
        ID_encoder.resnet.load_state_dict(torch.load('./Models/arcface_w600k_r50_pytorch.pt', map_location=torch.device('cuda')))
    else:
        weight = torch.load('./vit_b_fr_pgair.pt')
        ID_encoder = get_model('vit_b', dropout=0, fp16=False).cuda()
        ID_encoder.load_state_dict(weight)
        
    #Framework
    DPG = Doppelganger(swapper, ID_encoder, fine_tune=fine_tune)
    return DPG




def build_models(config=None,fine_tune=False,adv_train=True,exist_BN=True,enlarge=False,new_id_model=False,from_scretch=False):
    #Swapper
    if enlarge==True:
        swapper = Swapper_Enlarge(512)
        if from_scretch==False:
            print('Loading pre-trained model for the swapper')
            swapper.load_state_dict(torch.load('./Models/MMNet.pt',weights_only=True), strict=False)
    else:   
        swapper = Swapper(512,exist_BN) 
        if from_scretch==False:
            print('Loading pre-trained model for the swapper')
            swapper.load_state_dict(torch.load('./Models/MMNet.pt'), strict=False)
    #pdb.set_trace()
    
    swapper = swapper.cuda()
    #pdb.set_trace()

    # swapper_enlarged = Swapper_Enlarge(512)
    # swapper_enlarged.E.load_state_dict(torch.load('./Models/MMNet.pt'), strict=False)
    # swapper_enlarged= swapper_enlarged.cuda()

    #ID encoder
    #pdb.set_trace()
    print('Loading pre-trained model for the ID encoder')
    if new_id_model==False:
        ID_encoder = ResNet_ID_encoder(ema_path='./Models/emp.npy')
        ID_encoder.resnet.load_state_dict(torch.load('./Models/arcface_w600k_r50_pytorch.pt', map_location=torch.device('cuda')))
    else:
        weight = torch.load(config.id_network_path)
        ID_encoder = get_model(config.id_network, dropout=0, fp16=False).cuda()
        ID_encoder.load_state_dict(weight)
        
    #Framework
    DPG = Doppelganger(swapper, ID_encoder, fine_tune=fine_tune)
    if adv_train==True:
        DPG.Prepareing_adversrial_learning(img_size=256,max_conv_size=512)
        DPG.Preparing_VGG_percet_loss()
    return DPG

if __name__ == '__main__':
    build_models()

