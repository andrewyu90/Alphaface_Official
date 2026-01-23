import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from pytorch_msssim import ssim

# Assume the loss functions are already defined as in the previous code snippet:
# identity_loss, reconstruction_loss, generator_adversarial_loss, discriminator_adversarial_loss,
# gradient_penalty, weak_feature_matching_loss, PerceptualLoss

# For brevity, we reinclude them here (or import them from a module):
def identity_loss(embedding_generated: torch.Tensor, embedding_source: torch.Tensor) -> torch.Tensor:
    #embedding_generated = F.normalize(embedding_generated, p=2, dim=1)
    #embedding_source = F.normalize(embedding_source, p=2, dim=1)
    #cosine_sim = torch.sum(embedding_generated * embedding_source, dim=1)
    # pdb.set_trace()
    loss = (1 - F.cosine_similarity(embedding_source, embedding_generated, dim=1))
    return loss.mean()


def clip_text_loss(embedding_generated: torch.Tensor, embedding_source: torch.Tensor, clip_score: torch.Tensor) -> torch.Tensor:
    #embedding_generated = F.normalize(embedding_generated, p=2, dim=1)
    #embedding_source = F.normalize(embedding_source, p=2, dim=1)
    #cosine_sim = torch.sum(embedding_generated * embedding_source, dim=1)
    
    cos_sim =  F.cosine_similarity(embedding_source, embedding_generated, dim=1)
    #pdb.set_trace()
    loss = (clip_score-cos_sim)
    
    
    loss = torch.where(0 < loss, loss, 0.0)
    return loss.sum()


def identity_score(embedding_generated: torch.Tensor, embedding_source: torch.Tensor) -> torch.Tensor:
    return F.cosine_similarity(embedding_source, embedding_generated, dim=1)

def reconstruction_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(generated, target)

def masked_reconstruction_loss(generated: torch.Tensor, target: torch.Tensor,target_mask: torch.Tensor) -> torch.Tensor:
    #pdb.set_trace()
    return F.l1_loss(target_mask*generated, target_mask*target)

def structural_similarity_loss(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    ssim_loss = 1 - ssim(generated, target, data_range=1.0)
    return ssim_loss

def structural_similarity_score(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #ssim_loss = 1 - ssim(generated, target, data_range=1.0)
    return ssim(generated, target, data_range=1.0)

def multi_scale_adversarial_loss(discriminator_output, is_real=True):
    """
    Compute adversarial loss for the generator or discriminator using BCE with logits.

    Args:
        discriminator_output (torch.Tensor): Raw logits from the discriminator for a batch of images.
        is_real (bool): If True, we label this batch as real (1); if False, label as fake (0).

    Returns:
        torch.Tensor: Scalar adversarial loss value.
    """
    if is_real:
        # Target is 1 for real images (or when training the generator to fool the discriminator)
        target = torch.ones_like(discriminator_output)
    else:
        # Target is 0 for fake images (when training the discriminator)
        target = torch.zeros_like(discriminator_output)

    # Use binary cross-entropy with logits (assumes disc outputs are raw logits)
    loss = F.mse_loss(discriminator_output, target)
    return loss

def adversarial_loss(discriminator_output, is_real=True):
    """
    Compute adversarial loss for the generator or discriminator using BCE with logits.

    Args:
        discriminator_output (torch.Tensor): Raw logits from the discriminator for a batch of images.
        is_real (bool): If True, we label this batch as real (1); if False, label as fake (0).

    Returns:
        torch.Tensor: Scalar adversarial loss value.
    """
    if is_real:
        # Target is 1 for real images (or when training the generator to fool the discriminator)
        target = torch.ones_like(discriminator_output)
    else:
        # Target is 0 for fake images (when training the discriminator)
        target = torch.zeros_like(discriminator_output)

    # Use binary cross-entropy with logits (assumes disc outputs are raw logits)
    loss = F.binary_cross_entropy_with_logits(discriminator_output, target)
    return loss


def hinge_discriminator_loss(D_real, D_fake):
    """
    Hinge loss for discriminator.
    D_real: Discriminator output for real images.
    D_fake: Discriminator output for fake images.
    """
    loss_real = torch.mean(F.relu(1.0 - D_real))
    loss_fake = torch.mean(F.relu(1.0 + D_fake))
    return loss_real + loss_fake

def hinge_generator_loss(D_fake):
    """
    Hinge loss for generator.
    D_fake: Discriminator output for fake images.
    """
    return -torch.mean(D_fake)


def gradient_penalty(discriminator, real_images: torch.Tensor, fake_images: torch.Tensor, device: torch.device,
                     lambda_gp: float = 10.0) -> torch.Tensor:
    batch_size = real_images.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolated = epsilon * real_images + (1 - epsilon) * fake_images
    interpolated.requires_grad_(True)

    d_interpolated = discriminator(interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    return penalty


def attribute_preservation_loss(features_generated: list, features_target: list) -> torch.Tensor:
    loss = 0.0
    for feat_gen, feat_tgt in zip(features_generated, features_target):
        loss += F.l1_loss(feat_gen, feat_tgt, reduction='mean')
    return lossembedding_source

def masked_attribute_preservation_loss(features_generated: list, features_target: list,mask: torch.Tensor) -> torch.Tensor:
    loss = 0.0
    for feat_gen, feat_tgt in zip(features_generated, features_target):
        _shape = feat_gen.shape
        reshaped_mask = F.interpolate(mask,(_shape[2],_shape[3]),mode='bilinear', align_corners=True)
        loss += F.l1_loss(reshaped_mask[:,0:1,:,:]*feat_gen, reshaped_mask[:,0:1,:,:]*feat_tgt, reduction='mean')
    return loss


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super(PerceptualLoss, self).__init__()
        self.feature_extractor = feature_extractor
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def forward(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        features_gen = self.feature_extractor(generated)
        features_tgt = self.feature_extractor(target)
        return F.l1_loss(features_gen, features_tgt)

