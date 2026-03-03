# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
from torch_utils import persistence
from DiffAugment_pytorch import DiffAugment


#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

def shifted(img) :
    min_shift = 10
    max_shift = 25
    shift = min_shift + (max_shift - min_shift) * torch.rand(1)
    k = shift.long().item()
    new_data = img.clone()
    new_data[:,:,:k,:k] = img[:,:,-k:,-k:]
    new_data[:,:,:k,k:] = img[:,:,-k:,:-k]
    new_data[:,:,k:,:k] = img[:,:,:-k,-k:]
    return new_data

def get_perm(l) :
    perm = torch.randperm(l)
    while torch.all(torch.eq(perm, torch.arange(l))) :
        perm = torch.randperm(l)
    return perm

def jigsaw_k(data, k = 2) :
    with torch.no_grad() :
        actual_h = data.size()[2]
        actual_w = data.size()[3]
        h = torch.split(data, int(actual_h/k), dim = 2)
        splits = []
        for i in range(k) :
            splits += torch.split(h[i], int(actual_w/k), dim = 3)
        fake_samples = torch.stack(splits, -1)
        for idx in range(fake_samples.size()[0]) :
            perm = get_perm(k*k)
            # fake_samples[idx] = fake_samples[idx,:,:,:,torch.randperm(k*k)]
            fake_samples[idx] = fake_samples[idx,:,:,:,perm]
        fake_samples = torch.split(fake_samples, 1, dim=4)
        merged = []
        for i in range(k) :
            merged += [torch.cat(fake_samples[i*k:(i+1)*k], 2)]
        fake_samples = torch.squeeze(torch.cat(merged, 3), -1)
        return fake_samples

def stitch(data, k = 2) :
    #  = torch.randperm()
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    actual_h = data.size()[2]
    actual_w = data.size()[3]
    if torch.randint(0, 2, (1,))[0].item() == 0 :
        dim0, dim1 = 2,3
    else :
        dim0, dim1 = 3,2

    h = torch.split(data, int(actual_h/k), dim = dim0)
    h_1 = torch.split(data_perm, int(actual_h/k), dim = dim0)
    splits = []
    for i in range(k) :
        if i < int(k/2) :
            splits += torch.split(h[i], int(actual_w/k), dim = dim1)
        else :
            splits += torch.split(h_1[i], int(actual_w/k), dim = dim1)
    merged = []
    for i in range(k) :
        merged += [torch.cat(splits[i*k:(i+1)*k], dim1)]
    fake_samples = torch.cat(merged, dim0)

    return fake_samples

def mixup(data, alpha = 25.0) :
    lamb = np.random.beta(alpha, alpha)
    # indices = torch.randperm(data.size(0))
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    return data*lamb + (1-lamb)*data_perm

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutout(data) :
    min_k, max_k = 10, 20
    data = data.clone()
    h, w = data.size(2), data.size(3)
    b_size = data.size(0)
    for i in range(b_size) :
        k = (min_k + (max_k - min_k) * torch.rand(1)).long().item()
        h_pos = ((h - k) * torch.rand(1)).long().item()
        w_pos = ((w - k) * torch.rand(1)).long().item()
        patch = data[i,:,h_pos:h_pos+k,w_pos:w_pos+k]
        patch_mean = torch.mean(patch, axis = (1,2), keepdim = True)
        data[i,:,h_pos:h_pos+k,w_pos:w_pos+k] = torch.ones_like(patch) * patch_mean

    return data

def cut_mix(data, beta = 1.0) :
    data = data.clone()
    lam = np.random.beta(beta, beta)
    indices = get_perm(data.size(0))
    data_perm = data[indices]
    bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
    data[:, :, bbx1:bbx2, bby1:bby2] = data_perm[:, :, bbx1:bbx2, bby1:bby2]
    return data

def rotate(data, angle = 60) :
    batch_size = data.size(0)
    new_data = []
    for i in range(batch_size) :
        pil_img = transforms.ToPILImage()(data[i].cpu())
        img_rotated = transforms.functional.rotate(pil_img, angle)
        new_data.append(transforms.ToTensor()(img_rotated))
    new_data = torch.stack(new_data, 0)
    return new_data.cuda()

@persistence.persistent_class
class Patch_EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
    def adaptive_fixed_augmentation(self, images, p):
        device = images.device
        batch_size = images.shape[0]
        pseudo_flag = torch.ones([batch_size, 1, 1, 1], device=device)
        pseudo_flag = torch.where(torch.rand([batch_size, 1, 1, 1], device=device) < p,
                                  pseudo_flag, torch.zeros_like(pseudo_flag))
        if torch.allclose(pseudo_flag, torch.zeros_like(pseudo_flag)):
            return DiffAugment(images,policy='translation')
        else:            
            return images
    def patchify_with_coords(self, images, patch_size, i=None, j=None, apply_aug=True):
        device = images.device
        batch_size, h, w = images.size(0), images.size(2), images.size(3)

        if apply_aug:
            images = self.adaptive_fixed_augmentation(images, p=0.1)

        th, tw = patch_size, patch_size
        if i is None or j is None:
            if w == tw and h == th:
                i = torch.zeros((batch_size,), device=device).long()
                j = torch.zeros((batch_size,), device=device).long()
            else:
                i = torch.randint(0, h - th + 1, (batch_size,), device=device)
                j = torch.randint(0, w - tw + 1, (batch_size,), device=device)

        rows = torch.arange(th, dtype=torch.long, device=device) + i[:, None]
        cols = torch.arange(tw, dtype=torch.long, device=device) + j[:, None]
        images_t = images.permute(1, 0, 2, 3)
        patch = images_t[:, torch.arange(batch_size)[:, None, None], rows[:, torch.arange(th)[:, None]], cols[:, None]]
        patch = patch.permute(1, 0, 2, 3)
        return patch, i, j

    def latent_pos_channels(self, patch_size, resolution, i, j):
        device = i.device
        batch_size = i.shape[0]
        tw = th = patch_size
        x_pos = torch.arange(tw, dtype=torch.long, device=device).unsqueeze(0).repeat(th, 1).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        y_pos = torch.arange(th, dtype=torch.long, device=device).unsqueeze(1).repeat(1, tw).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
        x_pos = x_pos + j.view(-1, 1, 1, 1)
        y_pos = y_pos + i.view(-1, 1, 1, 1)
        x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
        y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
        return torch.cat((x_pos, y_pos), dim=1)

    def __call__(self, net, images, patch_size, resolution, labels=None, augment_pipe=None,
                 img_vae=None, latent_scale_factor=None, target_pixels=None, sigma_switch=0.5):
        images, i, j = self.patchify_with_coords(images, patch_size, apply_aug=True)
        images_pos = self.latent_pos_channels(patch_size, resolution, i, j)
        NDA_images = jigsaw_k(images, k=2)

        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        y1 = NDA_images

        # Hybrid latent/pixel training with explicit sigma-based switch.
        if target_pixels is not None:
            if img_vae is None or latent_scale_factor is None:
                raise ValueError('target_pixels requires img_vae and latent_scale_factor')

            # High-noise samples: latent-space denoising.
            latent_noise = torch.randn_like(y) * sigma

            # Low-noise samples: pixel-space noising -> encode -> latent denoising.
            pixel_patch_size = patch_size * 8
            target_patch, _, _ = self.patchify_with_coords(target_pixels, pixel_patch_size, i=i * 8, j=j * 8, apply_aug=False)
            pixel_noise = torch.randn_like(target_patch) * sigma
            y_pixel_noisy = target_patch + pixel_noise
            y_pixel_noisy_latent = img_vae.encode(y_pixel_noisy)['latent_dist'].sample() * latent_scale_factor

            sigma_mask = (sigma > sigma_switch).to(y.dtype)
            yn = sigma_mask * (y + latent_noise) + (1 - sigma_mask) * y_pixel_noisy_latent

            D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)
            pred_pixels = img_vae.decode(D_yn / latent_scale_factor).sample

            latent_loss = weight * ((D_yn - y) ** 2) + 1 / weight * 1 / (((D_yn - y1) ** 2) + 100)
            pixel_loss = (pred_pixels - target_patch) ** 2
            loss = sigma_mask * latent_loss + (1 - sigma_mask) * pixel_loss
        else:
            n = torch.randn_like(y) * sigma
            yn = y + n
            D_yn = net(yn, sigma, x_pos=images_pos, class_labels=labels, augment_labels=augment_labels)
            loss = weight * ((D_yn - y) ** 2) + 1 / weight * 1 / (((D_yn - y1) ** 2) + 100)
        return loss

#----------------------------------------------------------------------------

