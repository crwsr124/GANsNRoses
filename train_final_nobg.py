import argparse
import math
import random 
import os
from util import *
import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, autograd
from torch import optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist

from torchvision import transforms, utils
from tqdm import tqdm
from torch.optim import lr_scheduler
import copy
import kornia.augmentation as K
import kornia
import lpips

from model_cr import *
from dataset import ImageFolder
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)

mse_criterion = nn.MSELoss()
smooth_l1 = nn.SmoothL1Loss()


@torch.no_grad()
def getGaussianKernel(ksize, sigma=0):
    if sigma <= 0:
        # 根据 kernelsize 计算默认的 sigma，和 opencv 保持一致
        sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8 
    center = ksize // 2
    xs = (np.arange(ksize, dtype=np.float32) - center) # 元素与矩阵中心的横向距离
    kernel1d = np.exp(-(xs ** 2) / (2 * sigma ** 2)) # 计算一维卷积核
    # 根据指数函数性质，利用矩阵乘法快速计算二维卷积核
    kernel = kernel1d[..., None] @ kernel1d[None, ...] 
    kernel = torch.from_numpy(kernel)
    kernel = kernel / kernel.sum() # 归一化
    return kernel

def bilateralFilter(batch_img, ksize, sigmaColor=None, sigmaSpace=None):
    device = batch_img.device
    if sigmaSpace is None:
        sigmaSpace = 0.15 * ksize + 0.35  # 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8
    if sigmaColor is None:
        sigmaColor = sigmaSpace

    pad = (ksize - 1) // 2
    batch_img_pad = F.pad(batch_img, pad=[pad, pad, pad, pad], mode='reflect')

# batch_img 的维度为 BxcxHxW, 因此要沿着第 二、三维度 unfold
# patches.shape:  B x C x H x W x ksize x ksize
    patches = batch_img_pad.unfold(2, ksize, 1).unfold(3, ksize, 1)
    patch_dim = patches.dim() # 6 
# 求出像素亮度差
    diff_color = patches - batch_img.unsqueeze(-1).unsqueeze(-1)
# 根据像素亮度差，计算权重矩阵
    weights_color = torch.exp(-(diff_color ** 2) / (2 * sigmaColor ** 2))
# 归一化权重矩阵
    weights_color = weights_color / weights_color.sum(dim=(-1, -2), keepdim=True)

# 获取 gaussian kernel 并将其复制成和 weight_color 形状相同的 tensor
    weights_space = getGaussianKernel(ksize, sigmaSpace).to(device)
    weights_space_dim = (patch_dim - 2) * (1,) + (ksize, ksize)
    weights_space = weights_space.view(*weights_space_dim).expand_as(weights_color)

# 两个权重矩阵相乘得到总的权重矩阵
    weights = weights_space * weights_color
# 总权重矩阵的归一化参数
    weights_sum = weights.sum(dim=(-1, -2))
# 加权平均
    weighted_pix = (weights * patches).sum(dim=(-1, -2)) / weights_sum
    return weighted_pix


def test(args, genA2B, genB2A, testA_loader, testB_loader, name, step):
    testA_loader = iter(testA_loader)
    testB_loader = iter(testB_loader)
    with torch.no_grad():
        test_sample_num = 16

        genA2B.eval(), genB2A.eval() 
        A2B = []
        B2A = []
        for i in range(test_sample_num):
            real_A = testA_loader.next()
            real_B = testB_loader.next()

            real_A, real_B = real_A.cuda(), real_B.cuda()

            A2B_content, A2B_style = genA2B.encode(real_A)
            B2A_content, B2A_style = genB2A.encode(real_B)

            if i % 2 == 0:
                A2B_mod1 = torch.randn([1, args.latent_dim]).cuda()
                B2A_mod1 = torch.randn([1, args.latent_dim]).cuda()
                A2B_mod2 = torch.randn([1, args.latent_dim]).cuda()
                B2A_mod2 = torch.randn([1, args.latent_dim]).cuda()

            a_c, a_s = G_A2B.encode(real_A)
            fake_A2A, alphaA2A = G_B2A.decode(a_c, a_s)
            b_c, b_s = G_B2A.encode(real_B)
            fake_B2B, alphaB2B = G_A2B.decode(b_c, b_s)
            # fake_B2B, _, _ = genA2B(real_B)
            # fake_A2A, _, _ = genB2A(real_A)

            colsA = [real_A, fake_A2A]
            alphaA2A = alphaA2A.repeat(1, 3, 1, 1)
            colsA.append(alphaA2A)
            colsB = [real_B, fake_B2B]
            alphaB2B = alphaB2B.repeat(1, 3, 1, 1)
            colsB.append(alphaB2B)
            
            fake_A2B_1, alpha = genA2B.decode(A2B_content, A2B_mod1)
            fake_B2A_1, alpha = genB2A.decode(B2A_content, B2A_mod1)

            fake_A2B_2, alpha = genA2B.decode(A2B_content, A2B_mod2)
            fake_B2A_2, alpha = genB2A.decode(B2A_content, B2A_mod2)

            fake_A2B_3, alpha1 = genA2B.decode(A2B_content, B2A_style)
            fake_B2A_3, alpha2 = genB2A.decode(B2A_content, A2B_style)

            fake_A2B_2[:, 0:1, :, :] = alpha1
            fake_A2B_2[:, 1:2, :, :] = alpha1
            fake_A2B_2[:, 2:3, :, :] = alpha1
            fake_B2A_2[:, 0:1, :, :] = alpha2
            fake_B2A_2[:, 1:2, :, :] = alpha2
            fake_B2A_2[:, 2:3, :, :] = alpha2

            colsA += [fake_A2B_3, fake_A2B_1, fake_A2B_2]
            colsB += [fake_B2A_3, fake_B2A_1, fake_B2A_2]

            fake_A2B2A, _,  _, alpha = genB2A(fake_A2B_3, A2B_style)
            fake_B2A2B, _,  _, alpha = genA2B(fake_B2A_3, B2A_style)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            fake_A2B2A, _,  _, alpha = genB2A(fake_A2B_1, A2B_style)
            fake_B2A2B, _,  _, alpha = genA2B(fake_B2A_1, B2A_style)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            # fake_A2B2A, _,  _, alpha = genB2A(fake_A2B_2, A2B_style)
            # fake_B2A2B, _,  _, alpha = genA2B(fake_B2A_2, B2A_style)
            # colsA.append(fake_A2B2A)
            # colsB.append(fake_B2A2B)

            fake_A2B2A, _, _, alpha = genB2A(fake_A2B_1, B2A_mod1)
            fake_B2A2B, _, _, alpha = genA2B(fake_B2A_1, A2B_mod1)
            colsA.append(fake_A2B2A)
            colsB.append(fake_B2A2B)

            colsA = torch.cat(colsA, 2).detach().cpu()
            colsB = torch.cat(colsB, 2).detach().cpu()

            A2B.append(colsA)
            B2A.append(colsB)
        A2B = torch.cat(A2B, 0)
        B2A = torch.cat(B2A, 0)

        utils.save_image(A2B, f'{im_path}/{name}_A2B_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)
        utils.save_image(B2A, f'{im_path}/{name}_B2A_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)

        genA2B.train(), genB2A.train()


def train(args, trainA_loader, trainB_loader, testA_loader, testB_loader, G_A2B, G_B2A, D_A, D_B, G_optim, D_optim, device):
    G_A2B.train(), G_B2A.train(), D_A.train(), D_B.train()
    trainA_loader = sample_data(trainA_loader)
    trainB_loader = sample_data(trainB_loader)
    G_scheduler = lr_scheduler.StepLR(G_optim, step_size=100000, gamma=0.5)
    D_scheduler = lr_scheduler.StepLR(D_optim, step_size=100000, gamma=0.5)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.1)

    loss_dict = {}
    mean_path_length_A2B = 0
    mean_path_length_B2A = 0

    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length = 0

    if args.distributed:
        G_A2B_module = G_A2B.module
        G_B2A_module = G_B2A.module
        D_A_module = D_A.module
        D_B_module = D_B.module
        D_L_module = D_L.module

    else:
        G_A2B_module = G_A2B
        G_B2A_module = G_B2A
        D_A_module = D_A
        D_B_module = D_B
        D_L_module = D_L

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        #G_A2B.train(), G_A2B.encoder.eval(), G_B2A.eval(), D_A.train(), D_B.train()
        #for p_i in G_A2B.encoder.parameters():
        #    p_i.requires_grad=False
        #for p_i in G_B2A.parameters():
        #    p_i.requires_grad=False
        
        ori_A = next(trainA_loader)
        ori_B = next(trainB_loader)
        if isinstance(ori_A, list):
            ori_A = ori_A[0]
        if isinstance(ori_B, list):
            ori_B = ori_B[0]

        ori_A = ori_A.to(device)
        ori_B = ori_B.to(device)
        #aug_A = augA2(ori_A)
        # aug_A_smooth = bilateralFilter(aug_A, 15, 0.15, 5).detach()
        #aug_B = augB2(ori_B)
        aug_A = augA(ori_A)
        aug_B = augB(ori_B)
        # aug_A = DiffAugment(ori_A, policy='color,translation,cutout')
        # aug_B = DiffAugment(ori_B, policy='color,translation,cutout')
        # A = augA(ori_A[[np.random.randint(args.batch)]].expand_as(ori_A))
        # B = augB(ori_B[[np.random.randint(args.batch)]].expand_as(ori_B))
        # A = augA(ori_A)
        # B = augB(ori_B)
        # A = ori_A
        # B = ori_B

        batch_id = np.random.randint(args.batch)
        single_A_batch = ori_A[[batch_id]].expand_as(ori_A)
        single_B_batch = ori_B[[batch_id]].expand_as(ori_B)
        # single_A_batch = ori_A[[batch_id]].expand(ori_A.shape[0]+1, ori_A.shape[1], ori_A.shape[2], ori_A.shape[3])
        # single_B_batch = ori_B[[batch_id]].expand(ori_B.shape[0]+1, ori_B.shape[1], ori_B.shape[2], ori_B.shape[3])
        #A = augA3(single_A_batch)
        A = augA(single_A_batch)
        A[1] = torch.flip(A[0],[2])
        # B = augB3(single_B_batch)
        B = augB(single_B_batch)
        B[1] = torch.flip(B[0],[2])

        # A = augA2(ori_A)
        # B = augB2(ori_B)  

        if i % args.d_reg_every == 0:
            aug_A.requires_grad = True
            aug_B.requires_grad = True
        # if i % args.d_reg_every == 0:
        #     A.requires_grad = True
        #     B.requires_grad = True
        
        A2B_content, A2B_style = G_A2B.encode(A)
        B2A_content, B2A_style = G_B2A.encode(B)

        A_aug_style = G_A2B.style_encode(augA(single_A_batch))
        B_aug_style = G_B2A.style_encode(augB(single_B_batch))

        # get new style
        aug_A2B_style = G_B2A.style_encode(aug_B)
        aug_B2A_style = G_A2B.style_encode(aug_A)
        rand_A2B_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        rand_B2A_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        #print(rand_A2B_style.shape)


        # styles
        idx = torch.randperm(2*args.batch)
        #print(idx)
        #print(rand_A2B_style)
        #print(aug_A2B_style)
        input_A2B_style = torch.cat([rand_A2B_style, aug_A2B_style], 0)[idx][:args.batch]
        #print(A2B_style.shape)
        #print(input_A2B_style)

        idx = torch.randperm(2*args.batch)
        input_B2A_style = torch.cat([rand_B2A_style, aug_B2A_style], 0)[idx][:args.batch]

        A2B_content_withnoise = A2B_content + 0.001*torch.randn(A2B_content.shape).to(device).requires_grad_()
        B2A_content_withnoise = B2A_content + 0.001*torch.randn(B2A_content.shape).to(device).requires_grad_()
        fake_A2B, fake_A2B_alpha = G_A2B.decode(A2B_content_withnoise, input_A2B_style)
        fake_B2A, fake_B2A_alpha = G_B2A.decode(B2A_content_withnoise, input_B2A_style)

        # train disc
        # aug_A_smooth = bilateralFilter(aug_A, 15, 0.15, 5)
        real_A_logit = D_A(aug_A)
        real_B_logit = D_B(aug_B)
        # A_smooth = bilateralFilter(A, 15, 0.15, 5)
        # real_A_logit = D_A(A_smooth)
        # real_B_logit = D_B(B)
        real_L_logit1 = D_L(rand_A2B_style)
        real_L_logit2 = D_L(rand_B2A_style)

        fake_B_logit = D_B(fake_A2B.detach())
        fake_A_logit = D_A(fake_B2A.detach())
        # fake_B_logit = D_B(DiffAugment(fake_A2B.detach(), policy='color,translation,cutout'))
        # fake_A_logit = D_A(DiffAugment(fake_B2A.detach(), policy='color,translation,cutout'))
        fake_L_logit1 = D_L(aug_A2B_style.detach())
        fake_L_logit2 = D_L(aug_B2A_style.detach())

        # global loss
        D_loss = d_logistic_loss(real_A_logit, fake_A_logit) +\
                 d_logistic_loss(real_B_logit, fake_B_logit) +\
                 d_logistic_loss(real_L_logit1, fake_L_logit1) +\
                 d_logistic_loss(real_L_logit2, fake_L_logit2)

        loss_dict['D_adv'] = D_loss

        if i % args.d_reg_every == 0:
            # r1_A_loss = d_r1_loss(real_A_logit, A)
            # r1_B_loss = d_r1_loss(real_B_logit, B)
            r1_A_loss = d_r1_loss(real_A_logit, aug_A)
            r1_B_loss = d_r1_loss(real_B_logit, aug_B)
            r1_L_loss = d_r1_loss(real_L_logit1, rand_A2B_style) + d_r1_loss(real_L_logit2, rand_B2A_style)
            r1_loss = r1_A_loss + r1_B_loss + r1_L_loss
            D_r1_loss = (args.r1 / 2 * r1_loss * args.d_reg_every)
            D_loss += D_r1_loss

        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        #Generator
        # adv loss
        fake_B_logit = D_B(fake_A2B)
        fake_A_logit = D_A(fake_B2A)
        # fake_B_logit = D_B(DiffAugment(fake_A2B, policy='color,translation,cutout'))
        # fake_A_logit = D_A(DiffAugment(fake_B2A, policy='color,translation,cutout'))
        fake_L_logit1 = D_L(aug_A2B_style)
        fake_L_logit2 = D_L(aug_B2A_style)

        lambda_adv = (1, 1, 1)
        G_adv_loss = 1 * (g_nonsaturating_loss(fake_A_logit, lambda_adv) +\
                         g_nonsaturating_loss(fake_B_logit, lambda_adv) +\
                         2*g_nonsaturating_loss(fake_L_logit1, (1,)) +\
                         2*g_nonsaturating_loss(fake_L_logit2, (1,)))

        # style consis loss
        G_con_loss = 50 * (A2B_style.var(0, unbiased=False).sum() + B2A_style.var(0, unbiased=False).sum())
        # G_con_loss = 50 * (cosine_distance(A2B_style).sum() + cosine_distance(B2A_style).sum())
                    
        # cycle recon
        A2B2A_content, A2B2A_style = G_B2A.encode(fake_A2B)
        #print(A2B2A_content.shape)
        B2A2B_content, B2A2B_style = G_A2B.encode(fake_B2A)

        fake_A2B2A, _ = G_B2A.decode(A2B2A_content, shuffle_batch(A_aug_style))
        fake_B2A2B, _ = G_A2B.decode(B2A2B_content, shuffle_batch(B_aug_style))

        # fake_A2B2A, fake_A2B2A_alpha = G_B2A.decode(A2B2A_content, A2B_style)
        # fake_B2A2B, fake_B2A2B_alpha = G_A2B.decode(B2A2B_content, B2A_style)
        # fake_A2B2A, fake_A2B2A_alpha = G_B2A.decode(A2B2A_content, aug_B2A_style)
        # fake_B2A2B, fake_B2A2B_alpha = G_A2B.decode(B2A2B_content, aug_A2B_style)
        # fake_B2AA = G_B2A.decode(B2A_content, B2A2B_style)
        # fake_A2BB = G_A2B.decode(A2B_content, A2B2A_style)
        
        A_smooth = bilateralFilter(A, 15, 0.15, 5)

        # cycle loss
        # A_downsample = F.avg_pool2d(A_smooth, kernel_size=4, stride=4)
        # fake_A2B2A_downsample = F.avg_pool2d(fake_A2B2A, kernel_size=4, stride=4)
        # B_downsample = F.avg_pool2d(B, kernel_size=4, stride=4)
        # fake_B2A2B_downsample = F.avg_pool2d(fake_B2A2B, kernel_size=4, stride=4)
        # l1_cycle_loss = 1. * (F.l1_loss(fake_A2B2A_downsample, A_downsample) +\
        #                     F.l1_loss(fake_B2A2B_downsample, B_downsample))
        #lpips_loss = 1.0 * (lpips_fn(fake_A2B2A_downsample, A_downsample).mean() + lpips_fn(fake_B2A2B_downsample, B_downsample).mean())

        l1_cycle_loss = 20. * (F.l1_loss(fake_A2B2A, A_smooth) +\
                            F.l1_loss(fake_B2A2B, B))
        lpips_loss = 10. * (lpips_fn(fake_A2B2A, A_smooth).mean() + lpips_fn(fake_B2A2B, B).mean())
        cycle_loss = l1_cycle_loss + lpips_loss

        # flip loss
        flip_loss = 5.0 * (F.l1_loss(fake_A2B[0], torch.flip(fake_A2B[1],[2])) + F.l1_loss(fake_B2A[0], torch.flip(fake_B2A[1],[2])))


        # style identity loss
        style_loss_sameimg = 10 * (smooth_l1(A2B_style, A_aug_style) +\
                            smooth_l1(B2A_style, B_aug_style))
        style_loss_identity = 10 * (smooth_l1(A2B2A_style, input_A2B_style) +\
                            smooth_l1(B2A2B_style, input_B2A_style))
        
        G_style_loss = style_loss_identity + style_loss_sameimg
        # G_style_loss = 0

        # feature presering loss
        # kk1 = 1 + i/300000.0 * 3
        cf_loss = 80 * (F.l1_loss(A2B2A_content, A2B_content) +\
                            F.l1_loss(B2A2B_content, B2A_content))

        # identity loss
        b_c_, b_s_ = G_B2A.encode(B)
        b_c_ = b_c_ + 0.001*torch.randn(b_c_.shape).to(device).requires_grad_()
        fake_B2B, _ = G_A2B.decode(b_c_, b_s_)
        a_c_, a_s_ = G_A2B.encode(A)
        a_c_ = a_c_ + 0.001*torch.randn(a_c_.shape).to(device).requires_grad_()
        fake_A2A, _ = G_B2A.decode(a_c_, a_s_)
        # A_smooth = bilateralFilter(A, 15, 0.15, 5)
        # kk1 = 1 + i/300000.0 * 3.0
        ci_loss_f1 = 40 * (F.l1_loss(fake_A2A, A_smooth) +\
                            F.l1_loss(fake_B2B, B))
        ci_loss_lpips = 20 * (lpips_fn(fake_A2A, A_smooth).mean() + lpips_fn(fake_B2B, B).mean())
        ci_loss = ci_loss_f1 + ci_loss_lpips

        G_loss =  G_adv_loss + cf_loss  + G_con_loss + G_style_loss + flip_loss + cycle_loss
        if i % 2 == 0:
            G_loss = G_adv_loss + ci_loss + cf_loss  + G_con_loss + G_style_loss + flip_loss
        # if i % 2 == 0:
        #     kkkk = (i-100000)/200000.
        #     if kkkk > 1:
        #         kkkk = 1
        #     if kkkk < 0:
        #         kkkk = 0
        #     G_loss = G_loss * (0.5+0.5*kkkk)

        loss_dict['G_adv'] = G_adv_loss
        loss_dict['G_con'] = G_con_loss
        loss_dict['ci_loss'] = ci_loss
        loss_dict['cf_loss'] = cf_loss

        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        # g_regularize = i % args.g_reg_every == 0
        # if g_regularize:
        #     # path_batch_size = max(1, args.batch // args.path_batch_shrink)
        #     rand_A2B_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        #     fake_img, _ = G_A2B.decode(A2B_content.detach(), rand_A2B_style)

        #     path_loss, mean_path_length, path_lengths = g_path_regularize(
        #         fake_img, rand_A2B_style, mean_path_length
        #     )

        #     # G_A2B.decoder.zero_grad()
        #     G_A2B.encoder.eval(), G_B2A.eval()
        #     G_optim.zero_grad()
        #     weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

        #     # if args.path_batch_shrink:
        #     #     weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

        #     weighted_path_loss.backward()

        #     G_optim.step()

            # mean_path_length_avg = (
            #     reduce_sum(mean_path_length).item() / get_world_size()
            # )
            # if i % 50 == 0:
            #     print("path_loss: %.8f, mean_path_length_avg: %.8f, path_lengths: %.8f" % \
            #         (path_loss, mean_path_length_avg, path_lengths))

        G_scheduler.step()
        D_scheduler.step()

        # accumulate(G_A2B_ema, G_A2B_module)
        # accumulate(G_B2A_ema, G_B2A_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        D_adv_loss_val = loss_reduced['D_adv'].mean().item()

        G_adv_loss_val = loss_reduced['G_adv'].mean().item()
        #G_cycle_loss_val = loss_reduced['G_cycle'].mean().item()
        G_con_loss_val = loss_reduced['G_con'].mean().item()
        G_cycle_loss_val = 0
        lpips_val = 0
        # lpips_val = loss_reduced['lpips'].mean().item()
        ci_loss_val = 0
        # ci_loss_val = loss_reduced['ci_loss'].mean().item()
        # cf_loss_val = loss_reduced['cf_loss'].mean().item()
        cf_loss_val = cf_loss
        # c_adv_loss_val = loss_reduced['c_adv_loss'].mean().item()
        c_adv_loss_val = 0

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'Dadv: {D_adv_loss_val:.2f}'
                    f'Gadv: {G_adv_loss_val:.2f}; GMS: {G_con_loss_val:.2f} {G_style_loss:.2f}; Grrrrrrr: {ci_loss_val:.2f} {cf_loss_val:.2f} {c_adv_loss_val:.2f};'
                )
            )

            if i % 500 == 0:
                with torch.no_grad():
                    test(args, G_A2B, G_B2A, testA_loader, testB_loader, 'normal', i)
                    #test(args, G_A2B_ema, G_B2A_ema, testA_loader, testB_loader, 'ema', i)

            if (i+1) % 2000 == 0:
                torch.save(
                    {
                        # 'G_A2B': G_A2B_module.state_dict(),
                        'G_B2A': G_B2A_module.state_dict(),
                        'G_A2B_encoder': G_A2B.encoder.state_dict(),
                        'G_A2B_decoder': G_A2B.decoder.state_dict(),
                        # 'G_A2B_ema': G_A2B_ema.state_dict(),
                        # 'G_B2A_ema': G_B2A_ema.state_dict(),
                        'D_A': D_A_module.state_dict(),
                        'D_B': D_B_module.state_dict(),
                        'D_L': D_L_module.state_dict(),
                        'G_optim': G_optim.state_dict(),
                        'D_optim': D_optim.state_dict(),
                        'iter': i,
                    },
                    os.path.join(model_path, 'ck.pt'),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', type=int, default=600000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--lambda_cycle', type=int, default=1)
    parser.add_argument('--path_regularize', type=float, default=1)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=15)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-3)
    # parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_down', type=int, default=3)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--d_path', type=str, required=True)
    parser.add_argument('--latent_dim', type=int, default=8)
    parser.add_argument('--lr_mlp', type=float, default=0.01)
    parser.add_argument('--n_res', type=int, default=1)

    args = parser.parse_args()

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    save_path = f'./{args.name}'
    im_path = os.path.join(save_path, 'sample')
    model_path = os.path.join(save_path, 'checkpoint')
    os.makedirs(im_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    args.n_mlp = 5

    args.start_iter = 0

    G_A2B = Generator2_alpha( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)
    D_A = Discriminator(args.size).to(device)
    G_B2A = Generator2_alpha( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)
    D_B = Discriminator(args.size).to(device)
    D_L = LatDiscriminator(args.latent_dim).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # G_A2B_ema = copy.deepcopy(G_A2B).to(device).eval()
    # G_B2A_ema = copy.deepcopy(G_B2A).to(device).eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    G_optim = optim.Adam( list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=args.lr, betas=(0, 0.99))
    D_optim = optim.Adam(
        list(D_L.parameters()) + list(D_A.parameters()) + list(D_B.parameters()),
        lr=args.lr, betas=(0**d_reg_ratio, 0.99**d_reg_ratio))

    if args.ckpt is not None:
        ckpt = torch.load("/data/cairui/CRGANsNRoses/GANsNRoses/results11/checkpoint/ck.pt", map_location=lambda storage, loc: storage)
        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        # G_A2B.load_state_dict(ckpt['G_A2B'])
        G_A2B.encoder.load_state_dict(ckpt['G_A2B_encoder'])
        G_A2B.decoder.load_state_dict(ckpt['G_A2B_decoder'])
        G_B2A.load_state_dict(ckpt['G_B2A'])
        # G_A2B_ema.load_state_dict(ckpt['G_A2B_ema'])
        # G_B2A_ema.load_state_dict(ckpt['G_B2A_ema'])
        D_A.load_state_dict(ckpt['D_A'])
        D_B.load_state_dict(ckpt['D_B'])
        D_L.load_state_dict(ckpt['D_L'])

        G_optim.load_state_dict(ckpt['G_optim'])
        D_optim.load_state_dict(ckpt['D_optim'])
        args.start_iter = ckpt['iter']
        # args.start_iter = 0


        #crrrrrrrrrrrrrr add
        #torch.save(
        #    {
        #        'G_A2B_encoder': G_A2B.encoder.state_dict(),
        #        'G_B2A': ckpt['G_B2A'],
        #        'D_A': ckpt['D_A'],
        #        'D_B': ckpt['D_B'],
        #        'D_L': ckpt['D_L'],
        #        'G_optim': ckpt['G_optim'],
        #        'D_optim': ckpt['D_optim'],
        #        'iter': 0,
        #    },
        #    os.path.join(model_path, 'ck_encoder.pt'),
        #)

    if args.distributed:
        G_A2B = nn.parallel.DistributedDataParallel(
            G_A2B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        D_A = nn.parallel.DistributedDataParallel(
            D_A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        G_B2A = nn.parallel.DistributedDataParallel(
            G_B2A,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        D_B = nn.parallel.DistributedDataParallel(
            D_B,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
        D_L = nn.parallel.DistributedDataParallel(
            D_L,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    augB = nn.Sequential(
        K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )

    augA = nn.Sequential(
        K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )

    augB2 = nn.Sequential(
        # K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )

    augA2 = nn.Sequential(
        # K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )

    augB3 = nn.Sequential(
        # K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        # K.RandomHorizontalFlip(),
    )

    augA3 = nn.Sequential(
        # K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        # K.RandomHorizontalFlip(),
    )


    d_path = args.d_path
    
    trainA = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'A_256'), train_transform)
    trainB = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'B_512'), train_transform)
    testB = ImageFolder(os.path.join("/data/cairui/UGATIT-pytorch/dataset/selfie2anime", 'testB'), test_transform)
    testA = ImageFolder(os.path.join("/data/cairui/GANsNRoses/", 'testimg2'), test_transform)

    trainA_loader = data.DataLoader(trainA, batch_size=args.batch, 
            sampler=data_sampler(trainA, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=2)
    trainB_loader = data.DataLoader(trainB, batch_size=args.batch, 
            sampler=data_sampler(trainB, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=2)

    testA_loader = data.DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = data.DataLoader(testB, batch_size=1, shuffle=False)


    train(args, trainA_loader, trainB_loader, testA_loader, testB_loader, G_A2B, G_B2A, D_A, D_B, G_optim, D_optim, device)
    # with torch.no_grad():
    #    test(args, G_A2B, G_B2A, testA_loader, testB_loader, 'normal', 444444, trainA_bg_loader, trainB_bg_loader)



