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

from UNetDiscriminator import UNetDiscriminatorSN

import onnxruntime as rt

from img_process_util import USMSharp

from tiny_generator import ColorLoss, DWTLoss

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
# smooth_l1 = nn.SmoothL1Loss()
smooth_l1 = nn.MSELoss()
BCE_stable = torch.nn.BCEWithLogitsLoss()
color_loss = ColorLoss()
dwt_loss = DWTLoss()


test_style1 = torch.tensor([
            [ 0.1846, -0.2431, -1.3536, -0.3894,  0.2999,  0.5944,  1.6446,  0.5755],
            [ 1.2578, -0.9200,  1.7270, -0.4708, -0.3693, -0.2677, -0.9341, -2.4605],
            [-0.0409, -0.4309,  0.3021, -1.0657, -0.2561,  0.2646, -0.0557,  1.8049],
            [ 1.3629, -1.4933,  0.3541,  2.2974,  1.4396,  1.2666,  0.7013,  0.6278],
            [ 2.4832,  0.2847, -1.2948, -0.6347,  0.5751,  1.5493,  1.2978, -1.4459],
            [-2.2184,  0.7497, -1.3328, -0.5818,  0.9301, -0.7380,  0.3121, -0.3337],
            [-1.3919, -0.0446, -0.2010, -0.0618,  1.9520,  1.1634, -0.5198,  2.0453],
            [-1.2438, -0.9917,  0.2463,  0.3556, -0.4241, -1.2965, -1.5335,  0.1136]
    ]).cuda()
test_style2 = torch.tensor([
            [-1.6180,  0.4169, -0.7037,  2.0596,  0.9493,  0.2213,  0.7044,  0.5933],
            [-0.6961, -0.0417, -2.6189,  0.5123,  0.4004,  0.2895, -0.3006,  0.6708],
            [ 0.4231,  1.4305,  0.6857, -0.7670, -1.8744,  0.0484,  0.8418, -1.2577],
            [-1.0564,  0.9342, -1.8528,  0.9923, -0.2262,  2.9734, -1.3435, -0.3238],
            [-0.8024,  0.9277,  0.8792,  0.1586, -0.0048, -0.6916, -0.1100,  0.0137],
            [-0.0605,  0.7876, -0.7488,  0.2885,  0.4425, -0.0072, -0.4276, -1.0051],
            [ 0.1614,  1.8066, -0.3945,  1.3077, -1.2341, -0.6516,  0.5003, -1.0751],
            [ 1.4490, -1.3888, -1.6753,  0.2481, -0.3849,  1.0237, -1.2200, -0.4761]
    ]).cuda()
# test_style3 = torch.tensor([
#             [-1.6180,  0.4169, -0.7037,  2.0596,  0.9493,  0.2213,  0.7044,  0.5933],
#         [ 1.2578, -0.9200,  1.7270, -0.4708, -0.3693, -0.2677, -0.9341, -2.4605],
#         [ 0.4231,  1.4305,  0.6857, -0.7670, -1.8744,  0.0484,  0.8418, -1.2577],
#         [ 1.3629, -1.4933,  0.3541,  2.2974,  1.4396,  1.2666,  0.7013,  0.6278],
#         [-0.8024,  0.9277,  0.8792,  0.1586, -0.0048, -0.6916, -0.1100,  0.0137],
#         [-0.0605,  0.7876, -0.7488,  0.2885,  0.4425, -0.0072, -0.4276, -1.0051],
#         [ 0.1614,  1.8066, -0.3945,  1.3077, -1.2341, -0.6516,  0.5003, -1.0751],
#         [ 1.4490, -1.3888, -1.6753,  0.2481, -0.3849,  1.0237, -1.2200, -0.4761]
#     ]).cuda()
def test(args, genA2B, T_Encoder, T_Decoder, testA_loader, testB_loader, name, step):
    testA_loader = iter(testA_loader)
    testB_loader = iter(testB_loader)
    with torch.no_grad():
        test_sample_num = 16

        genA2B.eval()
        A2B = []
        B2A = []
        for i in range(test_sample_num):
            real_A = testA_loader.next()
            real_B = testB_loader.next()

            real_A, real_B = real_A.cuda(), real_B.cuda()

    
            # A2B_content, A2B_style = genA2B.encode(real_A)
            A2B_content, A2B_style = T_Encoder(real_A)

            A2B_mod1 = test_style1[i//2:i//2+1, :]
            A2B_mod2 = test_style2[i//2:i//2+1, :]
            # A2B_mod3 = test_style3[i//2:i//2+1, :]
            # if i % 2 == 0:
            #     A2B_mod1 = torch.randn([1, args.latent_dim]).cuda()
            #     A2B_mod2 = torch.randn([1, args.latent_dim]).cuda()

            A2B_content_t, _ = T_Encoder(real_A)
            fake_A2B_t, alpha_t = T_Decoder(A2B_content_t, A2B_mod1)
            alpha_t = alpha_t.repeat(1, 3, 1, 1)

            colsA = [real_A, fake_A2B_t, alpha_t]
            
            fake_A2B_1, alpha1 = genA2B.decode(A2B_content, A2B_mod1)
            # fake_A2B_1, alpha1 = T_Decoder(A2B_content, A2B_mod1)
            alpha1 = alpha1.repeat(1, 3, 1, 1)

            fake_A2B_2, alpha2 = genA2B.decode(A2B_content, A2B_mod2)
            # fake_A2B_2, alpha2 = T_Decoder(A2B_content, A2B_mod1)
            alpha2 = alpha2.repeat(1, 3, 1, 1)

            colsA += [fake_A2B_1, alpha1, fake_A2B_2, alpha2]

            colsA = torch.cat(colsA, 2).detach().cpu()

            A2B.append(colsA)
        A2B = torch.cat(A2B, 0)

        utils.save_image(A2B, f'{im_path}/{name}_A2B_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)

        genA2B.train()


def train(args, trainA_loader, trainA2_loader, trainA3_loader, trainB_loader, testA_loader, testB_loader, G_A2B, D_B, G_optim, D_optim, device, T_Decoder, T_Encoder, G_B2A):
    G_A2B.train(), D_B.train()
    trainA_loader = sample_data(trainA_loader)
    trainA2_loader = sample_data(trainA2_loader)
    trainA3_loader = sample_data(trainA3_loader)
    trainB_loader = sample_data(trainB_loader)
    G_scheduler = lr_scheduler.StepLR(G_optim, step_size=100000, gamma=0.5)
    D_scheduler = lr_scheduler.StepLR(D_optim, step_size=100000, gamma=0.5)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.1)

    loss_dict = {}
    mean_path_length_A2B = 0
    mean_path_length_B2A = 0

    if args.distributed:
        G_A2B_module = G_A2B.module
        D_B_module = D_B.module

    else:
        G_A2B_module = G_A2B
        D_B_module = D_B

    # matting
    sess = rt.InferenceSession('rvm_mobilenetv3_fp32.onnx')
    rec = [ np.zeros([1, 1, 1, 1], dtype=np.float32) ] * 4  # Must match dtype of the model.
    downsample_ratio = np.array([1], dtype=np.float32)

    sharper = USMSharp().cuda()
    loss_weights = torch.ones(6, requires_grad=False).to('cuda')
    decoder_l1_w = -1
    decoder_color_w = -1
    decoder_dwt_w = -1
    decoder_lpips_w = -1
    decoder_cycle_content_w = -1
    decoder_cycle_style_w = -1
    decoder_sum = -1
    decoder_adv = -1
    discriminator_adv = -1

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        #G_A2B.train(), G_A2B.encoder.eval(), G_B2A.eval(), D_A.train(), D_B.train(), T_Decoder.eval()
        #G_A2B.train(), G_A2B.decoder.eval(), G_B2A.eval(), D_A.train(), D_B.train(), T_Encoder.eval()
        # G_A2B.train(), G_B2A.eval(), D_A.train(), D_B.eval(), T_Encoder.eval(), T_Decoder.eval(), D_L.eval()
        # G_A2B.train(), G_B2A.eval(), D_A.eval(), D_B.train(), T_Encoder.eval(), T_Decoder.eval(), D_L.eval()
        G_A2B.train(), D_B.train(), T_Encoder.eval(), T_Decoder.eval(), G_B2A.eval()
        # G_A2B.train(), G_B2A.train(), D_A.train(), D_B.train(), T_Encoder.eval(), T_Decoder.eval(), D_L.train()
        # for p_i in G_A2B.decoder.parameters():
        #     p_i.requires_grad=True
        # for p_i in G_A2B.encoder.parameters():
        #     p_i.requires_grad=True
        for p_i in T_Encoder.parameters():
            p_i.requires_grad=False
        for p_i in T_Decoder.parameters():
            p_i.requires_grad=False
        
        ori_A = next(trainA_loader)
        if i % 11 == 0 and i % 2 == 0:
            ori_A = next(trainA2_loader)
        if i % 11 == 0 and i % 2 == 1:
            ori_A = next(trainA3_loader)
        ori_B = next(trainB_loader)
        if isinstance(ori_A, list):
            ori_A = ori_A[0]
        if isinstance(ori_B, list):
            ori_B = ori_B[0]

        ori_A = ori_A.to(device)
        ori_B = ori_B.to(device)
        aug_A = aug(ori_A)
        aug_B = aug(ori_B)
        # A = aug(ori_A[[np.random.randint(args.batch)]].expand_as(ori_A))
        # B = aug(ori_B[[np.random.randint(args.batch)]].expand_as(ori_B))
        A = aug_A
        B = aug_B

        # matting
        A_denorm = A*0.5+0.5
        A_denorm = A_denorm.detach().cpu().numpy()
        A_denorm = A_denorm.astype(np.float32)
        res_fake = []
        fir, alpha_ret, *res_fake= sess.run([], {
            'src': A_denorm, 
            'r1i': rec[0], 
            'r2i': rec[1], 
            'r3i': rec[2], 
            'r4i': rec[3], 
            'downsample_ratio': downsample_ratio
        })
        rvm_alpha = torch.from_numpy(alpha_ret).float().to(device)

        if i % args.d_reg_every == 0:
            aug_A.requires_grad = True
            aug_B.requires_grad = True
        
        # A2B_content, A2B_style = G_A2B.encode(A)
        A2B_content_t, A2B_style_t = T_Encoder(A)
        #print(A2B_content)

        # get new style
        rand_A2B_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        # fake_A2B, alpha = G_A2B.decode(A2B_content.detach(), rand_A2B_style)
        fake_A2B, alpha = G_A2B.decode(A2B_content_t, rand_A2B_style)
        # fake_A2B_onlyencoder, _ = T_Decoder(A2B_content, rand_A2B_style)
        # fake_A2B_t_origin, alpha_t = T_Decoder(A2B_content_t, rand_A2B_style)
        fake_A2B_t, alpha_t = T_Decoder(A2B_content_t, rand_A2B_style)

        fake_A2B_t = fake_A2B_t*0.5 + 0.5
        fake_A2B_t = sharper(fake_A2B_t)
        fake_A2B_t = ((fake_A2B_t-0.5)/0.5).detach()
        
        # train disc
        real_B_logit = D_B(aug_B)
        fake_B_logit = D_B(fake_A2B.detach())

        # global loss
        # D_loss = d_logistic_loss(real_B_logit, fake_B_logit)
        # D_loss = 1. * BCE_stable(real_B_logit - fake_B_logit, torch.ones_like(real_B_logit).to(device))
        D_loss = 2. * BCE_stable(real_B_logit - fake_B_logit, torch.ones_like(real_B_logit).to(device))

        if i % args.d_reg_every == 0:
            r1_loss = d_r1_loss(real_B_logit, aug_B)
            D_r1_loss = (args.r1 / 2 * r1_loss * args.d_reg_every)
            D_loss += D_r1_loss

        if i >= 200000:
        # if i >= 0:
            D_loss = loss_weights[2] * D_loss
        else:
            D_loss = 0 * D_loss
        loss_dict['D_adv'] = D_loss

        D_optim.zero_grad()
        D_loss.backward()
        # torch.nn.utils.clip_grad_norm_(D_B.parameters(), max_norm=5.)
        D_optim.step()

        #Generator
        # adv loss
        fake_B_logit = D_B(fake_A2B)

        # lambda_adv = (1, 1, 1)
        # G_adv_loss = 1 * g_nonsaturating_loss(fake_B_logit, lambda_adv) 
        real_B_logit = D_B(aug_B)
        # G_adv_loss = 5. * BCE_stable(fake_B_logit - real_B_logit.detach(), torch.ones_like(fake_B_logit).to(device))
        G_adv_loss = 2. * BCE_stable(fake_B_logit - real_B_logit.detach(), torch.ones_like(fake_B_logit).to(device))

        # teacher loss
        #c_tloss = 20*F.l1_loss(fake_A2B, fake_A2B_t)
        # kki = 1. + i/300000.0 * 9.0
        # kki = 10.0
        # kki = 5.0

        b_encoder_content, b_style = G_B2A.encode(fake_A2B)
        b_encoder_content2, b_style2 = G_B2A.encode(fake_A2B_t)

        decoder_l1 =  50*(F.l1_loss(fake_A2B, fake_A2B_t))
        # decoder_l1 =  200*(F.l1_loss(fake_A2B, fake_A2B_t))
        decoder_color =  50*color_loss(fake_A2B, fake_A2B_t)
        decoder_dwt =  50*(dwt_loss(fake_A2B, fake_A2B_t))
        # decoder_dwt =  5*(dwt_loss(fake_A2B, fake_A2B_t))
        decoder_lpips =  5*(lpips_fn(fake_A2B, fake_A2B_t).mean())
        decoder_cycle_content =  2000 * smooth_l1(b_encoder_content, b_encoder_content2)
        decoder_cycle_style =  0.2 * smooth_l1(b_style, b_style2)

        # decoder_lalpha = 0.01*(20*F.l1_loss(alpha, alpha_t))
        decoder_lalpha = 0.2*(F.l1_loss(alpha, rvm_alpha))
        
        # decoder_loss = loss_weights[0]*decoder_l1 + loss_weights[1]*decoder_color + loss_weights[2]*decoder_dwt + \
        #                loss_weights[3]*decoder_lpips + loss_weights[4]*decoder_cycle_content + loss_weights[5]*decoder_cycle_style + \
        #                decoder_lalpha
        decoder_loss = decoder_l1 + decoder_color + decoder_dwt + \
                       decoder_lpips + decoder_cycle_content + decoder_cycle_style + \
                       decoder_lalpha
        # decoder_loss = decoder_l1 + decoder_lpips + \
        #                decoder_lalpha

        # decoder_l1_e = 200*(F.l1_loss(fake_A2B_onlyencoder, fake_A2B_t_origin))
        # decoder_lpips_e = 100*(lpips_fn(fake_A2B_onlyencoder, fake_A2B_t_origin).mean())
        # encoder_c_loss = 1000 * smooth_l1(A2B_content, A2B_content_t)
        # encoder_s_loss = smooth_l1(A2B_style, A2B_style_t)
        # encoder_deloss = decoder_l1_e + decoder_lpips_e
        # encoder_loss = encoder_c_loss + encoder_deloss

        # if i % 50 == 0:
        #     print("decoder_l11: %.8f, decoder_lpips1: %.8f, encoder_c_loss1: %.8f, encoder_s_loss1: %.8f" % \
        #         (decoder_l11, decoder_lpips1, encoder_c_loss1, encoder_s_loss1))
        
        G_loss = decoder_loss
        # G_loss = decoder_l11 + 0.1*decoder_lpips1 + decoder_lalpha
        # G_loss = 0.1*decoder_l11 + 0.1*decoder_lpips1 + decoder_lalpha + decoder_color
        # G_loss = 0.1*decoder_l11 + 0.01*decoder_lpips1 + 0.1*decoder_lalpha
        G_adv_loss_f = loss_weights[1]*G_adv_loss
        if i < 20000:
            # G_loss = encoder_loss
            G_loss = decoder_loss
            # G_loss = decoder_loss
        elif i >= 20000 and i < 200000:
            # G_loss = decoder_loss + encoder_loss
            G_loss = decoder_loss
        else :
            G_loss = decoder_loss + G_adv_loss_f

        loss_dict['G_adv'] = G_adv_loss_f
        loss_dict['decoder_loss'] = decoder_loss

        # decoder_param = list(G_A2B.decoder.parameters())
        # w1 = torch.autograd.grad(decoder_loss, decoder_param[-1], retain_graph=True, create_graph=False)
        # w1 = torch.norm(w1[0][0:3], 2)
        # w2 = torch.autograd.grad(G_adv_loss, decoder_param[-1], retain_graph=True, create_graph=False)
        # w2 = torch.norm(w2[0][0:3], 2)
        # decoder_sum = w1 if decoder_sum == -1 else decoder_sum*0.98 + w1*0.02
        # decoder_adv = w2 if decoder_adv == -1 else decoder_adv*0.98 + w2*0.02

        # D_loss_tttt = 10. * BCE_stable(real_B_logit.detach() - fake_B_logit, torch.ones_like(real_B_logit).to(device))
        # tttt = torch.autograd.grad(D_loss_tttt, decoder_param[-1], retain_graph=True, create_graph=False)
        # tttt = torch.norm(tttt[0][0:3], 2)
        # discriminator_adv = tttt if discriminator_adv == -1 else discriminator_adv*0.98 + tttt*0.02


        G_optim.zero_grad()
        G_loss.backward()
        # G_loss.backward(retain_graph=True)

        # if i > 201000:
        # if i > 200100:
        #     if decoder_adv > 0.5 * decoder_sum:
        #         loss_weights[1].fill_( loss_weights[1]*0.88 + 0.12*(loss_weights[1] * decoder_adv/(0.5*decoder_sum)) )
        #         loss_weights[2].fill_( loss_weights[2]*0.88 + 0.12*(loss_weights[2] * (0.5*decoder_sum)/decoder_adv) )
        #         # loss_weights[1].fill_( 0.1*(decoder_sum/decoder_adv).detach() )
        #         # loss_weights[2].fill_( 0.5*0.1*(decoder_sum/decoder_adv).detach())
        # if i % 5 == 0:
        #     print("\nwwwww:::::::1", decoder_sum)
        #     print("wwwww:::::::2", decoder_adv)
        #     print("wwwww:::::::3", discriminator_adv)
        #     print("wwwww:::::::4", loss_weights[1])
        #     print("wwwww:::::::5", loss_weights[2])

        

        # grad-norm
        # decoder_param = list(G_A2B.decoder.parameters())
        # w1 = torch.autograd.grad(decoder_l1, decoder_param[-1], retain_graph=True, create_graph=False)
        # w1 = torch.norm(w1[0][0:3], 2)
        # w2 = torch.autograd.grad(decoder_color, decoder_param[-1], retain_graph=True, create_graph=False)
        # w2 = torch.norm(w2[0][0:3], 2)
        # w3 = torch.autograd.grad(decoder_dwt, decoder_param[-1], retain_graph=True, create_graph=False)
        # w3 = torch.norm(w3[0][0:3], 2)
        # w4 = torch.autograd.grad(decoder_lpips, decoder_param[-1], retain_graph=True, create_graph=False)
        # w4 = torch.norm(w4[0][0:3], 2)
        # w5 = torch.autograd.grad(decoder_cycle_content, decoder_param[-1], retain_graph=True, create_graph=False)
        # w5 = torch.norm(w5[0][0:3], 2)
        # w6 = torch.autograd.grad(decoder_cycle_style, decoder_param[-1], retain_graph=False, create_graph=False)
        # w6 = torch.norm(w6[0][0:3], 2)
        # decoder_l1_w = w1 if decoder_l1_w == -1 else decoder_l1_w*0.98 + w1*0.02
        # decoder_color_w = w2 if decoder_color_w == -1 else decoder_color_w*0.98 + w2*0.02
        # decoder_dwt_w = w3 if decoder_dwt_w == -1 else decoder_dwt_w*0.98 + w3*0.02
        # decoder_lpips_w = w4 if decoder_lpips_w == -1 else decoder_lpips_w*0.98 + w4*0.02
        # decoder_cycle_content_w = w5 if decoder_cycle_content_w == -1 else decoder_cycle_content_w*0.98 + w5*0.02
        # decoder_cycle_style_w = w6 if decoder_cycle_style_w == -1 else decoder_cycle_style_w*0.98 + w6*0.02
        if i % 500 == 0:
            # print("\nwwwww:::::::1", decoder_l1_w)
            # print("wwwww:::::::2", decoder_color_w)
            # print("wwwww:::::::3", decoder_dwt_w)
            # print("wwwww:::::::4", decoder_lpips_w)
            # print("wwwww:::::::5", decoder_cycle_content_w)
            # print("wwwww:::::::6", decoder_cycle_style_w)
            # print("wwwww--------", loss_weights)
            print("decoder_l1: %.8f, decoder_color: %.8f, decoder_dwt: %.8f, decoder_lpips: %.8f, decoder_cycle_content: %.8f, decoder_cycle_style: %.8f" % \
    (decoder_l1, decoder_color, decoder_dwt, decoder_lpips, decoder_cycle_content, decoder_cycle_style))
        # if i > 100:
        #     loss_weights[1] = 0.5*(decoder_l1_w/decoder_color_w)
        #     loss_weights[2] = 0.5*(decoder_l1_w/decoder_dwt_w)
        #     loss_weights[3] = 0.2*(decoder_l1_w/decoder_lpips_w)
        #     loss_weights[4] = 0.2*(decoder_l1_w/decoder_cycle_content_w)
        #     loss_weights[5] = 0.5*(decoder_l1_w/decoder_cycle_style_w)
        # torch.nn.utils.clip_grad_norm_(G_A2B.parameters(), max_norm=5.)
        G_optim.step()
        # torch.cuda.empty_cache()

        G_scheduler.step()
        D_scheduler.step()

        # accumulate(G_A2B_ema, G_A2B_module)
        # accumulate(G_B2A_ema, G_B2A_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        D_adv_loss_val = loss_reduced['D_adv'].mean().item()
        G_adv_loss_val = loss_reduced['G_adv'].mean().item()
        decoder_loss_val = loss_reduced['decoder_loss'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'Dadv: {D_adv_loss_val:.2f}; G_adv_loss_val: {G_adv_loss_val:.2f}'
                    f'decoder_loss_val: {decoder_loss_val:.2f};'
                )
            )

            if i % 500 == 0:
                with torch.no_grad():
                    test(args, G_A2B, T_Encoder, T_Decoder, testA_loader, testB_loader, 'normal', i)
                    #test(args, G_A2B_ema, G_B2A_ema, testA_loader, testB_loader, 'ema', i)

            if (i+1) % 2000 == 0:
                torch.save(
                    {
                        #'G_A2B': G_A2B_module.state_dict(),
                        'G_A2B_encoder': G_A2B.encoder.state_dict(),
                        'G_A2B_decoder': G_A2B.decoder.state_dict(),
                        # 'G_A2B_decoder2': T_Decoder.state_dict(),
                        # 'G_A2B_ema': G_A2B_ema.state_dict(),
                        # 'G_B2A_ema': G_B2A_ema.state_dict(),
                        'D_B': D_B_module.state_dict(),
                        'G_optim': G_optim.state_dict(),
                        'D_optim': D_optim.state_dict(),
                        'iter': i,
                    },
                    os.path.join(model_path, 'ck.pt'),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--iter', type=int, default=400000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--lambda_cycle', type=int, default=1)
    parser.add_argument('--path_regularize', type=float, default=2)
    parser.add_argument('--path_batch_shrink', type=int, default=2)
    parser.add_argument('--d_reg_every', type=int, default=8)
    # parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--mixing', type=float, default=0.9)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=2e-3)
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

    G_A2B = Generator3( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)
    T_Decoder = DecoderAlpha(args.size, args.num_down, args.latent_dim, args.n_mlp, args.n_res).to(device)
    T_Encoder = Encoder(args.size, 8, args.num_down, args.n_res, 1).to(device)
    G_B2A = Generator2_alpha( args.size, args.num_down, args.latent_dim, args.n_mlp, lr_mlp=args.lr_mlp, n_res=args.n_res).to(device)


    # D_B = Discriminator(args.size).to(device)
    D_B = UNetDiscriminatorSN(3).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # G_A2B_ema = copy.deepcopy(G_A2B).to(device).eval()
    # G_B2A_ema = copy.deepcopy(G_B2A).to(device).eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    G_optim = optim.Adam( list(G_A2B.parameters()), lr=args.lr, betas=(0, 0.99))
    D_optim = optim.Adam(list(D_B.parameters()), lr=args.lr, betas=(0**d_reg_ratio, 0.99**d_reg_ratio))

    if args.ckpt is not None:
        # ckpt = torch.load("/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/ck.pt", map_location=lambda storage, loc: storage)
        ckpt_teacher = torch.load("/data/cairui/CRGANsNRoses/GANsNRoses/result2/checkpoint/ck.pt", map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        # G_A2B.encoder.load_state_dict(ckpt['G_A2B_encoder'])
        # G_A2B.decoder.load_state_dict(ckpt['G_A2B_decoder'])
        # D_B.load_state_dict(ckpt['D_B'])
        # G_optim.load_state_dict(ckpt['G_optim'])
        # D_optim.load_state_dict(ckpt['D_optim'])
        # torch.save(G_A2B.encoder, "/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/encoder.pkl") 
        # torch.save(G_A2B.decoder, "/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/decoder.pkl") 
        T_Encoder.load_state_dict(ckpt_teacher['G_A2B_encoder'])
        T_Decoder.load_state_dict(ckpt_teacher['G_A2B_decoder'])
        G_B2A.load_state_dict(ckpt_teacher['G_B2A'])
        G_B2A.decoder = None

        # G_optim.load_state_dict(ckpt['G_optim'])
        # D_optim.load_state_dict(ckpt['D_optim'])
        # args.start_iter = ckpt2['iter']
        args.start_iter = 0
        # args.start_iter = 200000

    if args.distributed:
        G_A2B = nn.parallel.DistributedDataParallel(
            G_A2B,
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

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    test_transform = transforms.Compose([
        transforms.Resize((args.size, args.size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

    aug = nn.Sequential(
        K.RandomAffine(degrees=(-20,20), scale=(0.8, 1.2), translate=(0.1, 0.1), shear=0.15),
        kornia.geometry.transform.Resize(256+10),
        K.RandomCrop((256,256)),
        K.RandomHorizontalFlip(),
    )

    # trainA = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'A_256'), train_transform)
    # trainB = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'B_256'), train_transform)
    trainA = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'A_256'), train_transform)
    trainA2 = ImageFolder(os.path.join("/data/dataset/selfie2anime/", 'trainA'), train_transform)
    trainA3 = ImageFolder(os.path.join("/data/dataset/yellow", 'generated_yellow-stylegan2'), train_transform)
    trainB = ImageFolder(os.path.join("/data/dataset/crdata/CRDATA/", 'B_512'), train_transform)
    testB = ImageFolder(os.path.join("/data/cairui/UGATIT-pytorch/dataset/selfie2anime", 'testB'), test_transform)
    testA = ImageFolder(os.path.join("/data/cairui/GANsNRoses/", 'testimg2'), test_transform)

    trainA_loader = data.DataLoader(trainA, batch_size=args.batch, 
            sampler=data_sampler(trainA, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=3)
    trainA2_loader = data.DataLoader(trainA2, batch_size=args.batch, 
            sampler=data_sampler(trainA2, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=3)
    trainA3_loader = data.DataLoader(trainA3, batch_size=args.batch, 
            sampler=data_sampler(trainA3, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=3)
    trainB_loader = data.DataLoader(trainB, batch_size=args.batch, 
            sampler=data_sampler(trainB, shuffle=True, distributed=args.distributed), drop_last=True, pin_memory=True, num_workers=3)

    testA_loader = data.DataLoader(testA, batch_size=1, shuffle=False)
    testB_loader = data.DataLoader(testB, batch_size=1, shuffle=False)

    train(args, trainA_loader, trainA2_loader, trainA3_loader, trainB_loader, testA_loader, testB_loader, G_A2B, D_B, G_optim, D_optim, device,T_Decoder,T_Encoder,G_B2A)
    # with torch.no_grad():
    #    test(args, G_A2B, G_B2A, testA_loader, testB_loader, 'normal', 12244)



