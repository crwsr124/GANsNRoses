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

import onnxruntime as rt

from img_process_util import USMSharp

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

    
            A2B_content, A2B_style = genA2B.encode(real_A)

            if i % 2 == 0:
                A2B_mod1 = torch.randn([1, args.latent_dim]).cuda()
                A2B_mod2 = torch.randn([1, args.latent_dim]).cuda()

            A2B_content_t, _ = T_Encoder(real_A)
            fake_A2B_t, alpha_t = T_Decoder(A2B_content_t, A2B_mod1)
            alpha_t = alpha_t.repeat(1, 3, 1, 1)

            colsA = [real_A, fake_A2B_t, alpha_t]
            
            fake_A2B_1, alpha1 = genA2B.decode(A2B_content, A2B_mod1)
            alpha1 = alpha1.repeat(1, 3, 1, 1)

            fake_A2B_2, alpha2 = genA2B.decode(A2B_content, A2B_mod2)
            alpha2 = alpha2.repeat(1, 3, 1, 1)

            colsA += [fake_A2B_1, alpha1, fake_A2B_2, alpha2]

            colsA = torch.cat(colsA, 2).detach().cpu()

            A2B.append(colsA)
        A2B = torch.cat(A2B, 0)

        utils.save_image(A2B, f'{im_path}/{name}_A2B_{str(step).zfill(6)}.jpg', normalize=True, range=(-1, 1), nrow=16)

        genA2B.train()


def train(args, trainA_loader, trainA2_loader, trainA3_loader, trainB_loader, testA_loader, testB_loader, G_A2B, D_B, G_optim, D_optim, device, T_Decoder, T_Encoder):
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

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        #G_A2B.train(), G_A2B.encoder.eval(), G_B2A.eval(), D_A.train(), D_B.train(), T_Decoder.eval()
        #G_A2B.train(), G_A2B.decoder.eval(), G_B2A.eval(), D_A.train(), D_B.train(), T_Encoder.eval()
        # G_A2B.train(), G_B2A.eval(), D_A.train(), D_B.eval(), T_Encoder.eval(), T_Decoder.eval(), D_L.eval()
        # G_A2B.train(), G_B2A.eval(), D_A.eval(), D_B.train(), T_Encoder.eval(), T_Decoder.eval(), D_L.eval()
        G_A2B.train(), D_B.train(), T_Encoder.eval(), T_Decoder.eval()
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
        
        A2B_content, A2B_style = G_A2B.encode(A)
        A2B_content_t, A2B_style_t = T_Encoder(A)
        #print(A2B_content)

        # get new style
        rand_A2B_style = torch.randn([args.batch, args.latent_dim]).to(device).requires_grad_()
        fake_A2B, alpha = G_A2B.decode(A2B_content, rand_A2B_style)
        fake_A2B_t, alpha_t = T_Decoder(A2B_content_t, rand_A2B_style)

        fake_A2B_t = fake_A2B_t*0.5 + 0.5
        fake_A2B_t = sharper(fake_A2B_t)
        fake_A2B_t = ((fake_A2B_t-0.5)/0.5).detach()
        
        # train disc
        real_B_logit = D_B(aug_B)
        fake_B_logit = D_B(fake_A2B.detach())

        # global loss
        D_loss = d_logistic_loss(real_B_logit, fake_B_logit)

        

        if i % args.d_reg_every == 0:
            r1_loss = d_r1_loss(real_B_logit, aug_B)
            D_r1_loss = (args.r1 / 2 * r1_loss * args.d_reg_every)
            D_loss += D_r1_loss

        if i > 100000:
            D_loss = D_loss
        else:
            D_loss = 0 * D_loss

        loss_dict['D_adv'] = D_loss

        D_optim.zero_grad()
        D_loss.backward()
        D_optim.step()

        #Generator
        # adv loss
        fake_B_logit = D_B(fake_A2B)

        lambda_adv = (1, 1, 1)
        # G_adv_loss = g_nonsaturating_loss(fake_B_logit, lambda_adv)
        G_adv_loss = 1 * g_nonsaturating_loss(fake_B_logit, lambda_adv) 

        if i > 100000:
            G_adv_loss = G_adv_loss
        else:
            G_adv_loss = 0 * G_adv_loss

        # style consis loss
        G_con_loss = 0
        #G_con_loss = 50 * (A2B_style.var(0, unbiased=False).sum() + B2A_style.var(0, unbiased=False).sum())
                    
        # cycle recon
        # A2B2A_content, A2B2A_style = G_B2A.encode(fake_A2B)
        #print(A2B2A_content.shape)
        # B2A2B_content, B2A2B_style = G_A2B.encode(fake_B2A)

        # B2A2B_content_t, B2A2B_style_t = T_Encoder(fake_B2A)

        # A2B_style_s = shuffle_batch(A2B_style)
        # B2A_style_s = shuffle_batch(B2A_style)
        # fake_A2B2A = G_B2A.decode(A2B2A_content, A2B_style_s)
        # fake_B2A2B = G_A2B.decode(B2A2B_content, B2A_style_s)

        # fake_B2A2B_t = T_Decoder(B2A2B_content, B2A_style_s)

        #G_cycle_loss = 0 * (F.mse_loss(fake_A2B2A, A) + F.mse_loss(fake_B2A2B, B))
        G_cycle_loss = 0
        
        lpips_loss = 0.0
        #lpips_loss = 10 * (lpips_fn(fake_A2B2A, A).mean() + lpips_fn(fake_B2A2B, B).mean()) #10 for anime
        #A_downsample = F.avg_pool2d(A, kernel_size=4, stride=4)
        #fake_A2B2A_downsample = F.avg_pool2d(fake_A2B2A, kernel_size=4, stride=4)
        #B_downsample = F.avg_pool2d(B, kernel_size=4, stride=4)
        #fake_B2A2B_downsample = F.avg_pool2d(fake_B2A2B, kernel_size=4, stride=4)
        #lpips_loss = 10 * (lpips_fn(fake_A2B2A_downsample.mean(1), A_downsample.mean(1)).mean() + lpips_fn(fake_B2A2B_downsample.mean(1), B_downsample.mean(1)).mean()) #10 for anime
        #lpips_loss = 0

        # style reconstruction
        G_style_loss = 0
        # G_style_loss = 5 * (mse_criterion(A2B2A_style, input_A2B_style) +\
        #                     mse_criterion(B2A2B_style, input_B2A_style))

        # crloss
        #c_fake_B_logit = D_B(fake_B2A2B)
        #c_fake_A_logit = D_A(fake_A2B2A)
        #lambda_adv = (1, 1, 1)
        #c_adv_loss = 1 * (g_nonsaturating_loss(c_fake_A_logit, lambda_adv) +\
        #                 g_nonsaturating_loss(c_fake_B_logit, lambda_adv) )
        c_adv_loss = 0

        # feature presering loss
        # cf_loss = 100 * (F.l1_loss(A2B2A_content, A2B_content) +\
        #                     F.l1_loss(B2A2B_content, B2A_content))
        cf_loss = 0

        # identity loss
        #b_c, b_s = G_B2A.encode(B)
        #fake_B2B = G_A2B.decode(b_c, b_s)
        #a_c, a_s = G_A2B.encode(A)
        #fake_A2A = G_B2A.decode(a_c, a_s)
        #cf_loss_p = 25 * (F.l1_loss(fake_A2A, A) +\
                            # F.l1_loss(fake_B2B, B))
        #ci_loss = cf_loss_p + 15 * (lpips_fn(fake_A2A, A).mean() + lpips_fn(fake_B2B, B).mean())
        ci_loss = 0

        # teacher loss
        #c_tloss = 20*F.l1_loss(fake_A2B, fake_A2B_t)
        kki = 1. + i/300000.0 * 9.0
        decoder_l11 = kki*(20*F.l1_loss(fake_A2B, fake_A2B_t))
        decoder_lpips1 = kki*(10*lpips_fn(fake_A2B, fake_A2B_t).mean())
        # decoder_lalpha = 0.01*(20*F.l1_loss(alpha, alpha_t))
        decoder_lalpha = 0.01*(20*F.l1_loss(alpha, rvm_alpha))
        
        # decoder_l12 = (20*F.l1_loss(fake_B2A2B, fake_B2A2B_t))
        # decoder_lpips2 = (lpips_fn(fake_B2A2B, fake_B2A2B_t).mean())
        encoder_c_loss1 = 1000 * smooth_l1(A2B_content, A2B_content_t)
        #encoder_c_loss2 = 1000 * smooth_l1(B2A2B_content, B2A2B_content_t)
        encoder_s_loss1 = smooth_l1(A2B_style, A2B_style_t)
        #encoder_s_loss2 = smooth_l1(B2A2B_style, B2A2B_style_t)

        # if i % 50 == 0:
        #     print("decoder_l11: %.8f, decoder_lpips1: %.8f, encoder_c_loss1: %.8f, encoder_s_loss1: %.8f" % \
        #         (decoder_l11, decoder_lpips1, encoder_c_loss1, encoder_s_loss1))
        
        G_loss = decoder_l11 + decoder_lpips1  + decoder_lalpha + \
                encoder_c_loss1 + encoder_s_loss1  + G_adv_loss

        loss_dict['G_adv'] = G_adv_loss
        loss_dict['decoder_l11'] = decoder_l11
        loss_dict['decoder_lpips1'] = decoder_lpips1
        loss_dict['decoder_lalpha'] = decoder_lalpha
        loss_dict['encoder_c_loss1'] = encoder_c_loss1
        loss_dict['encoder_s_loss1'] = encoder_s_loss1

        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()

        G_scheduler.step()
        D_scheduler.step()

        # accumulate(G_A2B_ema, G_A2B_module)
        # accumulate(G_B2A_ema, G_B2A_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        D_adv_loss_val = loss_reduced['D_adv'].mean().item()
        G_adv_loss_val = loss_reduced['G_adv'].mean().item()
        decoder_l11_val = loss_reduced['decoder_l11'].mean().item()
        decoder_lpips1_val = loss_reduced['decoder_lpips1'].mean().item()
        decoder_lalpha_val = loss_reduced['decoder_lalpha'].mean().item()
        encoder_c_loss1_val = loss_reduced['encoder_c_loss1'].mean().item()
        encoder_s_loss1_val = loss_reduced['encoder_s_loss1'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'Dadv: {D_adv_loss_val:.2f}; G_adv_loss_val: {G_adv_loss_val:.2f}'
                    f'decoder_l11_val: {decoder_l11_val:.2f}; decoder_lpips1_val: {decoder_lpips1_val:.2f}; decoder_lalpha_val: {decoder_lalpha_val:.2f}; '
                    f'encoder_c_loss1_val: {encoder_c_loss1_val:.2f}; encoder_s_loss1_val: {encoder_s_loss1_val:.2f}; '
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
    parser.add_argument('--d_reg_every', type=int, default=16)
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


    D_B = Discriminator(args.size).to(device)
    lpips_fn = lpips.LPIPS(net='vgg').to(device)

    # G_A2B_ema = copy.deepcopy(G_A2B).to(device).eval()
    # G_B2A_ema = copy.deepcopy(G_B2A).to(device).eval()

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

    G_optim = optim.Adam( list(G_A2B.parameters()), lr=args.lr, betas=(0, 0.99))
    D_optim = optim.Adam(list(D_B.parameters()), lr=args.lr, betas=(0**d_reg_ratio, 0.99**d_reg_ratio))

    if args.ckpt is not None:
        ckpt = torch.load("/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/ck.pt", map_location=lambda storage, loc: storage)
        ckpt_teacher = torch.load("/data/cairui/CRGANsNRoses/GANsNRoses/result2/checkpoint/ck.pt", map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])
            
        except ValueError:
            pass
            
        G_A2B.encoder.load_state_dict(ckpt['G_A2B_encoder'])
        G_A2B.decoder.load_state_dict(ckpt['G_A2B_decoder'])
        torch.save(G_A2B.encoder, "/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/encoder.pkl") 
        torch.save(G_A2B.decoder, "/data/cairui/CRGANsNRoses/GANsNRoses/rlight9/checkpoint/decoder.pkl") 
        T_Encoder.load_state_dict(ckpt_teacher['G_A2B_encoder'])
        T_Decoder.load_state_dict(ckpt_teacher['G_A2B_decoder'])

        # G_optim.load_state_dict(ckpt['G_optim'])
        # D_optim.load_state_dict(ckpt['D_optim'])
        # args.start_iter = ckpt2['iter']
        args.start_iter = 100000

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

    train(args, trainA_loader, trainA2_loader, trainA3_loader, trainB_loader, testA_loader, testB_loader, G_A2B, D_B, G_optim, D_optim, device,T_Decoder,T_Encoder)
    # with torch.no_grad():
    #    test(args, G_A2B, G_B2A, testA_loader, testB_loader, 'normal', 12244)



