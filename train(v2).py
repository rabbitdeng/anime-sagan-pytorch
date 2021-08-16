import argparse
import torch
import torchvision
import torchvision.utils as vutils
import torch.nn as nn
from tqdm import tqdm
from model import Generator, Discriminator
from losses import Wasserstein, Hinge
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def weights_init(m):  # 初始化模型权重
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default =8)
parser.add_argument('--imageSize', type=int, default= 64)
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epoch', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--lrd', type=float, default=2e-4,
                    help="Discriminator's learning rate, default=0.00005")  # Discriminator's learning rate
parser.add_argument('--lrg', type=float, default=2e-4,
                    help="Generator's learning rate, default=0.00005")  # Generator's learning rate
parser.add_argument('--data_path', default='data/', help='folder to train data')  # 将数据集放在此处
parser.add_argument('--outf', default='img/',
                    help='folder to output images and model checkpoints')  # 输出生成图片以及保存模型的位置
opt = parser.parse_args()
# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像读入与预处理
transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(opt.imageSize),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

dataset = torchvision.datasets.ImageFolder(opt.data_path, transform=transforms)

dataloader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    drop_last=True,
)
netG = Generator().to(device)
netG.apply(weights_init)
print('Generator:')
print(sum(p.numel() for p in netG.parameters()))

netD = Discriminator().to(device)
netD.apply(weights_init)
print('Discriminator:')
print(sum(p.numel() for p in netD.parameters()))

print(dataset)
start_epoch = 0# 设置初始epoch大小
#netG.load_state_dict(torch.load('img/netG_0265.pth', map_location=device))  # 这两句用来读取预训练模型
#netD.load_state_dict(torch.load('img/netD_0265.pth', map_location=device))  # 这两句用来读取预训练模型
criterionG = Hinge()
optimizerG = torch.optim.RMSprop(netG.parameters(), lr=opt.lrg)
optimizerD = torch.optim.RMSprop(netD.parameters(), lr=opt.lrd)

lrd_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=5, eta_min=5E-5)
lrg_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=5, eta_min=5E-5)

criterionD = Hinge()

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
total_lossD = 0.0
total_lossG = 0.0
label = label.unsqueeze(1)

#dnum = 3
#gnum = 3
for epoch in range(start_epoch + 1, opt.epoch + 1):
    with tqdm(total=len(dataloader), desc=f'Epoch {epoch}/{opt.epoch}', postfix=dict, mininterval=0.3) as pbar:
        for i, (imgs, _) in enumerate(dataloader):
            # for i in range(1, 5):
            # 固定生成器G，训练鉴别器D
            # 让D尽可能的把真图片判别为1
            imgs = imgs.to(device)
            # for k in range(1,5):
            outputreal = netD(imgs)
            optimizerD.zero_grad()
            ## 让D尽可能把假图片判别为0
            # label.data.fill_(fake_label)
            noise = torch.randn(opt.batchSize, opt.nz, 1, 1)
            # noise = torch.randn(opt.batchSize, opt.nz)
            noise = noise.to(device)
            fake = netG(noise)  # 生成假图
            outputfake = netD(fake.detach())  # 避免梯度传到G，因为G不用更新
            lossD = criterionD(outputreal, outputfake)
            total_lossD += lossD.item()
            lossD.backward()
            optimizerD.step()
            lrd_scheduler.step()
            #for k in range(1, 3):
            # 固定鉴别器D，训练生成器G
            noise = torch.randn(opt.batchSize, opt.nz)
            # noise = torch.randn(opt.batchSize, opt.nz)
            noise = noise.to(device)
            fake = netG(noise)  # 生成假图
            optimizerG.zero_grad()
            # 让D尽可能把G生成的假图判别为1
            output = netD(fake)
            lossG = criterionG(output)
            total_lossG += lossG.item()
            lossG.backward()
            optimizerG.step()
            pbar.set_postfix(**{'total_lossD': total_lossD / ((i + 1)),
                            'lrd': get_lr(optimizerD), 'total_lossG': total_lossG / ((i + 1)),
                            'lrg': get_lr(optimizerG)})
            pbar.update(1)

    lrg_scheduler.step()

    vutils.save_image(fake.data,
                      '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                      normalize=True)
    log = open("./log.txt", 'a')
    print('[%d/%d] total_Loss_D: %.3f total_Loss_G %.3f' % (
        epoch, opt.epoch, total_lossD / (len(dataloader)), total_lossG / ((len(dataloader)))),
          file=log)
    total_lossG = 0.0
    total_lossD = 0.0
    log.close()
    if epoch % 5 == 0:  # 每5个epoch，保存一次模型参数.
        torch.save(netG.state_dict(), '%s/netG_%04d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_%04d.pth' % (opt.outf, epoch))
