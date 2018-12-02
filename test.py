#测试图像代码，输入单张图片
import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import functional
from pspnet import PSPNet
from voc_loader import pascalVOCLoader
from metrics import runningScore
import click
import numpy as np
import scipy.misc as misc
import PIL.Image as Image
import time
torch.cuda.set_device(2)

models = {
    'squeezenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='squeezenet'),
    'densenet': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=1024, deep_features_size=512, backend='densenet'),
    'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
    #'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
    'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
    'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
    'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}

def build_network(snapshot, backend):
    epoch = 0
    backend = backend.lower()
    net = models[backend]()
    # net = nn.DataParallel(net)
    net = net.cuda()
    if snapshot is not None:
        _, epoch,_ = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
    net = net.cuda()
    return net, epoch

@click.command()
@click.option('--data-path',type=str,default='/home/ycon/dataset/VOCdevkit/VOC2012',help='VOC data path')
@click.option('--image',type=str,default='person.jpg',help='input image')
@click.option('--snapshot',type=str,default='./model/PSPNet_92_0.1617.pth',help='model weight')
@click.option('--backend',type=str,default='resnet50',help='backend model')
def test(data_path,image,snapshot,backend):
    start_time = time.time()
    net,epoch = build_network(snapshot,backend)
    print('Load Model Time:{:.4f}s'.format(time.time()-start_time))
    loader = pascalVOCLoader(data_path,is_transform=True)
    img = Image.open(image)
    w,h = img.size
    img = img.resize((224,224))
    img = functional.to_tensor(img).float()
    img = functional.normalize(img,[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]).unsqueeze(0)

    net.eval()
    start_time = time.time()
    img = Variable(img).cuda()
    out,out_cls = net(img)
    print("Inference Time:{:.4f}s".format(time.time() - start_time))

    pred = np.squeeze(out.data.max(1)[1].cpu().numpy(),axis=0)
    pred = misc.imresize(pred,(h,w),interp='nearest', mode="F")
    decode_img = loader.decode_segmap(pred)

    misc.imsave('_'.join([image.split('.')[0],'mask'])+'.png',decode_img)
    # misc.imsave('_'.join([image.split('.')[0],'label'])+'.png',pred)
    misc.toimage(pred,high=np.max(pred),low=np.min(pred)).save('_'.join([image.split('.')[0],'label'])+'.png')


if __name__ == '__main__':
    test()