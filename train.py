import os
import logging
import datetime
import torch
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
import click
import numpy as np
from voc_loader import pascalVOCLoader
from pspnet import PSPNet
from metrics import runningScore
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
        _,epoch,_= os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        logging.info("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch

def get_logger(logdir):
    logger = logging.getLogger('PSPNet_segmentation')
    ts = str(datetime.datetime.now()).split('.')[0].replace(" ","_")
    ts = ts.replace(":","_").replace("-","_")
    file_path = os.path.join(logdir,'{}.log'.format(ts))
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger


logdir = os.path.join(os.getcwd(),'log')
logger = get_logger(logdir)
logger.info("Training start")

@click.command()
@click.option('--data-path', type=str, default='/home/ycon/dataset/VOCdevkit/VOC2012',help='Path to dataset folder')
@click.option('--models-path', type=str, default='/home/ycon/code/pspnet-pytorch/model',help='Path for storing model snapshots')
# @click.option('--data-path', type=str, default='/media/tyl/File/dataSets/VOCdevkit/VOC2012',help='Path to dataset folder')
# @click.option('--models-path', type=str, default='/home/tyl/Code/pycharm/pspnet-pytorch',help='Path for storing model snapshots')
@click.option('--backend', type=str, default='resnet50', help='Feature extractor')
@click.option('--snapshot', type=str, default=None, help='Path to pretrained weights')
@click.option('--crop_x', type=int, default=256, help='Horizontal random crop size')
@click.option('--crop_y', type=int, default=256, help='Vertical random crop size')
@click.option('--batch-size', type=int, default=16)
@click.option('--alpha', type=float, default=0.4, help='Coefficient for classification loss term')
@click.option('--epochs', type=int, default=1000, help='Number of training epochs to run')
@click.option('--gpu', type=str, default='2', help='List of GPUs for parallel training, e.g. 0,1,2,3')
@click.option('--start-lr', type=float, default=0.001)
@click.option('--milestones', type=str, default='15,25,40,50,60,70,80', help='Milestones for LR decreasing')
def train(data_path, models_path,
          backend, snapshot,
          crop_x, crop_y,
          batch_size, alpha,
          epochs,start_lr,
          milestones, gpu):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    net, starting_epoch = build_network(snapshot, backend)
    data_path = os.path.abspath(os.path.expanduser(data_path))
    models_path = os.path.abspath(os.path.expanduser(models_path))
    os.makedirs(models_path, exist_ok=True)

    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes, 
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''

    voc_data = pascalVOCLoader(root=data_path,is_transform=True,augmentations=None)
    # train_loader, class_weights, n_images = None, None, None
    train_loader = DataLoader(voc_data,batch_size=batch_size,shuffle=True,num_workers=0)
    max_steps = len(voc_data)
    class_weights = None

    optimizer = optim.Adam(net.parameters(), lr=start_lr)
    scheduler = MultiStepLR(optimizer, milestones=[int(x) for x in milestones.split(',')],gamma=0.1)
    running_score = runningScore(21)
    for epoch in range(starting_epoch, starting_epoch + epochs):
        seg_criterion = nn.NLLLoss2d(weight=class_weights)
        cls_criterion = nn.BCEWithLogitsLoss(weight=class_weights)
        epoch_losses = []
        # train_iterator = tqdm(train_loader, total=max_steps // batch_size + 1)
        net.train()
        print('------------epoch[{}]----------'.format(epoch+1))
        for i,(x, y, y_cls) in enumerate(train_loader):
            optimizer.zero_grad()
            x, y, y_cls = Variable(x).cuda(), Variable(y).cuda(), Variable(y_cls).float().cuda()
            out, out_cls = net(x)
            pred = out.data.max(1)[1].cpu().numpy()
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha * cls_loss
            epoch_losses.append(loss.item())
            running_score.update(y.data.cpu().numpy(),pred)
            if (i+1)%138 == 0:
                score,class_iou = running_score.get_scores()
                for k,v in score.items():
                    print(k,v)
                    logger.info('{}:{}'.format(k,v))
                running_score.reset()
            print_format_str = "Epoch[{}] batch[{}] loss = {:.4f} LR = {}"
            print_str = print_format_str.format(epoch+1,i+1,loss.item(),scheduler.get_lr()[0])
            print(print_str)
            logger.info(print_str)

            '''
            status = '[{}] loss = {:.4f} avg = {:.4f}, LR = {}'.format(
                epoch + 1, loss.item(), np.mean(epoch_losses), scheduler.get_lr()[0])
            train_iterator.set_description(status)
            '''
            loss.backward()
            optimizer.step()

        scheduler.step()
        if epoch+1 > 20:
            train_loss = ('%.4f'%np.mean(epoch_losses))
            torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch + 1),train_loss])+'.pth'))


        
if __name__ == '__main__':
    train()
