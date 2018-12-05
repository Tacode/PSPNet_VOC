'''
To combine precdicted label img with raw rgb img
'''
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import argparse

# fusion image
def overlayImage(mask, img, w, h):
    num_pixel = w*h
    srcimg = np.reshape(img,[num_pixel,3])
    maskimg = np.reshape(mask,[num_pixel,1])
    fusion_img = np.zeros([num_pixel,3],np.uint8)
    yuv_from_rgb = np.array([[0.299,0.587,0.144],
                [-0.14714119,-0.28886916,0.43601035],
                [0.61497538,-0.51496512,-0.10001026]])

    rgb_from_yuv = np.linalg.inv(yuv_from_rgb)

    color_map = np.array([
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ])

    for i in range(0,num_pixel):
        if maskimg[i] !=0 :
            Y = srcimg[i].dot(yuv_from_rgb[0].T.copy())
            U = color_map[maskimg[i][0]].dot(yuv_from_rgb[1].T.copy())
            V = color_map[maskimg[i][0]].dot(yuv_from_rgb[2].T.copy())
            rgb = np.array([Y,U,V]).dot(rgb_from_yuv.T.copy())
            rgb = np.clip(rgb,0,255,out=None)
            fusion_img[i] = rgb
        else:
            fusion_img[i] = srcimg[i]

    return np.reshape(fusion_img,[w,h,3])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_path',type=str,default='/home/ycon/code/pspnet-pytorch/person_label.png',help='mask img path')
    parser.add_argument('--img_path',type=str,default='/home/ycon/code/pspnet-pytorch/person.jpg',help='rgb img path')
    args = parser.parse_args()
    mask = np.array(Image.open(args.mask_path))
    img = np.array(Image.open(args.img_path))

    w,h = img.shape[0],img.shape[1]
    result = overlayImage(mask,img,w,h)

    plt.figure()
    plt.imshow(result)
    plt.axis('off')
    plt.show()
