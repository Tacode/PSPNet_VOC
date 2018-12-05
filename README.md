# PSPNet_VOC
Pytorch implementation of PSPNet on VOC, adapted from
[Lextal](https://github.com/Lextal/pspnet-pytorch) which just have
PSPNet model defination and training framework.Therefore,this repos complements the data reading code and model testing code.


## Training
**Datasets**: VOC2012+SBD<br>

|Metrics| Value|
|:--:|:--|
|Overall Acc|0.949|
|Mean Acc|0.876|
|FreqW IoU|0.905|
|Mean IoU|0.804|

## Testing
The model inference:
<div align="center">
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/mot.jpg" width="200" >
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/img/mot_mask.png"  width="200" >
</div>
<div align="center">
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/person.jpg" width="200" >
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/img/person_mask.png"  width="200" >
</div>

By `fusionMask.py`,we can get combanation of mask and img
<div align="center">
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/mot.jpg" width="200" >
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/img/mot_mask.png"  width="200" >
<img src="https://github.com/Tacode/PSPNet_VOC/blob/master/img/mot_fusion.png"  width="200" >
</div>
