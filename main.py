import cv2
import numpy as np
from torch.autograd import Variable
import torch
import torch.nn as nn
from models.vgg16_conv_deconv import VGG_Conv, VGG_Deconv, Identity
img = cv2.imread("dog.jpeg")
img = cv2.resize(img, (224, 224)) / 255.


INPUT = np.transpose(img, [2, 0, 1])

INPUT = Variable(torch.FloatTensor([INPUT])).cuda()

conv_net = VGG_Conv().cuda()
conv_net.eval()
deconv_net = VGG_Deconv(nn.ReLU).cuda()
deconv_net.eval()
out = conv_net(INPUT)

conv_layer_idxes = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]
out = deconv_net(conv_net.outputs[19][0][88][None, None,:,:], 19, 88, conv_net.maxpool_idx).cpu().data.numpy()[0]

out = np.transpose(out, [1, 2, 0])
out = (out-out.min()) /(out.max()-out.min()) * 255
out = out.astype(np.uint8)
cv2.imshow("raw img", img)
cv2.imshow("output", out)

cv2.waitKey(0)
