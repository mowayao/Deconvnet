import torch.nn as nn
from torchvision.models import vgg16




class VGG_Conv(nn.Module):
	def __init__(self):
		super(VGG_Conv, self).__init__()
		vgg_net = vgg16(pretrained=True)
		self.features = vgg_net.features
		for module in self.features:
			if isinstance(module, nn.MaxPool2d):
				module.return_indices = True
		self.maxpool_idx = {}
		self.outputs = [None for _ in xrange(len(self.features))]
	def forward(self, x):
		for i, layer in enumerate(self.features):
			if isinstance(layer, nn.MaxPool2d):
				x, idx = layer(x)
				self.maxpool_idx[i] = idx
			else:
				x = layer(x)
			self.outputs[i] = x
		return x
class Identity(nn.Module):

	def __init__(self):
		super(Identity, self).__init__()
	def forward(self, x):
		return x
class VGG_Deconv(nn.Module):
	def __init__(self, module):
		super(VGG_Deconv, self).__init__()
		
		self.features = nn.Sequential(
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
		)
		self.extras = nn.Sequential(
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=512, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=256, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
			nn.MaxUnpool2d(kernel_size=2, stride=2),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
			module(),
			nn.ConvTranspose2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
		)

		self._init_weights()
	def _init_weights(self):
		vgg_net = vgg16(pretrained=True)
		for i, layer in enumerate(reversed(vgg_net.features)):
			if isinstance(layer, nn.Conv2d):
				self.features[i].weight.data = layer.weight.data
				self.features[i].bias.data.zero_()
	def forward(self, x, layer_idx, feature_map_idx, pool_indices):
		sta = len(self.features) - layer_idx - 1
		assert isinstance(self.extras[sta], nn.ConvTranspose2d)
		in_c, out_c, k, k = self.features[sta].weight.data.size()
		self.extras[sta].weight.data = self.features[sta].weight[feature_map_idx].data.view(-1, out_c, k, k)
		self.extras[sta].bias.data = self.features[sta].bias.data
		x = self.extras[sta](x)
		for _ in xrange(sta+1, len(self.features)):
			if isinstance(self.features[_], nn.MaxUnpool2d):
				x = self.features[_](x, pool_indices[len(self.features)-1-_])
			else:
				x = self.features[_](x)
		return x
