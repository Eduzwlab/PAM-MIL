from __future__ import print_function, division
import os

import staintools
import torch
import numpy as np
import pandas as pd
import math
import re
import pdb
import pickle
from staintools import StainNormalizer
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils, models
import torch.nn.functional as F

from PIL import Image
import h5py

import stainNorm_Reinhard


def eval_transforms(pretrained=False):
	if pretrained:
		mean = (0.485, 0.456, 0.406)
		std = (0.229, 0.224, 0.225)

	else:
		mean = (0.5,0.5,0.5)
		std = (0.5,0.5,0.5)

	trnsfrms_val = transforms.Compose(
					[
					transforms.ToTensor(),
					 transforms.Normalize(mean = mean, std = std)
					]
				)

	return trnsfrms_val

# def eval_transforms(pretrained=False):
# 	# if pretrained:
# 	mean = (0.485, 0.456, 0.406)
# 	std = (0.229, 0.224, 0.225)
#
# 	# else:
# 	# mean = (0.5,0.5,0.5)
# 	# std = (0.5,0.5,0.5)
#
# 	transfrms_val = transforms.Compose(
# 					[
# 					transforms.CenterCrop(512),
# 					transforms.RandomHorizontalFlip(),
# 					transforms.RandomVerticalFlip(),
# 					transforms.RandomRotation(45),
# 					transforms.ColorJitter(0.25, 0.25, 0.25, 0.05),
# 					transforms.ToTensor(),
# 					transforms.Normalize(mean = mean, std = std)
# 					]
# 				)
#
# 	return transfrms_val

class Whole_Slide_Bag(Dataset):
	def __init__(self,
		file_path,
		pretrained=False,
		custom_transforms=None,
		target_patch_size=-1,
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
		"""
		self.pretrained=pretrained
		if target_patch_size > 0:
			self.target_patch_size = (target_patch_size, target_patch_size)
		else:
			self.target_patch_size = None

		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['imgs']
			self.length = len(dset)

		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['imgs']
		for name, value in dset.attrs.items():
			print(name, value)

		print('pretrained:', self.pretrained)
		print('transformations:', self.roi_transforms)
		if self.target_patch_size is not None:
			print('target_size: ', self.target_patch_size)

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			img = hdf5_file['imgs'][idx]
			coord = hdf5_file['coords'][idx]
		
		img = Image.fromarray(img)
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord




class Whole_Slide_Bag_FP(Dataset):
	def __init__(self,
		file_path,
		wsi,
		pretrained=False,
		custom_transforms=None,
		custom_downsample=1,
		target_patch_size=-1
		):
		"""
		Args:
			file_path (string): Path to the .h5 file containing patched data.
			pretrained (bool): Use ImageNet transforms
			custom_transforms (callable, optional): Optional transform to be applied on a sample
			custom_downsample (int): Custom defined downscale factor (overruled by target_patch_size)
			target_patch_size (int): Custom defined image size before embedding
		"""
		self.pretrained=pretrained
		self.wsi = wsi
		if not custom_transforms:
			self.roi_transforms = eval_transforms(pretrained=pretrained)
		else:
			self.roi_transforms = custom_transforms

		self.file_path = file_path

		with h5py.File(self.file_path, "r") as f:
			dset = f['coords']
			self.patch_level = f['coords'].attrs['patch_level']
			self.patch_size = f['coords'].attrs['patch_size']
			self.length = len(dset)
			if target_patch_size > 0:
				self.target_patch_size = (target_patch_size, ) * 2
			elif custom_downsample > 1:
				self.target_patch_size = (self.patch_size // custom_downsample, ) * 2
			else:
				self.target_patch_size = None
		self.summary()
			
	def __len__(self):
		return self.length

	def summary(self):
		hdf5_file = h5py.File(self.file_path, "r")
		dset = hdf5_file['coords']
		# 打出关于dset小图的相关信息
		for name, value in dset.attrs.items():
			print(name, value)

		print('\nfeature extraction settings')
		print('target patch size: ', self.target_patch_size)
		print('pretrained: ', self.pretrained)
		print('transformations: ', self.roi_transforms)


	# def __getitem__(self, idx):
	# 	with h5py.File(self.file_path,'r') as hdf5_file:
	# 		coord = hdf5_file['coords'][idx]
	# 	img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
	# 	if self.target_patch_size is not None:
	# 		img = img.resize(self.target_patch_size)
	# 	# 提取图片
	# 	means = [64.31591058, 24.72086653, -7.70439805]
	# 	stds = [20.2512827, 14.92409707, 9.88884396]
	#
	# 	means1 = [71.548488339274, 20.278528562594076, -6.573165458252838]
	# 	stds1 = [15.470829768258113, 14.440498566756364, 8.38252219312753]
	# 	# 初始化normalizer
	# 	normalizer = stainNorm_Reinhard.normalizer(means, stds)
	# 	img_normalized = normalizer.transform(img, means1, stds1)
	# 	img_normalized_pil = Image.fromarray(img_normalized)
	# 	save_path1 = "image.jpg"
	# 	save_path = "image-color.jpg"  # 替换成你想要保存的文件路径和名称
	# 	img_normalized_pil.save(save_path)
	# 	img.save(save_path1)
	# 	img = self.roi_transforms(img_normalized).unsqueeze(0)
	# 	# 标准化图片
	# 	return img, coord

	# def __getitem__(self, idx):
	# 	with h5py.File(self.file_path, 'r') as hdf5_file:
	# 		coord = hdf5_file['coords'][idx]
	# 	img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
	# 	if self.target_patch_size is not None:
	# 		img = img.resize(self.target_patch_size)
	# 	# 提取图片
	# 	means = [64.31591058, 24.72086653, -7.70439805]
	# 	stds = [20.2512827, 14.92409707, 9.88884396]
	#
	# 	means1 = [71.548488339274, 20.278528562594076, -6.573165458252838]
	# 	stds1 = [15.470829768258113, 14.440498566756364, 8.38252219312753]
	# 	# 初始化normalizer
	# 	normalizer = stainNorm_Reinhard.normalizer(means, stds)
	# 	img_normalized = normalizer.transform(img, means1, stds1)
	# 	# img_normalized_pil = Image.fromarray(img_normalized)
	# 	# 保存图片到两个不同的文件夹
	# 	# os.makedirs("folder5", exist_ok=True)
	# 	# os.makedirs("folder6", exist_ok=True)
	# 	# img_normalized_pil.save(os.path.join("folder5", f"image_color_{idx}.jpg"))
	# 	# img.save(os.path.join("folder6", f"image_{idx}.jpg"))
	#
	# 	img = self.roi_transforms(img_normalized).unsqueeze(0)
	# 	return img, coord

	def __getitem__(self, idx):
		with h5py.File(self.file_path,'r') as hdf5_file:
			coord = hdf5_file['coords'][idx]
		img = self.wsi.read_region(coord, self.patch_level, (self.patch_size, self.patch_size)).convert('RGB')
		if self.target_patch_size is not None:
			img = img.resize(self.target_patch_size)
		img = self.roi_transforms(img).unsqueeze(0)
		return img, coord

# 用于读取csv文件中的样本id
class Dataset_All_Bags(Dataset):

	def __init__(self, csv_path):
		self.df = pd.read_csv(csv_path)
	
	def __len__(self):
		return len(self.df)

	def __getitem__(self, idx):
		return self.df['slide_id'][idx]




