import os
import time
import math
import random
import numpy as np
import h5py
import glob
import scipy.interpolate
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from scipy.interpolate import RegularGridInterpolator
import mcubes
#import mcubes_module as mc
from utils import *
from mcubes import marching_cubes #, grid_interp
#pytorch 1.2.0 implementation

def grid_interp(vol, points):
  """
  Interpolate volume data at given points
  
  Inputs:
      vol: 4D torch tensor (C, Nz, Ny, Nx)
      points: point locations (Np, 3)
  Outputs:
      output: interpolated data (Np, C)    
  """
  #vol=torch.from_numpy(vol)#.cuda()
  if vol.is_cuda:
    return mc.grid_interp_cuda(vol, points)
  else:
    return mc.grid_interp_cpu(vol, points)  #'''===
      
          
class generator(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*1, 3, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)
		nn.init.normal_(self.linear_8.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_8.bias,0)
   
	def forward(self, points, z, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
   
		#print (points.shape, z.shape)
		pointz = torch.cat([points,zs],2)

		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		l7 = self.linear_7(l6)
		l8 = self.linear_8(l6)
   
		#l7 = torch.clamp(l7, min=0, max=1)
		l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l8 = torch.max(torch.min(l8, l8*0+1), l8*0) 
		#for i in range(4096):
		# #print ('l8',l8[0,i,:])
		return l7

class generator_color(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator_color, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim, self.gf_dim*8, bias=True)
		self.linear_2 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_3 = nn.Linear(self.gf_dim*8, self.gf_dim*8, bias=True)
		self.linear_4 = nn.Linear(self.gf_dim*8, self.gf_dim*4, bias=True)
		self.linear_5 = nn.Linear(self.gf_dim*4, self.gf_dim*2, bias=True)
		self.linear_6 = nn.Linear(self.gf_dim*2, self.gf_dim*1, bias=True)
		self.linear_7 = nn.Linear(self.gf_dim*1, 1, bias=True)
		self.linear_8 = nn.Linear(self.gf_dim*1, 3, bias=True)
		nn.init.normal_(self.linear_1.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_1.bias,0)
		nn.init.normal_(self.linear_2.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_2.bias,0)
		nn.init.normal_(self.linear_3.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_3.bias,0)
		nn.init.normal_(self.linear_4.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_4.bias,0)
		nn.init.normal_(self.linear_5.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_5.bias,0)
		nn.init.normal_(self.linear_6.weight, mean=0.0, std=0.02)
		nn.init.constant_(self.linear_6.bias,0)
		nn.init.normal_(self.linear_7.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_7.bias,0)
		nn.init.normal_(self.linear_8.weight, mean=1e-5, std=0.02)
		nn.init.constant_(self.linear_8.bias,0)
   
	def forward(self, points, z, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		pointz = torch.cat([points,zs],2)

		l1 = self.linear_1(pointz)
		l1 = F.leaky_relu(l1, negative_slope=0.02, inplace=True)

		l2 = self.linear_2(l1)
		l2 = F.leaky_relu(l2, negative_slope=0.02, inplace=True)

		l3 = self.linear_3(l2)
		l3 = F.leaky_relu(l3, negative_slope=0.02, inplace=True)

		l4 = self.linear_4(l3)
		l4 = F.leaky_relu(l4, negative_slope=0.02, inplace=True)

		l5 = self.linear_5(l4)
		l5 = F.leaky_relu(l5, negative_slope=0.02, inplace=True)

		l6 = self.linear_6(l5)
		l6 = F.leaky_relu(l6, negative_slope=0.02, inplace=True)

		#l7 = self.linear_7(l6)
		l8 = self.linear_8(l6)
   
		#l7 = torch.clamp(l7, min=0, max=1)
		#l7 = torch.max(torch.min(l7, l7*0.01+0.99), l7*0.01)
		l8 = torch.max(torch.min(l8, l8*0+1), l8*0) 
		#for i in range(4096):
		# #print ('l8',l8[0,i,:])
		return l8



class encoder(nn.Module):
	def __init__(self, ef_dim, z_dim):
		super(encoder, self).__init__()
		self.ef_dim = ef_dim
		self.z_dim = z_dim
		self.conv_1 = nn.Conv3d(1+3, self.ef_dim, 4, stride=2, padding=1, bias=False)
		self.in_1 = nn.InstanceNorm3d(self.ef_dim)
		self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 4, stride=2, padding=1, bias=False)
		self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
		self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 4, stride=2, padding=1, bias=False)
		self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
		self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 4, stride=2, padding=1, bias=False)
		self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
		self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		self.conv_6 = nn.Conv3d(self.ef_dim*8, self.z_dim, 4, stride=1, padding=0, bias=True)
		nn.init.xavier_uniform_(self.conv_1.weight)
		nn.init.xavier_uniform_(self.conv_2.weight)
		nn.init.xavier_uniform_(self.conv_3.weight)
		nn.init.xavier_uniform_(self.conv_4.weight)
		nn.init.xavier_uniform_(self.conv_5.weight)
		nn.init.constant_(self.conv_5.bias,0)
		nn.init.xavier_uniform_(self.conv_6.weight)
		nn.init.constant_(self.conv_6.bias,0)


	def forward(self, inputs, is_training=False):
		#print ('input',inputs.shape)
		d_1 = self.in_1(self.conv_1(inputs))
		d_1 = F.leaky_relu(d_1, negative_slope=0.02, inplace=True)

		d_2 = self.in_2(self.conv_2(d_1))
		d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True)
		
		d_3 = self.in_3(self.conv_3(d_2))
		d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)

		d_4 = self.in_4(self.conv_4(d_3))
		d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True)

		d_5 = self.conv_5(d_4)
		d_5 = d_5.view(-1, self.z_dim)
		d_5 = torch.sigmoid(d_5)

		d_6 = self.conv_6(d_4)
		d_6 = d_6.view(-1, self.z_dim)
		d_6 = torch.sigmoid(d_6)


		return d_5, d_6


class im_network(nn.Module):
	def __init__(self, ef_dim, gf_dim, z_dim, point_dim):
		super(im_network, self).__init__()
		self.ef_dim = ef_dim
		self.gf_dim = gf_dim
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.encoder = encoder(self.ef_dim, self.z_dim)
		self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)
		self.generator_color = generator_color(self.z_dim, self.point_dim, self.gf_dim)
	def forward(self, inputs, z_vector, z_vector_color, point_coord, is_training=False):
		if is_training:

			z_vector, z_vector_color = self.encoder(inputs, is_training=is_training)

			net_out = self.generator(point_coord, z_vector, is_training=is_training)
			net_out_color = self.generator_color(point_coord, z_vector_color, is_training=is_training)
		else:
			if inputs is not None:
				z_vector, z_vector_color = self.encoder(inputs, is_training=is_training)
			if z_vector is not None and point_coord is not None:
				net_out = self.generator(point_coord, z_vector, is_training=is_training)
				net_out_color = self.generator_color(point_coord, z_vector_color, is_training=is_training)
			else:
				net_out = None
				net_out_color = None

		return z_vector, z_vector_color, net_out, net_out_color


class IM_AE(object):
	def __init__(self, config):
		#progressive training
		#1-- (16, 16*16*16)
		#2-- (32, 16*16*16)
		#3-- (64, 16*16*16*4)
		self.sample_vox_size = config.sample_vox_size
		print (self.sample_vox_size)
		if self.sample_vox_size==16:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 64
		elif self.sample_vox_size==32:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 64
		elif self.sample_vox_size==64:
			self.load_point_batch_size = 16*16*16*4
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 28
		self.input_size = 64 #input voxel grid size

		self.ef_dim = 32
		self.gf_dim = 128
		self.z_dim = 256
		self.point_dim = 3

		self.dataset_name = config.dataset
		#self.dataset_load = self.dataset_name + '_train'
		self.data_paths=glob.glob('../hdf5_train_new/*.hdf5')
		for i in range(32):
		  self.data_paths.append(self.data_paths[0])
		#self.data_paths=[]
		#path1=glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5_train/*.hdf5')#[0]
		#for i in range(320):
		# self.data_paths.append(path1)
		if not (config.train):# or config.getz):
			self.data_paths=glob.glob('../hdf5_test_new/*.hdf5')
			self.data_paths.sort()
			#self.dataset_load = self.dataset_name + '_test'
		self.checkpoint_dir = config.checkpoint_dir
		self.data_dir = config.data_dir
		
   
		#data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
		#self.data_paths=glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5/*.hdf5')
		#print ('data name lzz',data_hdf5_name)
		'''if not (config.train or config.getz):
			self.dataset_load = self.dataset_name + '_test'
			data_hdf5_name = self.data_dir+'/'+self.dataset_load+'.hdf5'
			data_dict = h5py.File(data_hdf5_name, 'r')
			print ('load')
			self.data_points = (data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5
			self.data_values = data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32)
			self.data_colors = data_dict['colors_'+str(self.sample_vox_size)][:].astype(np.float32)/255.0
			self.data_voxels = data_dict['voxels'][:]
			self.data_voxels_colors = data_dict['voxels_colors'][:]/255.0
			self.data_voxels_colors = np.transpose(self.data_voxels_colors, (0,4,1,2,3))
			self.data_voxels_colors = np.reshape(self.data_voxels_colors, [-1,3,self.input_size,self.input_size,self.input_size])
			#reshape to NCHW
			self.data_voxels = np.reshape(self.data_voxels, [-1,1,self.input_size,self.input_size,self.input_size])
		#else:
		#	print("error: cannot load "+data_hdf5_name)
		#	exit(0)'''

		#print ('loaded')
		if torch.cuda.is_available():
			self.device = torch.device('cuda')
			torch.backends.cudnn.benchmark = True
		else:
			self.device = torch.device('cpu')

		#build model
		self.im_network = im_network(self.ef_dim, self.gf_dim, self.z_dim, self.point_dim)
		self.im_network.to(self.device)
		#print params
		#for param_tensor in self.im_network.state_dict():
		#	print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
		self.optimizer = torch.optim.Adam(self.im_network.parameters(), lr=config.learning_rate, betas=(config.beta1, 0.999))
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 2
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='IM_AE.model'
		self.checkpoint_manager_list = [None] * self.max_to_keep
		self.checkpoint_manager_pointer = 0
		#loss
		def network_loss(G,point_value):
			return torch.mean((G-point_value)**2)
		self.loss = network_loss

		def color_loss(G,point_color,mask):
			return torch.mean(((G-point_color)*mask)**2)
		self.color_loss = color_loss

		#keep everything a power of 2
		self.cell_grid_size = 4
		self.frame_grid_size = 64
		self.real_size = self.cell_grid_size*self.frame_grid_size #=256, output point-value voxel grid size in testing
		self.test_size = 32 #related to testing batch_size, adjust according to gpu memory size
		self.test_point_batch_size = self.test_size*self.test_size*self.test_size #do not change

		#get coords for training
		dima = self.test_size
		dim = self.frame_grid_size
		self.aux_x = np.zeros([dima,dima,dima],np.uint8)
		self.aux_y = np.zeros([dima,dima,dima],np.uint8)
		self.aux_z = np.zeros([dima,dima,dima],np.uint8)
		multiplier = int(dim/dima)
		multiplier2 = multiplier*multiplier
		multiplier3 = multiplier*multiplier*multiplier
		for i in range(dima):
			for j in range(dima):
				for k in range(dima):
					self.aux_x[i,j,k] = i*multiplier
					self.aux_y[i,j,k] = j*multiplier
					self.aux_z[i,j,k] = k*multiplier
		self.coords = np.zeros([multiplier3,dima,dima,dima,3],np.float32)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,0] = self.aux_x+i
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,1] = self.aux_y+j
					self.coords[i*multiplier2+j*multiplier+k,:,:,:,2] = self.aux_z+k
		self.coords = (self.coords.astype(np.float32)+0.5)/dim-0.5
		self.coords = np.reshape(self.coords,[multiplier3,self.test_point_batch_size,3])
		self.coords = torch.from_numpy(self.coords)
		self.coords = self.coords.to(self.device)
		

		#get coords for testing
		dimc = self.cell_grid_size
		dimf = self.frame_grid_size
		self.cell_x = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_y = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_z = np.zeros([dimc,dimc,dimc],np.int32)
		self.cell_coords = np.zeros([dimf,dimf,dimf,dimc,dimc,dimc,3],np.float32)
		self.frame_coords = np.zeros([dimf,dimf,dimf,3],np.float32)
		self.frame_x = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_y = np.zeros([dimf,dimf,dimf],np.int32)
		self.frame_z = np.zeros([dimf,dimf,dimf],np.int32)
		for i in range(dimc):
			for j in range(dimc):
				for k in range(dimc):
					self.cell_x[i,j,k] = i
					self.cell_y[i,j,k] = j
					self.cell_z[i,j,k] = k
		for i in range(dimf):
			for j in range(dimf):
				for k in range(dimf):
					self.cell_coords[i,j,k,:,:,:,0] = self.cell_x+i*dimc
					self.cell_coords[i,j,k,:,:,:,1] = self.cell_y+j*dimc
					self.cell_coords[i,j,k,:,:,:,2] = self.cell_z+k*dimc
					self.frame_coords[i,j,k,0] = i
					self.frame_coords[i,j,k,1] = j
					self.frame_coords[i,j,k,2] = k
					self.frame_x[i,j,k] = i
					self.frame_y[i,j,k] = j
					self.frame_z[i,j,k] = k
		self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.real_size-0.5
		self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
		self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
		self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
		self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
		self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
		self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
		self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
		self.frame_coords = (self.frame_coords.astype(np.float32)+0.5)/dimf-0.5
		self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])
		
		self.sampling_threshold = 0.5 #final marching cubes threshold

	@property
	def model_dir(self):
		return "{}_ae_{}".format(self.dataset_name, self.input_size)

	def train(self, config):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		print (checkpoint_txt)
		if self.sample_vox_size==32: #os.path.exists(checkpoint_txt):
			model_dir= config.initialize #'checkpoint/color_all_ae_64/IM_AE.model64-199_raw.pth' #'/mnt/sda/lzz/merge-nocyclic-multi-att-ori/checkpoint/color_all_ae_64/IM_AE.model64-199.pth'
			self.im_network.load_state_dict(torch.load(model_dir),strict=False)
		else:
			print(" [!] Load failed...")
			
		shape_num = len(self.data_paths)

		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		point_batch_num = int(self.load_point_batch_size/self.point_batch_size)
		#print (batch_index_list, 'batch index')
		for epoch in range(0, training_epoch): #int(model_dir.split('/')[-1].split('-')[-1].split('.')[0]), training_epoch):
			self.im_network.train()
			np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_loss_color = 0
			avg_loss_value = 0
			avg_num = 0
			self.data_points=np.zeros((self.shape_batch_size,self.load_point_batch_size,3))
			self.data_values=np.zeros((self.shape_batch_size,self.load_point_batch_size,1))
			self.data_colors=np.zeros((self.shape_batch_size,self.load_point_batch_size,3))
			self.data_voxels=np.zeros((self.shape_batch_size,1,64,64,64))
			self.data_voxels_colors=np.zeros((self.shape_batch_size,3,64,64,64))
      
			for idx in range(batch_num):
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]

				self.data_points[:]=0
				self.data_values[:]=0
				self.data_colors[:]=0
				self.data_voxels[:]=0
				self.data_voxels_colors[:]=0
				batch_paths=np.asarray(self.data_paths)[dxb]

				for b in range(batch_paths.shape[0]): #path in batch_paths:
				 path=batch_paths[b]
				 #print (path)
				 data_dict = h5py.File(path, 'r')
				 chosen=random.sample(range(0,data_dict['points_'+str(self.sample_vox_size)].shape[1]), self.data_points.shape[1])
				 self.data_points[b,:,:]=((data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5)[:,chosen,:]
				 self.data_values[b,:,:]=(data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32))[:,chosen,:]
				 self.data_colors[b,:,:]=(data_dict['colors_'+str(self.sample_vox_size)][:].astype(np.float32)/255.0)[:,chosen,:]
                                                                                                              
				 tmp_data_voxels_colors = data_dict['voxels_colors'][:]/255.0

				 tmp_data_voxels_colors = np.transpose(tmp_data_voxels_colors, (0,4,1,2,3))

				 self.data_voxels_colors[b,:,:,:,:]=(np.reshape(tmp_data_voxels_colors, [1,3,self.input_size,self.input_size,self.input_size]))
				 self.data_voxels[b,:,:,:,:]=(np.reshape(data_dict['voxels'][:], [1,1,self.input_size,self.input_size,self.input_size]))

				batch_voxels = self.data_voxels.astype(np.float32) #[dxb].astype(np.float32)
				batch_voxels_colors = self.data_voxels_colors.astype(np.float32)  # [dxb].astype(np.float32)
				if point_batch_num==1:
					point_coord = self.data_points#[dxb]
					point_value = self.data_values#[dxb]
					point_color = self.data_colors#[dxb]
				else:
					which_batch = 0 #np.random.randint(point_batch_num)
					point_coord = self.data_points[which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size] #[dxb][which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
					point_value = self.data_values[which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]#[dxb][which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
					point_color = self.data_colors[which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]#[dxb][which_batch*self.point_batch_size:(which_batch+1)*self.point_batch_size]
                      
				batch_voxels = torch.from_numpy(batch_voxels).float()
				batch_voxels_colors = torch.from_numpy(batch_voxels_colors).float()
				point_coord = torch.from_numpy(point_coord).float()
				point_value = torch.from_numpy(point_value).float()
				point_color = torch.from_numpy(point_color).float()
        
				batch_voxels = batch_voxels.to(self.device)
				batch_voxels_colors = batch_voxels_colors.to(self.device)
				point_coord = point_coord.to(self.device)
				point_value = point_value.to(self.device)
				point_color = point_color.to(self.device)
				#point_color[torch.where(point_color==0)]=1
        
				self.im_network.zero_grad()
        

				_,_, net_out, net_out_color = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, point_coord, is_training=True)
				errSP_value = self.loss(net_out, point_value)

				point_value3=point_value.repeat(1,1,3)


				errSP_color = self.color_loss(net_out_color, point_color, point_value3)*10.0
				errSP=errSP_value+errSP_color

				errSP.backward()
				self.optimizer.step()
				avg_loss_value += errSP_value.item()
				avg_loss_color += errSP_color.item()
				avg_loss_sp += errSP.item()
				avg_num += 1
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f, loss_value_sp: %.6f, loss_color_sp: %.6f, loss_sp: %.6f" % (epoch, training_epoch, time.time() - start_time, avg_loss_value/avg_num, avg_loss_color/avg_num, avg_loss_sp/avg_num))
			#if epoch%10==9:
			#	self.test_1(config,"train_"+str(self.sample_vox_size)+"_"+str(epoch))
			if epoch%20==19:
				if not os.path.exists(self.checkpoint_path):
					os.makedirs(self.checkpoint_path)
				save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(epoch)+"_raw.pth")
				self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
				#delete checkpoint
				if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
					if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
						os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
				#save checkpoint
				torch.save(self.im_network.state_dict(), save_dir)
				#update checkpoint manager
				self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
				#write file
				checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
				fout = open(checkpoint_txt, 'w')
				for i in range(self.max_to_keep):
					pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
					if self.checkpoint_manager_list[pointer] is not None:
						fout.write(self.checkpoint_manager_list[pointer]+"\n")
				fout.close()

		if not os.path.exists(self.checkpoint_path):
			os.makedirs(self.checkpoint_path)
		save_dir = os.path.join(self.checkpoint_path,self.checkpoint_name+str(self.sample_vox_size)+"-"+str(epoch)+".pth")
		self.checkpoint_manager_pointer = (self.checkpoint_manager_pointer+1)%self.max_to_keep
		#delete checkpoint
		#if self.checkpoint_manager_list[self.checkpoint_manager_pointer] is not None:
		#	if os.path.exists(self.checkpoint_manager_list[self.checkpoint_manager_pointer]):
		#		os.remove(self.checkpoint_manager_list[self.checkpoint_manager_pointer])
		#save checkpoint
		torch.save(self.im_network.state_dict(), save_dir)
		#update checkpoint manager
		self.checkpoint_manager_list[self.checkpoint_manager_pointer] = save_dir
		#write file
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		fout = open(checkpoint_txt, 'w')
		for i in range(self.max_to_keep):
			pointer = (self.checkpoint_manager_pointer+self.max_to_keep-i)%self.max_to_keep
			if self.checkpoint_manager_list[pointer] is not None:
				fout.write(self.checkpoint_manager_list[pointer]+"\n")
		fout.close()

	def test_1(self, config, name):
		multiplier = int(self.frame_grid_size/self.test_size)
		multiplier2 = multiplier*multiplier
		self.im_network.eval()
		t = np.random.randint(len(self.data_voxels))
		model_float = np.zeros([self.frame_grid_size+2,self.frame_grid_size+2,self.frame_grid_size+2],np.float32)
		batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
		batch_voxels = torch.from_numpy(batch_voxels)
		batch_voxels = batch_voxels.to(self.device)
   
   
   
		batch_voxels_colors = self.data_voxels_colors[t:t+1].astype(np.float32)
		batch_voxels_colors = torch.from_numpy(batch_voxels_colors)
		batch_voxels_colors = batch_voxels_colors.to(self.device)
        
        
		z_vector,z_vector_color, _, _ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, None, is_training=False)
		for i in range(multiplier):
			for j in range(multiplier):
				for k in range(multiplier):
					minib = i*multiplier2+j*multiplier+k
					point_coord = self.coords[minib:minib+1]
					_,_, net_out, net_out_color = self.im_network(None, z_vector,z_vector_color, point_coord, is_training=False)
					#net_out = torch.clamp(net_out, min=0, max=1)
					model_float[self.aux_x+i+1,self.aux_y+j+1,self.aux_z+k+1] = np.reshape(net_out.detach().cpu().numpy(), [self.test_size,self.test_size,self.test_size])
		
		vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
		vertices = (vertices.astype(np.float32)-0.5)/self.frame_grid_size-0.5
		#output ply sum
		#write_ply_triangle(config.sample_dir+"/"+name+".ply", vertices, triangles)
		#print("[sample]")



	def z2voxel(self, z, z_color):
		color_cube_float = np.zeros([3, self.real_size+2,self.real_size+2,self.real_size+2],np.float32)  #258*258*258
		model_float = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)  #258*258*258
		conf = np.zeros([self.real_size+2,self.real_size+2,self.real_size+2],np.float32)
		#print (model_float.shape)
		dimc = self.cell_grid_size  #4
		dimf = self.frame_grid_size   #64
		
		frame_flag = np.zeros([dimf+2,dimf+2,dimf+2],np.uint8)
		color_cube = np.ones([3,dimf+2,dimf+2,dimf+2]).astype('float32')

		queue = []
		
		frame_batch_num = int(dimf**3/self.test_point_batch_size)  #8
		assert frame_batch_num>0
   
		#print (dimf #64, dimf**3,262144, self.test_point_batch_size, 32768 , frame_batch_num 8)
		
		#get frame grid values
		for i in range(frame_batch_num):
			point_coord = self.frame_coords[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			point_coord = np.expand_dims(point_coord, axis=0)
			point_coord = torch.from_numpy(point_coord)
			point_coord = point_coord.to(self.device)
			_,_, model_out_, color_out_ = self.im_network(None, z, z_color, point_coord, is_training=False)  
			#print ('cube 0',torch.unique(color_out_.detach())) 
			#print ('model out', model_out_.shape, color_out_.shape)  torch.Size([1, 32768, 1]) torch.Size([1, 32768, 3])
			model_out = model_out_.detach().cpu().numpy()[0]
			color_out_ = color_out_.detach().cpu().numpy()[0]
			#print (color_out_.shape)
			color_out = np.transpose(color_out_,(1,0))
			x_coords = self.frame_x[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			y_coords = self.frame_y[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			z_coords = self.frame_z[i*self.test_point_batch_size:(i+1)*self.test_point_batch_size]
			#print (frame_flag.shape, x_coords,y_coords,z_coords, x_coords+1, y_coords+1,z_coords+1)
			#print (model_out.shape, color_out.shape, self.test_point_batch_size, color_flag[:,x_coords,y_coords,z_coords].shape) (32768, 1) (32768, 3) 32768 (3, 32768)
			frame_flag[x_coords+1,y_coords+1,z_coords+1] = np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size]) #66,66,66
			conf[x_coords+1,y_coords+1,z_coords+1] = np.reshape(model_out.astype(float), [self.test_point_batch_size])
			color_cube[:,x_coords+1,y_coords+1,z_coords+1] = np.reshape(color_out, [3, self.test_point_batch_size]) #66,66,66
			#print (x_coords,y_coords,z_coords,x_coords.shape,y_coords.shape,z_coords.shape)
			#print ('cube 1',color_out.shape, np.reshape((model_out>self.sampling_threshold).astype(np.uint8), [self.test_point_batch_size]).shape, np.reshape(color_out, [3, self.test_point_batch_size]).shape, np.unique(color_cube), color_cube[:,x_coords,y_coords,z_coords].shape, frame_flag[x_coords+1,y_coords+1,z_coords+1].shape)
		#get queue and fill up ones
		for i in range(1,dimf+1):
			for j in range(1,dimf+1):
				for k in range(1,dimf+1):
        
					                                                      
					x_coords = self.cell_x+(i-1)*dimc
					#print ('xcorrds',x_coords,self.cell_x, i-1, dimc)
					#print ('cellx,dimc',self.cell_x, dimc)   cellx,dimc [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3] 4
					y_coords = self.cell_y+(j-1)*dimc
					z_coords = self.cell_z+(k-1)*dimc
					#model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
					#print (color_cube[:,i,j,k].shape, color_cube_float[:,x_coords+1,y_coords+1,z_coords+1])
					color_cube_float[0,x_coords+1,y_coords+1,z_coords+1] =  color_cube[0,i,j,k]
					color_cube_float[1,x_coords+1,y_coords+1,z_coords+1] =  color_cube[1,i,j,k]
					color_cube_float[2,x_coords+1,y_coords+1,z_coords+1] =  color_cube[2,i,j,k]
					#print (i,j,k,color_cube[0,i,j,k]*255,color_cube[1,i,j,k]*255,color_cube[2,i,j,k]*255)
                                
					maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					if maxv!=minv:
						queue.append((i,j,k))
					elif maxv==1:
						x_coords = self.cell_x+(i-1)*dimc
						#print ('xcorrds',x_coords,self.cell_x, i-1, dimc)
						#print ('cellx,dimc',self.cell_x, dimc)   cellx,dimc [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3] 4
						y_coords = self.cell_y+(j-1)*dimc
						z_coords = self.cell_z+(k-1)*dimc
						model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0
						#print (color_cube[:,i,j,k].shape, color_cube_float[:,x_coords+1,y_coords+1,z_coords+1])
						#color_cube_float[0,x_coords+1,y_coords+1,z_coords+1] =  color_cube[0,i,j,k]
						#color_cube_float[1,x_coords+1,y_coords+1,z_coords+1] =  color_cube[1,i,j,k]
						#color_cube_float[2,x_coords+1,y_coords+1,z_coords+1] =  color_cube[2,i,j,k]
						#print ('c',color_cube[:,i,j,k], color_cube[:,i,j,k].shape)
                                  
                                                                    
		cell_batch_size = dimc**3
		cell_batch_num = int(self.test_point_batch_size/cell_batch_size)
		assert cell_batch_num>0
		#run queue
		while len(queue)>0:
			batch_num = min(len(queue),cell_batch_num)
			point_list = []
			cell_coords = []
			for i in range(batch_num):
				point = queue.pop(0)
				point_list.append(point)
				cell_coords.append(self.cell_coords[point[0]-1,point[1]-1,point[2]-1])
			cell_coords = np.concatenate(cell_coords, axis=0)
			cell_coords = np.expand_dims(cell_coords, axis=0)
			cell_coords = torch.from_numpy(cell_coords)
			cell_coords = cell_coords.to(self.device)
			_,_, model_out_batch_, color_out_batch_ = self.im_network(None, z,z_color, cell_coords, is_training=False)
			model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
			color_out_batch = color_out_batch_.detach().cpu().numpy()[0]
			for i in range(batch_num):
				point = point_list[i]
				#print (model_out_batch.shape, color_out_batch.shape)
				model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
				#color_out = color_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,:]
        
				#print ('color out',color_out.shape)
				
				x_coords = self.cell_x+(point[0]-1)*dimc
				y_coords = self.cell_y+(point[1]-1)*dimc
				z_coords = self.cell_z+(point[2]-1)*dimc
				model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
				#for c in range(3):                            
				#  color_cube_float[c,x_coords+1,y_coords+1,z_coords+1] =  color_out[:,c]
                          
				if np.max(model_out)>self.sampling_threshold:
					for i in range(-1,2):
						pi = point[0]+i
						if pi<=0 or pi>dimf: continue
						for j in range(-1,2):
							pj = point[1]+j
							if pj<=0 or pj>dimf: continue
							for k in range(-1,2):
								pk = point[2]+k
								if pk<=0 or pk>dimf: continue
								if (frame_flag[pi,pj,pk] == 0):
									frame_flag[pi,pj,pk] = 1
									queue.append((pi,pj,pk))
		return model_float, color_cube_float, color_cube
	
	#may introduce foldovers
	def optimize_mesh(self, vertices, z,z_color, iteration = 3):
		new_vertices = np.copy(vertices)

		new_vertices_ = np.expand_dims(new_vertices, axis=0)
		new_vertices_ = torch.from_numpy(new_vertices_)
		new_vertices_ = new_vertices_.to(self.device)
		_,_, new_v_out_,_= self.im_network(None, z,z_color, new_vertices_, is_training=False)
		new_v_out = new_v_out_.detach().cpu().numpy()[0]
		
		for iter in range(iteration):
			for i in [-1,0,1]:
				for j in [-1,0,1]:
					for k in [-1,0,1]:
						if i==0 and j==0 and k==0: continue
						offset = np.array([[i,j,k]],np.float32)/(self.real_size*6*2**iter)
						current_vertices = vertices+offset
						current_vertices_ = np.expand_dims(current_vertices, axis=0)
						current_vertices_ = torch.from_numpy(current_vertices_)
						current_vertices_ = current_vertices_.to(self.device)
						_, _,current_v_out_,_ = self.im_network(None, z,z_color, current_vertices_, is_training=False)
						current_v_out = current_v_out_.detach().cpu().numpy()[0]
						keep_flag = abs(current_v_out-self.sampling_threshold)<abs(new_v_out-self.sampling_threshold)
						keep_flag = keep_flag.astype(np.float32)
						new_vertices = current_vertices*keep_flag+new_vertices*(1-keep_flag)
						new_v_out = current_v_out*keep_flag+new_v_out*(1-keep_flag)
			vertices = new_vertices
		
		return vertices





	#output shape as ply and point cloud as ply
	def test_mesh_point(self, config):
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			model_dir='checkpoint/color_all_ae_64/IM_AE.model32-499.pth'
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS", model_dir)
      
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		#for t in range(config.start, min(len(self.data_voxels),config.end)):
   
		tables=[]
		chairs=[]
    
    
		import csv

   
		idx=0
        
        
		self.data_paths.sort()
		for path in self.data_paths[config.start:config.end]:


			#path=glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5_test/cd942*')[0]
			name=path.split('/')[-1]
			#if '52b66' not in name:
			#  continue		 
			#if name.split('.')[0] not in table_chairs[table_chair]:
			#  continue
			idx+=1
               
               

			data_dict = h5py.File(path, 'r')
			self.data_points=((data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5)
			self.data_values=(data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32))
			self.data_colors=(data_dict['colors_'+str(self.sample_vox_size)][:].astype(np.float32)/255.0)
				 
				 
			tmp_data_voxels_colors = data_dict['voxels_colors'][:]/255.0
			tmp_data_voxels_colors = np.transpose(tmp_data_voxels_colors, (0,4,1,2,3))
			self.data_voxels_colors=(np.reshape(tmp_data_voxels_colors, [-1,3,self.input_size,self.input_size,self.input_size]))
			self.data_voxels=(np.reshape(data_dict['voxels'][:], [-1,1,self.input_size,self.input_size,self.input_size]))
                                                                     

			t=0
			batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
			batch_voxels = torch.from_numpy(batch_voxels_)
			batch_voxels = batch_voxels.to(self.device)
      
      
      
         
   
			batch_voxels_colors = self.data_voxels_colors[t:t+1].astype(np.float32)
			batch_voxels_colors = torch.from_numpy(batch_voxels_colors)
			batch_voxels_colors = batch_voxels_colors.to(self.device)
			#print (torch.unique(batch_voxels_colors))
        
        
			#z_vector, _, _ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None, None, is_training=False)
			model_z,z_vector_color,_,_ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, None, is_training=False)
      
			'''z=model_z.detach.detach().cpu().numpy()
			np.save('../feat32/'+name+'.npy',z)
			continue'''
      
			model_float, color_cube_float, color_cube = self.z2voxel(model_z, z_vector_color)
      
      


      
			
			from plyfile import PlyData,PlyElement
			#print (color_cube.shape,'color cube',model_float.shape,np.unique(color_cube))
			some_array=[]
			size=258
			for i in range(0,64,1):
			  for j in range(0,64,1):
			    for k in range(0,64,1):
			      if model_float[1:-1,1:-1,1:-1][i*4,j*4,k*4]>0.5:
			       some_array.append((i,j,k,min(255,color_cube[:,1:-1,1:-1,1:-1][2,int(i),int(j),int(k)]*255),min(255,color_cube[:,1:-1,1:-1,1:-1][1,int(i),int(j),int(k)]*255),min(255,color_cube[:,1:-1,1:-1,1:-1][0,int(i),int(j),int(k)]*255))) #255,255,255))
			       #print (i,j,k,color_cube[2,i,j,k]*255,color_cube[1,i,j,k]*255,color_cube[0,i,j,k]*255)
			       #some_array.append((i,j,k,color_cube[2,int(i/4),int(j/4),int(k/4)]*255,color_cube[1,int(i/4),int(j/4),int(k/4)]*255,color_cube[0,int(i/4),int(j/4),int(k/4)]*255))
      
      
			some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
			el = PlyElement.describe(some_array, 'vertex')
      
			PlyData([el]).write("result/ae/"+str(name)+'final.ply')






			model_pad=np.zeros((66,66,66))
			model_pad[1:-1,1:-1,1:-1]=model_float[1:-1:4,1:-1:4,1:-1:4]

			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)



			x = np.linspace(0, 257, 258)
			y = np.linspace(0, 257, 258)
			z = np.linspace(0, 257, 258)

			color_cube_float[np.where(color_cube_float<0)]=0
			color_cube_float[np.where(color_cube_float>255)]=255

			color_cube_float[:,0,:,:]=color_cube_float[:,1,:,:]
			color_cube_float[:,:,0,:]=color_cube_float[:,:,1,:]
			color_cube_float[:,:,:,0]=color_cube_float[:,:,:,1]
			color_cube_float[:,-1,:,:]=color_cube_float[:,-2,:,:]
			color_cube_float[:,:,-1,:]=color_cube_float[:,:,-2,:]
			color_cube_float[:,:,:,-1]=color_cube_float[:,:,:,-2]


			my_interpolating_function0 = RegularGridInterpolator((x, y, z), color_cube_float[0,:,:,:]) #_float[0,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function1 = RegularGridInterpolator((x, y, z), color_cube_float[1,:,:,:]) #_float[1,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function2 = RegularGridInterpolator((x, y, z), color_cube_float[2,:,:,:]) #_float[2,1:-1:4,1:-1:4,1:-1:4])
      
			color0=my_interpolating_function0(vertices)
			color1=my_interpolating_function1(vertices)
			color2=my_interpolating_function2(vertices)
      

			colors=np.zeros((color0.shape[0],3))
			colors[:,0]=color0
			colors[:,1]=color1
			colors[:,2]=color2
			write_ply_triangle("result/ae/"+str(name)+"_mesh_pred_fine.ply", vertices, triangles, colors)
      
      
      
      
      
      
      
      
			model_pad=np.zeros((66,66,66))
			model_pad[1:-1,1:-1,1:-1]=model_float[1:-1:4,1:-1:4,1:-1:4]

			vertices, triangles = mcubes.marching_cubes(model_pad, self.sampling_threshold)



			x = np.linspace(0, 65, 66)
			y = np.linspace(0, 65, 66)
			z = np.linspace(0, 65, 66)

			color_cube[np.where(color_cube<0)]=0
			color_cube[np.where(color_cube>255)]=255

			color_cube[:,0,:,:]=color_cube[:,1,:,:]
			color_cube[:,:,0,:]=color_cube[:,:,1,:]
			color_cube[:,:,:,0]=color_cube[:,:,:,1]
			color_cube[:,-1,:,:]=color_cube[:,-2,:,:]
			color_cube[:,:,-1,:]=color_cube[:,:,-2,:]
			color_cube[:,:,:,-1]=color_cube[:,:,:,-2]


			my_interpolating_function0 = RegularGridInterpolator((x, y, z), color_cube[0,:,:,:]) #_float[0,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function1 = RegularGridInterpolator((x, y, z), color_cube[1,:,:,:]) #_float[1,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function2 = RegularGridInterpolator((x, y, z), color_cube[2,:,:,:]) #_float[2,1:-1:4,1:-1:4,1:-1:4])
      
			color0=my_interpolating_function0(vertices)
			color1=my_interpolating_function1(vertices)
			color2=my_interpolating_function2(vertices)
      

			colors=np.zeros((color0.shape[0],3))
			colors[:,0]=color0
			colors[:,1]=color1
			colors[:,2]=color2
			write_ply_triangle("result/ae/"+str(name)+"_mesh_pred.ply", vertices, triangles, colors)




			sampled_points_normals = sample_points_triangle(vertices, triangles, 2048)
			vertices_tensor=torch.from_numpy(vertices.astype(np.float32)).cuda()

      
			sampled_points_normals_int=sampled_points_normals.astype('int')
			#print (sampled_points_normals.shape, np.unique(sampled_points_normals_int[:,:3]), np.unique(sampled_points_normals[:,3:] ) )
			colors=color_cube[:,sampled_points_normals_int[:,0],sampled_points_normals_int[:,1],sampled_points_normals_int[:,2]]

      
			write_ply_point_normal("result/ae/"+str(name)+"_pc.ply", sampled_points_normals, colors) #, colrs)
			



			'''model_pad=np.zeros((66,66,66))
			model_pad[1:-1,1:-1,1:-1]=self.data_voxels

			vertices, triangles = mcubes.marching_cubes(model_pad, self.sampling_threshold)



			x = np.linspace(0, 65, 66)
			y = np.linspace(0, 65, 66)
			z = np.linspace(0, 65, 66)

			colorpad=np.zeros((3,66,66,66))
			colorpad[:,1:-1,1:-1,1:-1]=self.data_voxels_colors[0,:,:,:,:]'''
			
			'''colorpad[:,:-2,1:-1,1:-1]=np.maximum(colorpad[:,:-2,1:-1,1:-1],self.data_voxels_colors[0,:,:,:,:])
			colorpad[:,2:,1:-1,1:-1]=np.maximum(colorpad[:,2:,1:-1,1:-1],self.data_voxels_colors[0,:,:,:,:])
			colorpad[:,1:-1,:-2,1:-1]=np.maximum(colorpad[:,1:-1,:-2,1:-1],self.data_voxels_colors[0,:,:,:,:])
			colorpad[:,1:-1,2:,1:-1]=np.maximum(colorpad[:,1:-1,2:,1:-1],self.data_voxels_colors[0,:,:,:,:])
			colorpad[:,1:-1,1:-1,:-2]=np.maximum(colorpad[:,1:-1,1:-1,:-2],self.data_voxels_colors[0,:,:,:,:])
			colorpad[:,1:-1,1:-1,2:]=np.maximum(colorpad[:,1:-1,1:-1,2:],self.data_voxels_colors[0,:,:,:,:])

			colorpad[:,1:-1,1:-1,1:-1][np.where(self.data_voxels_colors[0,:,:,:,:]!=0)]=self.data_voxels_colors[0,:,:,:,:][np.where(self.data_voxels_colors[0,:,:,:,:]!=0)]'''


			'''my_interpolating_function0 = RegularGridInterpolator((x, y, z), colorpad[0,:,:,:], method='nearest')
			my_interpolating_function1 = RegularGridInterpolator((x, y, z), colorpad[1,:,:,:], method='nearest')
			my_interpolating_function2 = RegularGridInterpolator((x, y, z), colorpad[2,:,:,:], method='nearest')
      
			color0=my_interpolating_function0(vertices)
			color1=my_interpolating_function1(vertices)
			color2=my_interpolating_function2(vertices)
      

			colors=np.zeros((color0.shape[0],3))
			colors[:,0]=color0
			colors[:,1]=color1
			colors[:,2]=color2
			write_ply_triangle("val/"+str(name)+"_mesh_gt.ply", vertices, triangles, colors)


			
			print("[sample]")'''
	
	def get_z(self, config):
 		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			model_dir='checkpoint/color_all_input_ae_64/IM_AE.model32-299.pth'
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS", model_dir)
      
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		#for t in range(config.start, min(len(self.data_voxels),config.end)):
   
   
   
   
		for path in self.data_paths: #[config.start:config.end]:
     
			print (path)
			name=path.split('/')[-1]
			data_dict = h5py.File(path, 'r')
			self.data_points=((data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5)
			self.data_values=(data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32))
			self.data_colors=(data_dict['colors_'+str(self.sample_vox_size)][:].astype(np.float32)/255.0)
				 
				 
			tmp_data_voxels_colors = data_dict['voxels_colors'][:]/255.0
			tmp_data_voxels_colors = np.transpose(tmp_data_voxels_colors, (0,4,1,2,3))
			self.data_voxels_colors=(np.reshape(tmp_data_voxels_colors, [-1,3,self.input_size,self.input_size,self.input_size]))
			self.data_voxels=(np.reshape(data_dict['voxels'][:], [-1,1,self.input_size,self.input_size,self.input_size]))
                                                                     

			t=0
			batch_voxels_ = self.data_voxels[t:t+1].astype(np.float32)
			batch_voxels = torch.from_numpy(batch_voxels_)
			batch_voxels = batch_voxels.to(self.device)
      
      
      
         
   
			batch_voxels_colors = self.data_voxels_colors[t:t+1].astype(np.float32)
			batch_voxels_colors = torch.from_numpy(batch_voxels_colors)
			batch_voxels_colors = batch_voxels_colors.to(self.device)
			#print (torch.unique(batch_voxels_colors))
        
        
			#z_vector, _, _ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None, None, is_training=False)
			model_z,_,_ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, None, is_training=False)
      
			z=model_z.detach().cpu().numpy()
			print (z.shape)
			np.save('../feat32_val/'+name+'.npy',z)

      
		'''#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return

		hdf5_path = self.checkpoint_dir+'/'+self.model_dir+'/'+self.dataset_name+'_train_z.hdf5'
		shape_num = len(self.data_voxels)
		hdf5_file = h5py.File(hdf5_path, mode='w')
		hdf5_file.create_dataset("zs", [shape_num,self.z_dim], np.float32)

		self.im_network.eval()
		#print(shape_num)
		for t in range(shape_num):
			batch_voxels = self.data_voxels[t:t+1].astype(np.float32)
			batch_voxels = torch.from_numpy(batch_voxels)
			batch_voxels = batch_voxels.to(self.device)
			out_z,_ ,_= self.im_network(batch_voxels, None, None, is_training=False)
			hdf5_file["zs"][t:t+1,:] = out_z.detach().cpu().numpy()

		hdf5_file.close()
		print("[z]")'''
		

	def test_z(self, config, batch_z, dim):
		could_load, checkpoint_counter = self.load(self.checkpoint_dir)
		if could_load:
			print(" [*] Load SUCCESS")
		else:
			print(" [!] Load failed...")
			return
		
		for t in range(batch_z.shape[0]):
			model_z = batch_z[t:t+1]
			model_z = torch.from_numpy(model_z)
			model_z = model_z.to(self.device)
			model_float = self.z2voxel(model_z)
			#img1 = np.clip(np.amax(model_float, axis=0)*256, 0,255).astype(np.uint8)
			#img2 = np.clip(np.amax(model_float, axis=1)*256, 0,255).astype(np.uint8)
			#img3 = np.clip(np.amax(model_float, axis=2)*256, 0,255).astype(np.uint8)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_1t.png",img1)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_2t.png",img2)
			#cv2.imwrite(config.sample_dir+"/"+str(t)+"_3t.png",img3)
      
      #print (model_float)
			
			vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)
			vertices = (vertices.astype(np.float32)-0.5)/self.real_size-0.5
			#vertices = self.optimize_mesh(vertices,model_z)
			write_ply(config.sample_dir+"/"+"out"+str(t)+".ply", vertices, triangles)
			
			print("[sample Z]")


