import os,csv
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
import mcubes as mc
from utils import *
import copy
from mcubes import marching_cubes #, grid_interp
#pytorch 1.2.0 implementation
#from dalle_pytorch import OpenAIDiscreteVAE, DALLE
#from dalle_pytorch.transformer import Transformer,Transformer_mutual
from transformers import AutoModelForSequenceClassification, AutoConfig

from pytorch_lamb import Lamb

from transformers import (
    AdamW,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
)


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
    

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.d_model)
        #add constant to embedding
        seq_len = x.size(1)
        #print ('xshape', x.shape, seq_len)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda()
        return x

def attention(q, k, v, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
    
    #print ('mask score ', mask.shape, scores.shape)
    #print ('s1',scores.shape)
    
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    #print ('s2',scores.shape)
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    output = torch.matmul(scores, v)
    #print ('output',output.shape)
    return output
    
    
    
        
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model, d_model, bias=True)
        self.v_linear = nn.Linear(d_model, d_model, bias=True)
        self.k_linear = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model, bias=True)
    
    def forward(self, q, k, v, mask=None):
        
        bs = q.size(0)
        
        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)

        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * h * sl * d_model
       
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        #print (k.shape, q.shape, v.shape, self.d_k, mask.shape)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        #print ('score',scores.shape)
        
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.d_model)
        #print ('cct',concat.shape)    
        output = self.out(concat)
    
        return output
        
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=16, dropout = 0.1):
        super().__init__() 
        # We set d_ff as a default to 2048
        
        
        self.linear_1 = nn.Linear(d_model, d_ff, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, bias=True)
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
        
        
class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-5):
        super().__init__()
    
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm



class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        #self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)
        
        #self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        #self.attn_1 = MultiHeadAttention(heads, d_model)    
        self.attn_2 = MultiHeadAttention(heads, d_model) #nn.MultiheadAttention(embed_dim=16, num_heads=4)  
        self.ff = FeedForward(d_model).cuda()
        
    def forward(self, x, e_outputs, src_mask):
        #print ('1',self.norm_2.bias)
        #x2 = self.norm_1(x)
        #x = x + self.dropout_1(self.attn_1(x2, x2, x2))  # trg_mask
        x = self.norm_2(x)
        #print ('2',torch.unique(x))
        #x=torch.transpose(x,0,1)
        #e_outputs=torch.transpose(e_outputs,0,1)
        #print ('x,e',x.shape, e_outputs.shape)
        #print (self.attn_2(x, e_outputs, e_outputs)[0].shape, x.shape)
        x = x +self.dropout_2(self.attn_2(x, e_outputs, e_outputs.clone(), src_mask))
        # x=torch.transpose(x,0,1)
        #print ('3',torch.unique(x))
        x = self.norm_3(x)
        #print ('4',torch.unique(x))
        x = x+self.dropout_3(self.ff(x))
        #print ('5',torch.unique(x))
        return x
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])





class generator(nn.Module):
	def __init__(self, z_dim, point_dim, gf_dim):
		super(generator, self).__init__()
		self.z_dim = z_dim
		self.point_dim = point_dim
		self.gf_dim = gf_dim
		d_model=32
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim+d_model, self.gf_dim*8, bias=True)
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


		self.linear_text_k = nn.Linear(768, d_model, bias=True)
		#self.linear_text_v = nn.Linear(768, d_model, bias=True)
		self.linear_shape_q = nn.Linear(259, d_model, bias=True)
		self.linear_final = nn.Linear(d_model, d_model, bias=True)
		nn.init.normal_(self.linear_text_k.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_text_k.bias,0)
		#nn.init.normal_(self.linear_text_v.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_text_v.bias,0)
		nn.init.normal_(self.linear_shape_q.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_shape_q.bias,0)
   
		self.N=4
		self.layers = get_clones(DecoderLayer(d_model, 4), self.N)
		self.pe = PositionalEncoder(d_model)
   
		'''dropout=0.1
		self.softmax=torch.nn.Softmax(1)
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.norm_3 = Norm(d_model)
        
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)
        
		#self.attn_1 = MultiHeadAttention(heads, d_model)
		self.attn_2 = MultiHeadAttention(4, d_model)
		self.ff = FeedForward(d_model).cuda()'''
   
   
	def forward(self, points, z, texts, masks, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
   
		#print (points.shape, z.shape)
		pointz = torch.cat([points,zs],2)
		#print (texts.shape, pointz.shape)
		#print (torch.unique(points),torch.unique(zs))
		linear_text_k =  self.linear_text_k(texts)
		#linear_text_v =  self.linear_text_v(texts)   
		linear_shape_q =  self.linear_shape_q(pointz.detach())
		#print (linear_text_k.shape, linear_shape_q.shape)
		'''att1=torch.einsum('btd,bsd->bts', linear_text_k, linear_shape_q)  #b, t, s
		att1=self.softmax(att1)
		position_sense_feat=torch.einsum('bts,btd->bsd', att1, linear_text_v ) '''
		#print ('pointz',torch.unique(pointz), torch.unique(texts))
		#print ('weight', torch.unique(self.linear_text_k.weight), torch.unique(self.linear_shape_q.weight))
		#print ('bias', torch.unique(self.linear_text_k.bias), torch.unique(self.linear_shape_q.bias))
		x=linear_shape_q
		src_mask=masks
		#print (masks.shape)
		'''x =  self.dropout_2(self.attn_2(linear_shape_q, linear_text_k, linear_text_v, src_mask))
		x2 = self.norm_3(x)
		x =  self.dropout_3(self.ff(x2))'''
		linear_text_k = self.pe(linear_text_k)  

		#print ('x1',torch.unique(x),self.linear_text_k.)
		#print ('linear_text_k',torch.unique(linear_text_k))
		for i in range(self.N):
		  x = self.layers[i](x, linear_text_k, src_mask)

		x=self.linear_final(x)/5.0
		#print ('pointz',torch.unique(pointz)) 
		#print ('x2',torch.unique(x))
		#print (torch.unique(pointz) ,torch.unique(x))
		#print (torch.unique(pointz),torch.unique(x))
		pointz = torch.cat([pointz, x],2)
		#print (torch.unique(position_sense_feat))
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
		d_model=32
		self.linear_1 = nn.Linear(self.z_dim+self.point_dim+d_model, self.gf_dim*8, bias=True)
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


		self.linear_text_k = nn.Linear(768, d_model, bias=True)
		#self.linear_text_v = nn.Linear(768, d_model, bias=True)
		self.linear_shape_q = nn.Linear(259, d_model, bias=True)

		self.linear_final = nn.Linear(d_model, d_model, bias=True)
   
		nn.init.normal_(self.linear_text_k.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_text_k.bias,0)
		#nn.init.normal_(self.linear_text_v.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_text_v.bias,0)
		nn.init.normal_(self.linear_shape_q.weight, mean=1e-5, std=0.02)
		#nn.init.constant_(self.linear_shape_q.bias,0)
		self.N=4
		self.layers = get_clones(DecoderLayer(d_model, 4), self.N)
		self.pe = PositionalEncoder(d_model)



		#multihead_attn = nn.MultiheadAttention(embed_dim=16, num_heads=4)

		#self.transformer_model = nn.Transformer(d_model=16, nhead=4, num_encoder_layers=0, num_decoder_layers=1, dim_feedforward=16)
		'''self.softmax=torch.nn.Softmax(1)

   
		dropout=0.1
		self.softmax=torch.nn.Softmax(1)
		self.norm_1 = Norm(d_model)
		self.norm_2 = Norm(d_model)
		self.norm_3 = Norm(d_model)
        
		self.dropout_1 = nn.Dropout(dropout)
		self.dropout_2 = nn.Dropout(dropout)
		self.dropout_3 = nn.Dropout(dropout)
        
		#self.attn_1 = MultiHeadAttention(heads, d_model)
		self.attn_2 = MultiHeadAttention(4, d_model)
		self.ff = FeedForward(d_model).cuda()'''
   
   

	def forward(self, points, z, texts, masks, is_training=False):
		zs = z.view(-1,1,self.z_dim).repeat(1,points.size()[1],1)
		pointz = torch.cat([points,zs],2)
		pointz = torch.cat([points,zs],2)
		#print (texts.shape, pointz.shape)
		#print (torch.unique(points),torch.unique(zs))
		linear_text_k =  self.linear_text_k(texts)
		#linear_text_v =  self.linear_text_v(texts)   
		linear_shape_q =  self.linear_shape_q(pointz.detach())
		#print (linear_text_k.shape, linear_shape_q.shape)
		'''att1=torch.einsum('btd,bsd->bts', linear_text_k, linear_shape_q)  #b, t, s
		att1=self.softmax(att1)
		position_sense_feat=torch.einsum('bts,btd->bsd', att1, linear_text_v ) '''
   

		x=linear_shape_q

		#linear_text_k = self.pe(linear_text_k)
		#print ('generator color',torch.unique(x))
		src_mask=masks

		for i in range(self.N):
		  x = self.layers[i](x, linear_text_k, src_mask)
		x=self.linear_final(x)/5.0
		#print ('pointz',torch.unique(pointz)) 
		#print ('x2',torch.unique(x))
		#print (torch.unique(pointz) ,torch.unique(x))
   
		#torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', custom_encoder=None, custom_decoder=None)
   
		#attn_output, attn_output_weights = multihead_attn(x, key, value)
   
		#print (x.shape,linear_text_k.shape)
		#x = self.transformer_model(torch.transpose(linear_text_k,0,1), torch.transpose(x,0,1) )
		#print (x.shape)
		#x=torch.transpose(x,0,1)
		#print (torch.unique(pointz),torch.unique(x))
		#print (masks.shape)
		'''x =self.dropout_2(self.attn_2(linear_shape_q, linear_text_k, linear_text_v, src_mask))
		x2 = self.norm_3(x)
		x = self.dropout_3(self.ff(x2))'''
		'''linear_text_k =  self.linear_text_k(texts)
		linear_text_v =  self.linear_text_v(texts)   
		linear_shape_q =  self.linear_shape_q(pointz)

		att1=torch.einsum('btd,bsd->bts', linear_text_k, linear_shape_q)  #b, t, s
		att1=self.softmax(att1)
		position_sense_feat=torch.einsum('bts,btd->bsd', att1, linear_text_v ) '''
		pointz = torch.cat([pointz, x],2)
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

		pretrained_path='bert-base-uncased'
		config = AutoConfig.from_pretrained(
		  str(pretrained_path), #num_labels=len(dataBunch.labels)
		)

		self.model = AutoModelForSequenceClassification.from_pretrained(
		  str(pretrained_path), config=config, state_dict=None
		)


		self.encoder = encoder(self.ef_dim, self.z_dim)
		self.generator = generator(self.z_dim, self.point_dim, self.gf_dim)
		self.generator_color = generator_color(self.z_dim, self.point_dim, self.gf_dim)

	def forward(self, texts, masks, inputs, z_vector, z_vector_color, z_vector_c2,out_all,point_coord, words, is_training=False):
		if texts!=None:
		  text_inputs = {
		   "input_ids": texts,
		   "attention_mask": masks,
		  }
		  

		if is_training:
			#print ('traiing')

			z_vector_std, z_vector_c_std = self.encoder(inputs, is_training=is_training)


			#net_out_color = self.generator_color(point_coord,  z_vector_c2, z_vector, is_training=is_training)
			z_vector,z_vector_color, z_vector_c2, words = self.model(**text_inputs)
			#net_out = self.generator(point_coord, z_vector, words, masks.detach(), is_training=is_training)
			#residue_color = self.generator_color(point_coord, z_vector_c2, words, masks.detach(), is_training=1)
			net_out_std = self.generator(point_coord,  z_vector_std, words, masks.detach(), is_training=is_training)
			residue_color_std= self.generator_color(point_coord, z_vector_c_std,words, masks.detach(),  is_training=1)


			return z_vector,z_vector_color, z_vector_c2,  z_vector_std,None, z_vector_c_std,net_out_std, residue_color_std, net_out_std, residue_color_std, words


	
		else:
			if texts is not None:
				z_vector,z_vector_color, z_vector_c2, words = self.model(**text_inputs)
				return z_vector, None, z_vector_c2,None, None,None,None, words

			if z_vector is not None and point_coord is not None:
				#print (point_coord.shape, z_vector.shape)
				net_out = self.generator(point_coord, z_vector, words, masks, is_training=is_training)

				net_out_color = self.generator_color(point_coord, z_vector_c2,  words, masks,   is_training=is_training)
				#print ('net out unique', torch.unique(net_out))
				return None,None,None, net_out, net_out_color, None, None #, residue_color+s1_color, s1_color


			#elif z_vector is not None and point_coord is not None:
			#  net_out = self.generator(point_coord, z_vector, is_training=is_training)
			#  return None,None,None, net_out, None,None,None,
                 
			elif (inputs is not None) and (inputs.shape[1]==4):
			  #z_vector_std, z_vector_color_std, z_vector_c2_std = self.encoder(inputs, is_training=is_training)
                        
			  z_vector_std, z_vector_c_std = self.encoder(inputs, is_training=is_training) 
			  return z_vector_std,None, z_vector_c_std,None, None,None,None  #, net_out, None,None,None,            
                            

                 
                 

class IM_res64(object):
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
			self.shape_batch_size = 32
		elif self.sample_vox_size==32:
			self.load_point_batch_size = 16*16*16
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 40
		elif self.sample_vox_size==64:
			self.load_point_batch_size = 16*16*16*4
			self.point_batch_size = 16*16*16
			self.shape_batch_size = 12
		self.input_size = 64 #input voxel grid size

		self.ef_dim = 32
		self.gf_dim = 128
		self.z_dim = 256
		self.point_dim = 3

		self.dataset_name = config.dataset
		#self.dataset_load = self.dataset_name + '_train'
		#self.data_paths=glob.glob('hdf5/*.hdf5') #/ccd5e*.hdf5')
		

		self.datas=[]

		#start=1
		with open('train_official.csv', newline='') as csvfile:
		    spamreader = csv.reader(csvfile)
		    for row in spamreader:

		      #if start==1:
		      #  start=0
		      #  continue
		      text=row[2]
		      name=row[1]
		      self.datas.append((text,name))
		      #break

		#for i in range(32):
		#  self.datas.append(self.datas[0])





		if not (config.train):# or config.getz):
			#self.data_paths=glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5/*.hdf5')


                        self.datas=[]
                        with open('test_official.csv', newline='') as csvfile:
                         spamreader = csv.reader(csvfile)
                         for row in spamreader:
                          text=row[2]
                          name=row[1]
                          text_str=row[0]
                          self.datas.append((text,name,text_str))


			#self.data_paths.sort()
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
		#for param_tensor in self.im_network.encoder.parameters():
		#	param_tensor.requires_grad=False #print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
		'''for param_tensor in self.im_network.generator.parameters():
			param_tensor.requires_grad=False #print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
		for param_tensor in self.im_network.generator_color.parameters():
			param_tensor.requires_grad=False #print(param_tensor, "\t", self.im_network.state_dict()[param_tensor].size())
		#for param_tensor in self.im_network.encoder_color.parameters():
		#	param_tensor.requires_grad=False '''









		params = list(self.im_network.generator.parameters())
		#params.extend(list(self.im_network.generator_color.parameters()))
		#params.extend(list(self.im_network.encoder.parameters()))	
		self.optimizer2 = torch.optim.Adam([{"params": self.im_network.generator.parameters()}, {"params": self.im_network.generator_color.parameters()},  {"params": self.im_network.encoder.parameters()}], lr=config.learning_rate*0.1, betas=(config.beta1, 0.999))
		self.optimizer = self.get_optimizer(0.0001, optimizer_type="lamb")
		base_params=[]
		#for param_tensor in self.im_network.encoder.parameters():
		#  base_params.append(param_tensor)
		for param_tensor in self.im_network.generator.parameters():
		  base_params.append(param_tensor)
		for param_tensor in self.im_network.generator_color.parameters():
		  base_params.append(param_tensor)

		#self.optimizer = torch.optim.Adam([{'params': base_params}, {'params': self.im_network.model.parameters(), 'lr': 0.001}], lr=config.learning_rate*1,  betas=(config.beta1, 0.999))

		#self.scheduler = self.get_scheduler(
		# self.optimizer, t_total=int(60470*config.epoch), schedule_type="warmup_cosine"
		#)
		#pytorch does not have a checkpoint manager
		#have to define it myself to manage max num of checkpoints to keep
		self.max_to_keep = 2
		self.checkpoint_path = os.path.join(self.checkpoint_dir, self.model_dir)
		self.checkpoint_name='res64.model'
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
		self.test_point_batch_size_in_training=4096
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
		self.frame_coords_train = torch.zeros([16,16,16,3]).cuda()
		self.frame_coords_train32 = torch.zeros([32,32,32,3]).cuda()
		self.frame_x = np.zeros([dimf,dimf,dimf],np.int32) #.long()
		self.frame_y = np.zeros([dimf,dimf,dimf],np.int32) #.long()
		self.frame_z = np.zeros([dimf,dimf,dimf],np.int32) #.long()
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

		for i in range(16):
			for j in range(16):
				for k in range(16):
					self.frame_coords_train[i,j,k,0] = i
					self.frame_coords_train[i,j,k,1] = j
					self.frame_coords_train[i,j,k,2] = k


		for i in range(32):
			for j in range(32):
				for k in range(32):
					self.frame_coords_train32[i,j,k,0] = i
					self.frame_coords_train32[i,j,k,1] = j
					self.frame_coords_train32[i,j,k,2] = k

		self.cell_coords = (self.cell_coords.astype(np.float32)+0.5)/self.real_size-0.5
		self.cell_coords = np.reshape(self.cell_coords,[dimf,dimf,dimf,dimc*dimc*dimc,3])
		self.cell_x = np.reshape(self.cell_x,[dimc*dimc*dimc])
		self.cell_y = np.reshape(self.cell_y,[dimc*dimc*dimc])
		self.cell_z = np.reshape(self.cell_z,[dimc*dimc*dimc])
		self.frame_x = np.reshape(self.frame_x,[dimf*dimf*dimf])
		self.frame_y = np.reshape(self.frame_y,[dimf*dimf*dimf])
		self.frame_z = np.reshape(self.frame_z,[dimf*dimf*dimf])
		self.frame_coords = (self.frame_coords+0.5)/dimf-0.5
		self.frame_coords = np.reshape(self.frame_coords,[dimf*dimf*dimf,3])
		self.frame_coords_train = (self.frame_coords_train+0.5)/16.0-0.5
		self.frame_coords_train = torch.reshape(self.frame_coords_train,[16*16*16,3])
		self.frame_coords_train32 = (self.frame_coords_train32+0.5)/32.0-0.5
		self.frame_coords_train32 = torch.reshape(self.frame_coords_train32,[32*32*32,3])
		#self.conv_edge = nn.Conv3d(3, 3, 3, stride=1, padding=1, groups=3, bias=False)
		#self.conv_edge.to(self.device)
		self.sampling_threshold = 0.5 #final marching cubes threshold
		self.upsample=nn.Upsample(scale_factor=4,mode='trilinear').cuda()
		self.upsample32=nn.Upsample(scale_factor=2,mode='trilinear').cuda()










	def get_optimizer(self, lr, optimizer_type="lamb"):

           # Prepare optimiser and schedule
           no_decay = [] #"bias", "LayerNorm.weight"]
           optimizer_grouped_parameters = [
              {
                "params": [
                    p
                    for n, p in self.im_network.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0, #self.weight_decay,
              },
           ]
           if optimizer_type == "lamb":
            optimizer = Lamb(optimizer_grouped_parameters, lr=lr, eps=1e-8)
            
           elif optimizer_type == "adamw":
            optimizer = AdamW(
                optimizer_grouped_parameters, lr=lr, eps=1e-8
            )

           return optimizer










	def get_scheduler(self, optimizer, t_total, schedule_type="warmup_cosine"):

          SCHEDULES = {
            "warmup_cosine": get_cosine_schedule_with_warmup,
          }

          if schedule_type == None or schedule_type == "none":
            return SCHEDULES[schedule_type](optimizer)

          elif schedule_type == "warmup_constant":
            return SCHEDULES[schedule_type](
                optimizer, num_warmup_steps=0 #self.warmup_steps
            )

          else:
            return SCHEDULES[schedule_type](
                optimizer,
                num_warmup_steps=0, #self.warmup_steps,
                num_training_steps=t_total,
            )




	def z2voxel(self, z, z_color, words, masks, config):
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

			_,_,_, model_out_, color_out_,_,_ = self.im_network(None,masks,None, z,None, z_color,None, point_coord, words, is_training=False)  
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
		if config.high_resolution:
			
		 for i in range(1,dimf+1):
			 for j in range(1,dimf+1):
				 for k in range(1,dimf+1):
        
					                                                      
					 x_coords = self.cell_x+(i-1)*dimc
					 y_coords = self.cell_y+(j-1)*dimc
					 z_coords = self.cell_z+(k-1)*dimc
					 color_cube_float[0,x_coords+1,y_coords+1,z_coords+1] =  color_cube[0,i,j,k]
					 color_cube_float[1,x_coords+1,y_coords+1,z_coords+1] =  color_cube[1,i,j,k]
					 color_cube_float[2,x_coords+1,y_coords+1,z_coords+1] =  color_cube[2,i,j,k]                   
					 maxv = np.max(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					 minv = np.min(frame_flag[i-1:i+2,j-1:j+2,k-1:k+2])
					 if maxv!=minv:
						 queue.append((i,j,k))
					 elif maxv==1:
						 x_coords = self.cell_x+(i-1)*dimc
						 y_coords = self.cell_y+(j-1)*dimc
						 z_coords = self.cell_z+(k-1)*dimc
						 model_float[x_coords+1,y_coords+1,z_coords+1] = 1.0

                                  
		                                                                
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

			 _,_,_, model_out_batch_, color_out_batch_,_,_ = self.im_network(None, masks,None,z,None,z_color,None, cell_coords, words, is_training=False)
			 model_out_batch = model_out_batch_.detach().cpu().numpy()[0]
			 color_out_batch = color_out_batch_.detach().cpu().numpy()[0]
			 for i in range(batch_num):
				 point = point_list[i]
				 model_out = model_out_batch[i*cell_batch_size:(i+1)*cell_batch_size,0]
				 x_coords = self.cell_x+(point[0]-1)*dimc
				 y_coords = self.cell_y+(point[1]-1)*dimc
				 z_coords = self.cell_z+(point[2]-1)*dimc
				 model_float[x_coords+1,y_coords+1,z_coords+1] = model_out
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
		return model_float, color_cube_float, frame_flag, color_cube



	@property
	def model_dir(self):
		return "{}_ae_{}".format(self.dataset_name, self.input_size)

	def train(self, config):#64: 0-100 with cyc loss gpu0, 100-200 no cyc loss gpu1 
		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		print (checkpoint_txt)
		if 1: #os.path.exists(checkpoint_txt):
			pass
			'''model_dir='checkpoint/color_all_ae_64/IM_AE.model64-239_raw.pth' #'init_32_499.pth' #$'../im-color-base/checkpoint/color_all_ae_64/IM_AE.model32-499_raw.pth' #'init.pth'
			self.im_network.load_state_dict(torch.load(model_dir),strict=False)'''

			model_dir= config.initialize #/mnt/sda/lzz/merge-cyclic-mulit-att/checkpoint/color_all_ae_64/IM_AE.model32-199.pth'  #'IM_AE.model32-189_raw.pth' #'/mnt/sda/lzz/merge-nocyclic-multi-att/checkpoint/color_all_ae_64_2/IM_AE.model32-189_raw.pth' #'checkpoint/color_all_ae_64/IM_AE.model32-199_raw.pth' #'../reg149-cg-1.pth' #'IM_AE.model32-169_raw.pth' #'/mnt/sda/lzz/merge-nocyclic-multi-att/checkpoint/color_all_ae_64_2/IM_AE.model32-189_raw.pth'
			model=torch.load(model_dir)
			'''model2={}
			for k in model.keys():
			  if 'bert' in k: #or 'generator' in k:
			    model2[k]=model[k]'''
			self.im_network.load_state_dict(model,strict=False)

			print(" [*] Load SUCCESS",model_dir)
		else:
			print(" [!] Load failed...")
			
		shape_num = len(self.datas)
		batch_index_list = np.arange(shape_num)
		
		print("\n\n----------net summary----------")
		print("training samples   ", shape_num)
		print("-------------------------------\n\n")
		start_time = time.time()
		assert config.epoch==0 or config.iteration==0
		training_epoch = config.epoch + int(config.iteration/shape_num)
		batch_num = int(shape_num/self.shape_batch_size)
		point_batch_num = int(self.load_point_batch_size/self.point_batch_size)

		for epoch in range(0, training_epoch): #0-50 no l2 norm, 50-100 l2 norm, both 2dec  #150-200 no l2 norm, and big loss  #1dec 50-100 big cyc loss, 100-150
			self.im_network.train()
			np.random.shuffle(batch_index_list)
			avg_loss_sp = 0
			avg_loss_color = 0
			avg_loss_color2 = 0
			avg_loss_value = 0

			avg_value_out =0
			avg_color_out  =0
			avg_value_out_std  =0
			avg_color_out_std =0
        
			avg_loss_value_rec =0
			avg_loss_color_rec =0
        
			avg_num = 0
			self.data_points=np.zeros((self.shape_batch_size,self.load_point_batch_size,3))
			self.data_values=np.zeros((self.shape_batch_size,self.load_point_batch_size,1))
			self.data_colors=np.zeros((self.shape_batch_size,self.load_point_batch_size,3))
			self.data_voxels=np.zeros((self.shape_batch_size,1,64,64,64))
			self.data_voxels_colors=np.zeros((self.shape_batch_size,3,64,64,64))
			#self.pred_voxels=torch.zeros((self.shape_batch_size,1,64,64,64)).to(self.device)
			#self.pred_voxels_colors=torch.zeros((self.shape_batch_size,3,64,64,64)).to(self.device)
      


			for idx in range(batch_num):
				#print (idx)
				dxb = batch_index_list[idx*self.shape_batch_size:(idx+1)*self.shape_batch_size]
				#print (dxb)
				self.data_points[:]=0
				self.data_values[:]=0
				self.data_colors[:]=0
				self.data_voxels[:]=0
				self.data_voxels_colors[:]=0
				#self.pred_voxels[:]=0
				#self.pred_voxels_colors[:]=0


				batch_paths=np.asarray(self.datas)[dxb]
				texts=np.zeros((batch_paths.shape[0], 64))
				masks=np.zeros((batch_paths.shape[0], 64))
				for b in range(batch_paths.shape[0]): #path in batch_paths:



				 text_list=batch_paths[b][0].split(' ')[:-1] #.astype('int')
				 text_array = np.asarray(list(map(int, text_list)))
				 


				 path='../hdf5_train_new/'+batch_paths[b][1]+'.hdf5'
				 name=batch_paths[b][1]
				 data_dict = h5py.File(path, 'r')

				 self.data_points[b,:,:]=((data_dict['points_'+str(self.sample_vox_size)][:].astype(np.float32)+0.5)/256-0.5)
				 self.data_values[b,:,:]=(data_dict['values_'+str(self.sample_vox_size)][:].astype(np.float32))
				 self.data_colors[b,:,:]=(data_dict['colors_'+str(self.sample_vox_size)][:].astype(np.float32)/255.0)

				 texts[b,:min(64,len(text_list))]=text_array[:min(64,len(text_list))]

				 masks[b,:min(64,len(text_list))]=1
            
				 #print (self.data_points.shape,self.data_values.shape, self.data_colors.shape)
				 
				 
				 tmp_data_voxels_colors = data_dict['voxels_colors'][:]/255.0
				 tmp_data_voxels_colors = np.transpose(tmp_data_voxels_colors, (0,4,1,2,3))
				 self.data_voxels_colors[b,:,:,:,:]=(np.reshape(tmp_data_voxels_colors, [-1,3,self.input_size,self.input_size,self.input_size]))
				 self.data_voxels[b,:,:,:,:]=(np.reshape(data_dict['voxels'][:], [-1,1,self.input_size,self.input_size,self.input_size]))
            
				
				#print ('datapoints', data_dict['points_'+str(self.sample_vox_size)].shape, self.data_points.shape)
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

				#step=1 #round(batch_voxels_colors.shape[-1]/self.sample_vox_size)
				#print (step)
				#batch_voxels_colors_16=batch_voxels_colors[:,:,0:64:step,0:64:step,0:64:step].to(self.device)
				#print ('voxels color 16',batch_voxels_colors_16.shape)
		
				point_coord = torch.from_numpy(point_coord).float()
				point_value = torch.from_numpy(point_value).float()
				point_color = torch.from_numpy(point_color).float()

				batch_voxels = batch_voxels.to(self.device)
				batch_voxels_colors = batch_voxels_colors.to(self.device)
				point_coord = point_coord.to(self.device)
				point_value = point_value.to(self.device)
				point_color = point_color.to(self.device)
				texts=torch.from_numpy(texts).to(self.device).long()
				masks=torch.from_numpy(masks).to(self.device).bool()

				self.im_network.zero_grad()



				z_vector,z_vector_color, z_vector_c2, z_vector_std, z_vector_color_std, z_vector_color2_std, net_out, residue_color, net_out_std, residue_color_std, words = self.im_network(texts,masks, torch.cat((batch_voxels,batch_voxels_colors),1), None,None,None,None, point_coord, None, is_training=True)


				

				with torch.no_grad():

				  frame_batch_num = 1 
				  point_coord = self.frame_coords_train
				  point_coord = torch.unsqueeze(point_coord, 0)
				  point_coord = point_coord.repeat(z_vector.shape[0],1,1)

        
				  _,_,_,model_out,color_final,_,_ = self.im_network(None, masks, None, z_vector, z_vector_color, z_vector_c2, None, point_coord, words, is_training=False) 
				  model_out[torch.where(model_out>self.sampling_threshold)]=1
				  model_out[torch.where(model_out<=self.sampling_threshold)]=0

				  model_out=torch.reshape(model_out, (-1,1,16,16,16))
				  pred_shape=self.upsample(model_out)  #self.pred_voxels[:]=   
				  color_final=torch.transpose(color_final,1,2)  
				  color_final=torch.reshape(color_final, (-1,3,16,16,16))
				  pred_color=self.upsample(color_final)


				  #pred_color[:,0,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=0
				  #pred_color[:,1,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=0
				  #pred_color[:,2,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=0  #gt mask 0 3, pred mask 50 7, no mask 100 5








				z_vector_rec,  z_vector_c2_rec =self.im_network.encoder(torch.cat((pred_shape, pred_color),1), is_training=False)

				'''batch_voxels_colors[:,2,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=pred_color[:,2,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]
				batch_voxels_colors[:,1,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=pred_color[:,1,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]
				batch_voxels_colors[:,0,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]=pred_color[:,0,:,:,:][torch.where(batch_voxels[:,0,:,:,:]==0)]

				origin_shape=torch.cat((batch_voxels,batch_voxels_colors),1)

				z_vector_std2,  z_vector_color2_std2 =self.im_network.encoder(origin_shape, is_training=False)'''
				#z_vector_rec=z_vector_rec.detach()
				#z_vector_c2_rec=z_vector_c2_rec.detach()


				
				errSP_value = self.loss(z_vector, z_vector_std)*1
				errSP_color2 = self.loss(z_vector_c2, z_vector_color2_std)*1.0


				#errSP_value_out = self.loss(net_out, point_value)*0.02
				point_value3_2=point_value.repeat(1,1,3)
				#errSP_color_out = self.color_loss(residue_color, point_color, point_value3_2)*1 #0.002, 0.1, gpu2, 100-150; 0.02,1, gpu0 0-50

        
				errSP_value_out_std = self.loss(net_out_std, point_value)*1
				errSP_color_out_std = self.color_loss(residue_color_std, point_color, point_value3_2)*10.0    
        
				errSP_value_rec = self.loss(z_vector_rec, z_vector_std)*0.001
				errSP_color2_rec = self.loss(z_vector_c2_rec, z_vector_color2_std)*0.0005
        
				errSP=errSP_value+  errSP_color2+   errSP_value_out_std+errSP_color_out_std +errSP_value_rec + errSP_color2_rec# +errSP_value_rec+errSP_color_rec+errSP_color2_rec +errSP_value_rec_text +errSP_color_rec_text +errSP_color2_rec_text 

				errSP.backward()
				#nn.utils.clip_grad_norm(list(self.im_network.generator_color.parameters())+list(self.im_network.dalle.parameters()) , 0.05)
				torch.nn.utils.clip_grad_norm_(
				   self.im_network.parameters(), 1
				)

				self.optimizer.step()
				self.optimizer2.step()
				avg_loss_value += errSP_value.item()
				avg_loss_color2 += errSP_color2.item()
				#avg_value_out += errSP_value_out.item()
				#avg_color_out += errSP_color_out.item()
				avg_value_out_std += errSP_value_out_std.item()
				avg_color_out_std += errSP_color_out_std.item()
				avg_loss_value_rec += errSP_value_rec.item()
				avg_loss_color_rec += errSP_color2_rec.item()
				#avg_loss_color2_rec += errSP_color2_rec.item()
				'''avg_loss_value_rec += errSP_value_rec.item()
				avg_loss_color_rec += errSP_color_rec.item()
				avg_loss_color2_rec += errSP_color2_rec.item()
				avg_loss_value_rec_text += errSP_value_rec_text.item()
				avg_loss_color_rec_text += errSP_color_rec_text.item()
				avg_loss_color2_rec_text += errSP_color2_rec_text.item()'''


				avg_loss_sp += errSP.item()
				avg_num += 1
				#print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f,loss_value_sp: %.6f, loss_color_sp: %.6f,  loss_value_out_std: %.6f,  loss_color_out_std: %.6f, loss_value_sp_rec: %.6f,  loss_color_2_rec: %.6f, loss_sp: %.6f" % (epoch, training_epoch, time.time() - start_time,avg_loss_value/avg_num, avg_loss_color2/avg_num, avg_value_out_std/avg_num, avg_color_out_std/avg_num, avg_loss_value_rec/avg_num,  avg_loss_color2_rec/avg_num, avg_loss_sp/avg_num))
			print(str(self.sample_vox_size)+" Epoch: [%2d/%2d] time: %4.4f,loss_value_sp: %.6f, loss_color_sp: %.6f,  loss_value_out_std: %.6f,  loss_color_out_std: %.6f,  loss_value_rec: %.6f,  loss_color_rec: %.6f, loss_sp: %.6f" % (epoch, training_epoch, time.time() - start_time,avg_loss_value/avg_num, avg_loss_color2/avg_num, avg_value_out_std/avg_num, avg_color_out_std/avg_num, avg_loss_value_rec/avg_num, avg_loss_color_rec/avg_num, avg_loss_sp/avg_num))

			if epoch%10==9:
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




	#output shape as ply and point cloud as ply
	def test_mesh_point(self, config):


		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if 1:
			model_dir='checkpoint/color_all_ae_64/res64.model64-199.pth'
			self.im_network.load_state_dict(torch.load(model_dir),strict=True)
			print(" [*] Load SUCCESS", model_dir)
      
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		#print (self.im_network)
		#self.im_network.model.dropout.train()
		#for t in range(config.start, min(len(self.data_voxels),config.end)):
   
   
		idx=0
		for data in self.datas[config.start:config.end]:

			text_list=data[0].split(' ')[:-1] #.astype('int')
			text_array = np.asarray(list(map(int, text_list)))
			#print (data[1])
			#if 'b0531a' not in data[1]:  #c3b6c, ad174, b4ee137
			#  continue


			path='../hdf5_test_new/'+data[1]+'.hdf5'
			data_dict = h5py.File(path, 'r')





			#path=glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5_test/cd942*')[0]
			name=path.split('/')[-1]
			#if os.path.exists("val/"+str(name)+str(data[2][:50])+"_mesh_pred.ply"):
			#  continue
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


			texts=np.zeros((1, 32))
			masks=np.zeros((1, 32))


			texts[0,:min(32,len(text_list))]=text_array[:min(32,len(text_list))]

			masks[0,:min(32,len(text_list))]=1

			texts=torch.from_numpy(texts).to(self.device).long()
			masks=torch.from_numpy(masks).to(self.device).bool()


            
	


        
			#z_vector, _, _ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None, None, is_training=False)
			#model_z,z_vector_color,out_all,_,_,_,_ = self.im_network(texts, masks, None, None, None,None, None, is_training=False) 

			#model_z, _, z_vector_c2,_,_,_,_ = self.im_network(texts, masks, None, None,None, None,None, None, is_training=False)
      
      
			model_z,_, z_vector_c2, _,_,_,_, words = self.im_network(texts,masks, None, None,None,None,None, None, None)
			#print (model_z.shape,z_vector_color.shape,'modelz')

			#z_path=glob.glob('../fast-bert-color/result_test/'+name.split('.')[0]+'*')[0]

			'''z_path=glob.glob('../fast-bert-color/result_train/'+'*')[idx]
			name=z_path.split('/')[-1]
			z_cct=np.load(z_path)
			text=z_path.split('/')[-1]
			z_cct=torch.from_numpy(z_cct).cuda()
			print (idx,z_cct.shape)
			model_z=torch.unsqueeze(z_cct[:256],0)
			z_vector_color=torch.unsqueeze(z_cct[256:],0)
			idx+=1'''
      
      
      
      
			model_float, color_cube_float, frame_flag, color_cube = self.z2voxel(model_z, z_vector_c2, words, masks, config)
      
			from plyfile import PlyData,PlyElement
			#print (color_cube.shape,'color cube',model_float.shape,np.unique(color_cube))
			some_array=[]
			size=258
			for i in range(1,64):
			  for j in range(1,64):
			    for k in range(1,64):
			      if frame_flag[1:-1,1:-1,1:-1][int(i),int(j),int(k)]>0.5:
			       some_array.append((i,j,k,color_cube[2,int(i),int(j),int(k)]*255,color_cube[1,int(i),int(j),int(k)]*255,color_cube[0,int(i),int(j),int(k)]*255))
			some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
			el = PlyElement.describe(some_array, 'vertex')
      
			PlyData([el]).write('result/res64/'+name+str(data[2][:50].replace('/',' '))+'test_new_input.ply')



			some_array=[]
			size=258
			#print (self.data_voxels.shape, self.data_voxels_colors.shape)
			for i in range(0,64):
			  for j in range(0,64):
			    for k in range(0,64):
			      if self.data_voxels[0,0,i,j,k]>0.5:
			       some_array.append((i,j,k,self.data_voxels_colors[0,2,int(i),int(j),int(k)]*255,self.data_voxels_colors[0,1,int(i),int(j),int(k)]*255,self.data_voxels_colors[0,0,int(i),int(j),int(k)]*255)) #255,255,255))
			some_array = np.array(some_array, dtype=[('x', 'float32'), ('y', 'float32'),    ('z', 'float32'),   ('red', 'uint8'),    ('green', 'uint8'),    ('blue', 'uint8')])
			el = PlyElement.describe(some_array, 'vertex')
      
			PlyData([el]).write('result/res64/'+name+str(data[2][:50].replace('/',' '))+'_gt.ply')





			model_pad=np.zeros((66,66,66))
			model_pad[1:-1,1:-1,1:-1]=frame_flag[1:-1,1:-1,1:-1] #model_float[1:-1:4,1:-1:4,1:-1:4]

			vertices, triangles = mcubes.marching_cubes(model_pad, self.sampling_threshold)



			x = np.linspace(0, 66, 66)
			y = np.linspace(0, 66, 66)
			z = np.linspace(0, 66, 66)

			#color_cube[:,1:-1,1:-1,1:-1]=color_cube
			color_cube[:,0,:,:]=color_cube[:,1,:,:]
			color_cube[:,:,0,:]=color_cube[:,:,1,:]
			color_cube[:,:,:,0]=color_cube[:,:,:,1]
			color_cube[:,-1,:,:]=color_cube[:,-2,:,:]
			color_cube[:,:,-1,:]=color_cube[:,:,-2,:]
			color_cube[:,:,:,-1]=color_cube[:,:,:,-2]
			#color_cube[:,1:-1,1:-1,1:-1]=self.data_voxels_colors[0,:,:,:,:]


			my_interpolating_function0 = RegularGridInterpolator((x, y, z), color_cube[0,:,:,:],method='nearest') #_float[0,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function1 = RegularGridInterpolator((x, y, z), color_cube[1,:,:,:],method='nearest') #_float[1,1:-1:4,1:-1:4,1:-1:4])
			my_interpolating_function2 = RegularGridInterpolator((x, y, z), color_cube[2,:,:,:],method='nearest') #_float[2,1:-1:4,1:-1:4,1:-1:4])
      
			color0=my_interpolating_function0(vertices)
			color1=my_interpolating_function1(vertices)
			color2=my_interpolating_function2(vertices)
      

			colors=np.zeros((color0.shape[0],3))
      
      			
      
			colors[:,0]=color0
			colors[:,1]=color1
			colors[:,2]=color2
			write_ply_triangle("result/res64/"+str(name)+str(data[2][:50].replace('/',' '))+"_mesh_pred.ply", vertices, triangles, colors)


			if config.high_resolution:
			 model_pad=np.zeros((258,258,258))
			 model_pad[1:-1,1:-1,1:-1]= model_float[1:-1,1:-1,1:-1] #model_float[1:-1:4,1:-1:4,1:-1:4]

			 vertices, triangles = mcubes.marching_cubes(model_float, self.sampling_threshold)



			 x = np.linspace(0, 258,258)
			 y = np.linspace(0, 258,258)
			 z = np.linspace(0, 258,258)
        
			 color_cube=color_cube_float

			 #color_cube[:,1:-1,1:-1,1:-1]=color_cube
			 color_cube[:,0,:,:]=color_cube[:,1,:,:]
			 color_cube[:,:,0,:]=color_cube[:,:,1,:]
			 color_cube[:,:,:,0]=color_cube[:,:,:,1]
			 color_cube[:,-1,:,:]=color_cube[:,-2,:,:]
			 color_cube[:,:,-1,:]=color_cube[:,:,-2,:]
			 color_cube[:,:,:,-1]=color_cube[:,:,:,-2]
			 #color_cube[:,1:-1,1:-1,1:-1]=self.data_voxels_colors[0,:,:,:,:]


			 my_interpolating_function0 = RegularGridInterpolator((x, y, z), color_cube[0,:,:,:],method='nearest') #_float[0,1:-1:4,1:-1:4,1:-1:4])
			 my_interpolating_function1 = RegularGridInterpolator((x, y, z), color_cube[1,:,:,:],method='nearest') #_float[1,1:-1:4,1:-1:4,1:-1:4])
			 my_interpolating_function2 = RegularGridInterpolator((x, y, z), color_cube[2,:,:,:],method='nearest') #_float[2,1:-1:4,1:-1:4,1:-1:4])
      
			 color0=my_interpolating_function0(vertices)
			 color1=my_interpolating_function1(vertices)
			 color2=my_interpolating_function2(vertices)
      

			 colors=np.zeros((color0.shape[0],3))
      
      			
      
			 colors[:,0]=color0
			 colors[:,1]=color1
			 colors[:,2]=color2
			 write_ply_triangle("result/res64/"+str(name)+str(data[2][:50].replace('/',' '))+"_mesh_258_"+str(idx)+".ply", vertices, triangles, colors)



			sampled_points_normals = sample_points_triangle(vertices, triangles, 2048)
			vertices_tensor=torch.from_numpy(vertices.astype(np.float32)).cuda()

      
			sampled_points_normals_int=sampled_points_normals.astype('int')
			#print (sampled_points_normals.shape, np.unique(sampled_points_normals_int[:,:3]), np.unique(sampled_points_normals[:,3:] ) )
			colors=color_cube[:,sampled_points_normals_int[:,0],sampled_points_normals_int[:,1],sampled_points_normals_int[:,2]]

      
			write_ply_point_normal("result/res64/"+str(name)+str(data[2][:50].replace('/',' '))+"_pc.ply", sampled_points_normals, colors) #, colrs)
			
			#print("[sample]")
	
	def get_z(self, config):
 		#load previous checkpoint
		checkpoint_txt = os.path.join(self.checkpoint_path, "checkpoint")
		if os.path.exists(checkpoint_txt):
			fin = open(checkpoint_txt)
			model_dir = fin.readline().strip()
			fin.close()
			model_dir='checkpoint/color_all_ae_64/IM_AE.model16-199_raw.pth'
			self.im_network.load_state_dict(torch.load(model_dir))
			print(" [*] Load SUCCESS", model_dir)
      
		else:
			print(" [!] Load failed...")
			return
		
		self.im_network.eval()
		#for t in range(config.start, min(len(self.data_voxels),config.end)):
   
   
   
   
		for path in glob.glob('/mnt/sdb/lzz/transform/IM-NET-pytorch/point_sampling/hdf5_train/*.hdf5'): #self.data_paths: #[config.start:config.end]:
     
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
			#model_z,_,_ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, None, is_training=False)
			model_z,z_vector_color,_,_ = self.im_network(torch.cat((batch_voxels,batch_voxels_colors),1), None,None, None, is_training=False)
 
			z=model_z.detach().cpu().numpy()
			z_vector_color=z_vector_color.detach().cpu().numpy()
			#print (z.shape, z_vector_color.shape)
			z=np.concatenate((z,z_vector_color),1)
			print (z.shape)
			np.save('../feat32_color_train/'+name+'.npy',z)

      
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


