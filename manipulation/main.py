import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np

from model_color_chair import IM_color_chair
from model_shape_chair import IM_shape_chair
from model_color_table import IM_color_table
from model_shape_table import IM_shape_table
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", action="store", dest="epoch", default=0, type=int, help="Epoch to train [0]")
parser.add_argument("--iteration", action="store", dest="iteration", default=0, type=int, help="Iteration to train. Either epoch or iteration need to be zero [0]")
parser.add_argument("--learning_rate", action="store", dest="learning_rate", default=0.00005, type=float, help="Learning rate for adam [0.00005]")
parser.add_argument("--beta1", action="store", dest="beta1", default=0.5, type=float, help="Momentum term of adam [0.5]")
parser.add_argument("--dataset", action="store", dest="dataset", default="color_all", help="The name of dataset")
parser.add_argument("--checkpoint_dir", action="store", dest="checkpoint_dir", default="checkpoint", help="Directory name to save the checkpoints [checkpoint]")
parser.add_argument("--data_dir", action="store", dest="data_dir", default="./data/all_vox256_img/", help="Root directory of dataset [data]")
parser.add_argument("--sample_dir", action="store", dest="sample_dir", default="./samples/", help="Directory name to save the image samples [samples]")
parser.add_argument("--sample_vox_size", action="store", dest="sample_vox_size", default=16, type=int, help="Voxel resolution for coarse-to-fine training [64]")
parser.add_argument("--train", action="store_true", dest="train", default=False, help="True for training, False for testing [False]")
parser.add_argument("--start", action="store", dest="start", default=0, type=int, help="In testing, output shapes [start:end]")
parser.add_argument("--end", action="store", dest="end", default=16, type=int, help="In testing, output shapes [start:end]")


parser.add_argument("--color_chair", action="store_true", dest="color_chair", default=False, help="True for manipulate color for chair [False]")
parser.add_argument("--color_table", action="store_true", dest="color_table", default=False, help="True for manipulate color for table [False]")
parser.add_argument("--shape_chair", action="store_true", dest="shape_chair", default=False, help="True for manipulate shape for chair [False]")
parser.add_argument("--shape_table", action="store_true", dest="shape_table", default=False, help="True for manipulate shape for table  [False]")
parser.add_argument("--initialize", action="store", dest="initialize", default="", type=str, help="Init model [.pth]")
parser.add_argument("--high_resolution", action="store_true", dest="high_resolution", default=False, help="True for high_resolution [False]")



FLAGS = parser.parse_args()



if not os.path.exists(FLAGS.sample_dir):
	os.makedirs(FLAGS.sample_dir)

if FLAGS.color_chair:
	im_color_chair = IM_color_chair(FLAGS)

	if FLAGS.train:
		im_color_chair.train(FLAGS)
	else:
		#im_ae.test_mesh(FLAGS)
		im_color_chair.test_mesh_point(FLAGS)
elif FLAGS.color_table:
	im_color_table = IM_color_table(FLAGS)

	if FLAGS.train:
		im_color_table.train(FLAGS)
	else:
		#im_svr.test_mesh(FLAGS)
		im_color_table.test_mesh_point(FLAGS)
elif FLAGS.shape_chair:
	im_shape_chair = IM_shape_chair(FLAGS)

	if FLAGS.train:
		im_shape_chair.train(FLAGS)
	else:
		#im_svr.test_mesh(FLAGS)
		im_shape_chair.test_mesh_point(FLAGS)

elif FLAGS.shape_table:
	im_shape_table = IM_shape_table(FLAGS)
	if FLAGS.train:
		im_shape_table.train(FLAGS)
	else:
		#im_svr.test_mesh(FLAGS)
		im_shape_table.test_mesh_point(FLAGS)