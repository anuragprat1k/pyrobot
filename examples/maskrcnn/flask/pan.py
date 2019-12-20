from pyrobot import Robot
from matplotlib import pyplot as plt
import cv2
import flask_client
import tempfile
import numpy as np

# imageio imports
import imageio
import imgaug as ia
from imgaug.augmentables.heatmaps import HeatmapsOnImage

from pyrobot.locobot.camera import DepthImgProcessor
# from orb_slam2_ros.pcdlib import DepthImgProcessor

bot = Robot('locobot')

bot.camera.reset()

# pan = 0.3
# bot.camera.set_pan(pan, wait=True)

def write_rgbd(rgb, depth):
	tf = tempfile.NamedTemporaryFile(prefix='botcapture_', suffix='.jpg', dir='.', delete=False)
	cv2.imwrite(tf.name, rgb[:, :, ::-1])
	cv2.imwrite('depth' + tf.name, depth)

def segment_xyz(bot):
	rgb, depth = bot.camera.get_rgb_depth()
	print("RGB {}, {}".format(type(rgb), rgb.shape))
	print("Depth {}, {}".format(type(depth), depth.shape))
	dip = DepthImgProcessor()

	print("depth size {}".format(depth.shape))
	l = depth.shape[0]
	b = depth.shape[1]

	xs = []
	ys = []
	for x in np.arange(l):
		for y in np.arange(b):
			xs.append(x)
			ys.append(y)

	pts = dip.get_pix_3dpt(depth, np.asarray(xs), np.asarray(ys)) 
	pts = pts.copy(order='C')
	print("get_pix_3dpt type {}, val {}, shape {} dtype {}".format(type(pts), pts[:, :5], pts.shape, pts.dtype))
	flask_client.encode_and_infer_D(rgb, pts)


while(True):
	segment_xyz(bot)
	# flask_client.encode_and_infer(rgb)

	# depth = depth/depth.max()
	# dpth = HeatmapsOnImage(depth.astype(np.float32), shape=rgb.shape) #, min_value=0.0, max_value=1.0)
	# ukn = dpth.draw_on_image(rgb)[0]
	# print("draw_on_image {}, len {}, shape {}".format(type(ukn), len(ukn), ukn[0].shape))
	# ia.imshow(ukn)
	# plt.imshow(ukn[0])
	# plt.show()

	# write_rgbd(rgb, depth)
	# norm_d = cv2.normalize(depth, None, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	#
	# print(type(norm_d))
	# # cv2.imshow('Depth', norm_d)
	#
	# plt.imshow(norm_d)
	# plt.show()
	# break

	# print(depth.shape)
	# cv2.imshow('Color', rgb[:,:,::-1])

	# depth = cv2.convertScaleAbs(depth)

	# print(depth[[1,2], [1,2]])

	# dip = DepthImgProcessor()
	#
	# print("depth size {}".format(depth.shape))
	# l = depth.shape[0]
	# b = depth.shape[1]
	#
	# xs = []
	# ys = []
	# for x in np.arange(l):
	# 	for y in np.arange(b):
	# 		xs.append(x)
	# 		ys.append(y)
	#
	#
	# pts = dip.get_pix_3dpt(depth, np.asarray(xs), np.asarray(ys)) # np.arange(depth.shape[0]), np.arange(depth.shape[1]))
	#
	# pts = pts.copy(order='C')
	# print("get_pix_3dpt type {}, val {}, shape {} dtype {}".format(type(pts), pts[:, :5], pts.shape, pts.dtype))
	#
	# flask_client.encode_and_infer_D(rgb, pts)
	# take this ndarray and send it to server, server will just pick one



	break

	# cv2.imshow('Depth', depth)
	# print(depth[100:120, 100:120])
	# break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

