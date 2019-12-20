from pyrobot import Robot
from matplotlib import pyplot as plt
import cv2
import flask_client
import numpy as np

from pyrobot.locobot.camera import DepthImgProcessor
# from orb_slam2_ros.pcdlib import DepthImgProcessor

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
