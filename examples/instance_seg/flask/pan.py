from pyrobot import Robot
from matplotlib import pyplot as plt
import cv2
import flask_client

from pyrobot.locobot.camera import DepthImgProcessor

bot = Robot('locobot')

bot.camera.reset()

# pan = 1.0
# bot.camera.set_pan(pan, wait=True)

while(True):
	rgb, depth = bot.camera.get_rgb_depth()
	print("RGB {}, {}".format(type(rgb), rgb.shape))
	print("Depth {}, {}".format(type(depth), depth.shape))
	flask_client.encode_and_infer(rgb)

	cv2.imshow('Depth', depth/1000)

	break

	# print(depth.shape)
	# cv2.imshow('Color', rgb[:,:,::-1])

	# depth = cv2.convertScaleAbs(depth)

	# dip = DepthImgProcessor()
	# print(dip.get_pix_3dpt(depth, [1,2], [1,2]))


	# cv2.imshow('Depth', depth)
	# print(depth[100:120, 100:120])
	# break
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
