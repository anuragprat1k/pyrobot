from flask import Flask, jsonify, request
import sys
import numpy as np

from mask_inference import get_prediction, get_prediction_with_centers
import matplotlib.pyplot as plt
import io
import cv2
import base64
from PIL import ImageDraw, ImageFont
import tempfile


app = Flask(__name__)

def encode_PIL(img):
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()

def draw_xyz(img, xyz, cent):
    d = ImageDraw.Draw(img)
    fnt = ImageFont.truetype('/usr/share/fonts/truetype/tlwg/Loma.ttf', 100)
    d.text(cent, str(xyz[0]) + ',\n' + str(xyz[1]) +',\n'+str(xyz[2]), fill=(255,255,255))
    d.point(cent, fill=(255,0,0))
    return img

def get_coords(img, enc_xyz, centers):
    # decode
    xyz = base64.decodebytes(enc_xyz)
    coords = np.frombuffer(xyz, dtype=np.float64).reshape((4,-1))
    coords = np.around(coords, decimals=2)
    print("Decode success ? {} \n{}".format(coords.shape, coords[:, :5]))
    print("Image size {}".format(img.size))
    # map each x,y to a an index into coords
    # img = draw_xyz(img, [0,0,0], (0,0))
    for cent in centers:
        xyz = coords[:3, cent[1]*img.size[0] + cent[0]] # what should this index be
        print(type(xyz))
        img = draw_xyz(img, xyz, cent)
    # draw the coords on the img
    return img

@app.route('/segment', methods=['POST'])
def masksegment():
    if request.method == 'POST':
        encoded_image = request.files['encoded_image'].read()
        print("encoded payload received {}, {}".format(type(encoded_image), encoded_image))

        ret = get_prediction(encoded_image)
        ret.show() # this works but pops up a different window for each image

        return jsonify({'msg' : 'Change the world'})

@app.route('/segment_orient', methods=['POST'])
def masksegmentD():
    if request.method == 'POST':
        encoded_image = request.files['encoded_image'].read()
        encoded_coords = request.files['point_cloud'].read()

        print("encoded image received {}, ".format(type(encoded_image)))
        print("encoded point cloud received {}, ".format(type(encoded_coords)))

        ret, centers, _ = get_prediction_with_centers(encoded_image)
        ret = get_coords(ret, encoded_coords, centers)
        tf = tempfile.NamedTemporaryFile(prefix='segment_xyz_', suffix='.jpg', dir='.', delete=False)
        ret.save(tf.name)
        
        ret.show() # this works but pops up a different window for each image
        # ocv = cv2.cvtColor(np.array(ret), cv2.COLOR_RGB2BGR)
        # # print(type(ocv), ocv.shape)
        # cv2.imshow('ret', ocv)
        # plt.imshow(ocv)

        return jsonify({'msg' : 'Change the world'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')

