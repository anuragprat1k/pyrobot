from flask import Flask, jsonify, request
import sys
import numpy as np

from mask_inference import get_prediction
import matplotlib.pyplot as plt

import cv2

app = Flask(__name__)

@app.route('/segment', methods=['POST'])
def masksegment():
    if request.method == 'POST':
        encoded_image = request.files['encoded_image'].read()
        print(type(encoded_image), encoded_image)

        ret = get_prediction(encoded_image)
        ret.show()
        # ocv = cv2.cvtColor(np.array(ret), cv2.COLOR_RGB2BGR)
        # print(type(ocv), ocv.shape)
        # # cv2.imshow('ret', ocv)
        # plt.imshow(ocv)

        return jsonify({'msg' : 'Change the world'})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
