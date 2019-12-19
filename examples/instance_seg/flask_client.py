import requests, json
import base64

endpoint = "http://192.168.1.6:5000/segment"

file = '/home/locobot/low_cost_ws/src/pyrobot/robots/LoCoBot/locobot_control/nodes/botcapture_1FZxE7.jpg'

def get_encoded_image(file):
    with open(file, 'rb') as image:
        f = base64.b64encode(image.read())
    return f

payload = {'encoded_image':get_encoded_image(file)}
# resp = requests.post(endpoint, data = json.dumps(payload))
resp = requests.post(endpoint, files = payload)
print(resp.text)


def infer_camera_image(encoded_nd):
    payload = {'encoded_image' : encoded_nd}
    resp = requests.post(endpoint, files = payload)
    print(resp.text)
