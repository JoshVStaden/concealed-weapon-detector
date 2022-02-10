# Instructions taken from https://pytorch.org/hub/ultralytics_yolov5/
import torch
from util.util import show_video

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# imgs = ['https://ultralytics.com/images/zidane.jpg']

show_video(model)