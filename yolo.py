# Instructions taken from https://pytorch.org/hub/ultralytics_yolov5/
import torch
from util.util import show_video

GUN_DETECTION_MODEL = torch.hub.load('JoshVStaden/yolov5', 'custom', "yolopistol.pt")

# imgs = ['https://ultralytics.com/images/zidane.jpg']

if __name__ == '__main__':
    show_video(GUN_DETECTION_MODEL)
