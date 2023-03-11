from models.yolo import Model
from models.experimental import attempt_load
import torch
weights_path = '../yolov7.pt'
model = attempt_load(weights_path, map_location='cpu')
x = torch.ones(1, 3, 640, 640)
torch.onnx.export(model, x, "../cfg/deploy/yolov7.onnx")
