import torch
model = torch.load('../yolov7.pt')
print(model['model'])