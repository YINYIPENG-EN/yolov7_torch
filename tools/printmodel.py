import torch
model = torch.load('../yolov7.pt')
print(model['model'].model[:10])  # print backbone

# model.keys() = dict_keys(['model', 'optimizer', 'training_results', 'epoch'])