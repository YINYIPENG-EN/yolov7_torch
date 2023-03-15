# import
from copy import deepcopy
import sys
sys.path.append('./')
from models.yolo import Model
from models.common import Conv
from models.experimental import Ensemble
import torch
import torch.nn as nn
from utils.torch_utils import select_device, is_parallel
import argparse


def attempt_load(weights, map_location=None, inplace=True):
    from models.yolo import Detect, Model

    model = Ensemble()
    ckpt = torch.load(weights, map_location=map_location)  # load weights
    model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())  # without layer fuse  权值的加载

    # Compatibility updates
    for m in model.modules():  # 取出每一层
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model]:
            m.inplace = inplace  # pytorch 1.7.0 compatibility
            if type(m) is Detect:  # 判断是否为目标检测
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m) is Conv:  # 卷积层
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1], ckpt  # return model
    else:
        print(f'Ensemble created with {weights}\n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
        return model, ckpt  # return ensemble

def Yolov7_reparameterization(ckpt_path, yaml_path, num_classes):
    if opt.pruned:
        model, ckpt = attempt_load(ckpt_path, map_location=device)
    else:
        # model trained by cfg/training/*.yaml
        ckpt = torch.load(ckpt_path, map_location=device)  # load model,包含了所有Key
        # reparameterized model in cfg/deploy/*.yaml
        model = Model(yaml_path, ch=3, nc=num_classes).to(device)  # 模式实例化

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()  # 获得模型权重【并不是模型结构】
    exclude = []
    # intersect_state_dict = {k: v for k, v in state_dict.items() if
    #                         k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
    #                         model.state_dict()[k].shape}
    # 上面这句可以直接用下面这句替换，特别是对主干网络进行了修改
    # 获取预权重
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in state_dict.keys() == model.state_dict().keys()}
    # 将预权重加载到模型里
    model.load_state_dict(intersect_state_dict, strict=False)
    # 获取类名
    model.names = ckpt['model'].names  # 类的名字。例如person,car,....
    # 获取类的数量
    model.nc = ckpt['model'].nc  # 类的数量

    # reparametrized YOLOR
    # # reparametrized YOLOR 255 = (nc + 5) * anchor = 85*3=255
    for i in range((num_classes+5)*3):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i,
                                                                       ::].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i,
                                                                       ::].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i,
                                                                       ::].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(
        state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(
        state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(
        state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()

    # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}
    # save reparameterized model
    torch.save(ckpt, 'cfg/deploy/yolov7.pt')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='yolov7 trained model reparameterization')
    parser.add_argument('--ckpt', default='cfg/training/yolov7_training.pt', type=str, help='ckpt path')
    parser.add_argument('--device', type=str, default='0', help='gpu device,0,1,2..., or cpu')
    parser.add_argument('--yaml_path', type=str, default='cfg/deploy/yolov7.yaml', help='yaml file path')
    parser.add_argument('--num_classes', type=int, default=80, help='number of classes')
    parser.add_argument('--pruned', action='store_true', default=False, help='pruned model Reparam')
    opt = parser.parse_args()
    print(opt)
    device = select_device(opt.device, batch_size=1)
    ckpt_path = opt.ckpt
    yaml_path = opt.yaml_path
    num_classes = opt.num_classes
    Yolov7_reparameterization(ckpt_path, yaml_path, num_classes)