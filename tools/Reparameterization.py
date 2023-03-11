# import
from copy import deepcopy
from models.yolo import Model
import torch
from utils.torch_utils import select_device, is_parallel

def Yolov7_reparameterization(ckpt_path, yaml_path, num_classes):
    # model trained by cfg/training/*.yaml
    ckpt = torch.load(ckpt_path, map_location=device)  # load model
    # reparameterized model in cfg/deploy/*.yaml
    model = Model(yaml_path, ch=3, nc=num_classes).to(device)

    # copy intersect weights
    state_dict = ckpt['model'].float().state_dict()  # 获得权重
    exclude = []
    # intersect_state_dict = {k: v for k, v in state_dict.items() if
    #                         k in model.state_dict() and not any(x in k for x in exclude) and v.shape ==
    #                         model.state_dict()[k].shape}
    # 上面这句可以直接用下面这句替换，特别是对主干网络进行了修改
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in state_dict.keys() == model.state_dict().keys()}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names  # 类的名字。例如person,car,....
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
    torch.save(ckpt, '../cfg/deploy/yolov7.pt')

if __name__ == "__main__":
    device = select_device('0', batch_size=1)
    ckpt_path = '../cfg/training/yolov7_training.pt'
    yaml_path = '../cfg/deploy/yolov7.yaml'
    num_classes = 80
    Yolov7_reparameterization(ckpt_path, yaml_path, num_classes)