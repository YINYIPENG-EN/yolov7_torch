import torch_pruning as tp
from loguru import logger
from models.common import *
from models.experimental import Ensemble
from utils.torch_utils import select_device
"""
剪枝的时候根据模型结构去剪，不要盲目的猜
剪枝完需要进行一个微调训练
"""

# 加载模型
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


@logger.catch
def layer_pruning(weights):
    logger.add('../logs/layer_pruning.log', rotation='1 MB')
    device = select_device('cpu')
    model, ckpt = attempt_load(weights, map_location=device)
    for para in model.parameters():
        para.requires_grad = True
    # 创建输入样例，可在此修改输入大小
    x = torch.zeros(1, 3, 640, 640)
    # -----------------对整个模型的剪枝--------------------
    strategy = tp.strategy.L1Strategy()  # L1策略
    DG = tp.DependencyGraph()  # 依赖图
    DG = DG.build_dependency(model, example_inputs=x)

    """
    这里写要剪枝的层
    这里以backbone为例
    """
    included_layers = []
    for layer in model.model[:10]:  # 获取backbone
        if type(layer) is Conv:
            included_layers.append(layer.conv)
            included_layers.append(layer.bn)
    logger.info(included_layers)
    # 获取未剪枝之前的参数量
    num_params_before_pruning = tp.utils.count_params(model)
    # 模型遍历
    for m in model.modules():
        # 判断是否为卷积并且是否在需要剪枝的层里
        if isinstance(m, nn.Conv2d) and m in included_layers:
            # amount是剪枝率
            # 卷积剪枝
            pruning_plan = DG.get_pruning_plan(m, tp.prune_conv, idxs=strategy(m.weight, amount=0.8))
            logger.info(pruning_plan)
            # 执行剪枝
            pruning_plan.exec()
        if isinstance(m, nn.BatchNorm2d) and m in included_layers:
            # BN层剪枝
            pruning_plan = DG.get_pruning_plan(m, tp.prune_batchnorm, idxs=strategy(m.weight, amount=0.8))
            logger.info(pruning_plan)
            pruning_plan.exec()
    # 获得剪枝以后的参数量
    num_params_after_pruning = tp.utils.count_params(model)
    # 输出一下剪枝前后的参数量
    logger.info("  Params: %s => %s\n" % (num_params_before_pruning, num_params_after_pruning))
    # 剪枝完以后模型的保存(不要用torch.save(model.state_dict(),...))

    model_ = {
        'model': model.half(),
        # 'optimizer': ckpt['optimizer'],
        # 'training_results': ckpt['training_results'],
        'epoch': ckpt['epoch']
    }
    torch.save(model_, '../model_data/layer_pruning.pt')
    del model_, ckpt
    logger.info("剪枝完成\n")


layer_pruning('../runs/train/exp/weights/best.pt')
#layer_pruning('../yolov7.pt')

