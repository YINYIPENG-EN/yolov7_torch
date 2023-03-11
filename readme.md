

# 训练自己的训练集

此处的数据集是采用VOC的格式。

数据集存放格式：

> ─dataset
>  │  ├─Annotations # 存放xml标签文件
>  │  ├─images # 存放图片
>  │  ├─ImageSets # 存放图片名称的txt文件
>  │  └─labels # 存放标签txt文件

先运行项目代码makeTXT：

```python
python makeTXT.py
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

此时会在ImageSets下生成4个txt文件(这四个txt中仅包含每个图像的名称)

> ImageSets/
>  |-- test.txt
>  |-- train.txt
>  |-- trainval.txt
>  `-- val.txt

打开voc_label.py.修改classes为自己的类。

然后运行该代码。

```python
python voc_label.py
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

 将会在dataset文件下生成test.txt、train.txt、val.txt【这些txt仅包含图像路径】。然后在dataset/labels下会生成每个图像的txt【这些txt格式内容表示为类别索引+(center_x,center_y,w,h)】

**接下来是配置文件的修改**。

打开cfg/training/yolov7.yaml。将nc修改为自己的类别数量。

接下来在data/文件下新建一个yaml文件【我这里写的是mydata.yaml】，内容如下，需要修改两个地方：

> ```
> train: ./dataset/train.txt
> val: ./dataset/val.txt
> test: ./dataset/test.txt
> 
> # number of classes
> nc: 1 # 修改处1  修改为自己的类
> 
> # class names
> names: [ 'target' ]  # 修改处2 类的名称
> ```

 有关训练中的超参数设置【比如初始学习率，动量，权重衰减等，可自行在data/hyp.scratch.p5.yaml中修改】。

训练：

```python
python train.py --weights yolov7.pt --batch-size 2 --device 0
```

![点击并拖拽以移动](data:image/gif;base64,R0lGODlhAQABAPABAP///wAAACH5BAEKAAAALAAAAAABAAEAAAICRAEAOw==)

 正常的训练将会看到以下信息。

```bash
2023-03-11 11:50:48.658 | INFO     | __main__:train:316 - 
     Epoch   gpu_mem       box       obj       cls     total    labels  img_size
     0/299     2.58G   0.04649    0.4474         0    0.4939         5       640: 100%|██████████████████████████████████████████████████████████████| 359/359 [02:39<00:00,  2.25it/s] 
               Class      Images      Labels           P           R      mAP@.5  mAP@.5:.95: 100%|████████████████████████████████████████████████████| 20/20 [00:07<00:00,  2.78it/s]
                 all          80         147         0.2       0.204       0.102      0.0191
```

# 生成推理阶段的模型

由于yolov7中训练与推理并不是一个模型，是将训练后的模型进行重参数生成新模型。

因此需要运行tools/Reparameterization.py文件。【运行前注意修改文件中的权重路径以及类的数量】

# torch转onnx

修改tools/pytorch2onnx.py中的权重路径

运行该代码即可得到onnx模型



**后续将更新tensorrt以及剪枝处理**，请持续关注