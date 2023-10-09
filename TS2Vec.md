```Python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1          [2, 64, 128, 128]           9,408
       BatchNorm2d-2          [2, 64, 128, 128]             128
              ReLU-3          [2, 64, 128, 128]               0
         MaxPool2d-4            [2, 64, 64, 64]               0
            Conv2d-5            [2, 64, 64, 64]          36,864
       BatchNorm2d-6            [2, 64, 64, 64]             128
              ReLU-7            [2, 64, 64, 64]               0
            Conv2d-8            [2, 64, 64, 64]          36,864
       BatchNorm2d-9            [2, 64, 64, 64]             128
             ReLU-10            [2, 64, 64, 64]               0
       BasicBlock-11            [2, 64, 64, 64]               0
           Conv2d-12            [2, 64, 64, 64]          36,864
      BatchNorm2d-13            [2, 64, 64, 64]             128
             ReLU-14            [2, 64, 64, 64]               0
           Conv2d-15            [2, 64, 64, 64]          36,864
      BatchNorm2d-16            [2, 64, 64, 64]             128
             ReLU-17            [2, 64, 64, 64]               0
       BasicBlock-18            [2, 64, 64, 64]               0
           Conv2d-19           [2, 128, 32, 32]          73,728
      BatchNorm2d-20           [2, 128, 32, 32]             256
             ReLU-21           [2, 128, 32, 32]               0
           Conv2d-22           [2, 128, 32, 32]         147,456
      BatchNorm2d-23           [2, 128, 32, 32]             256
           Conv2d-24           [2, 128, 32, 32]           8,192
      BatchNorm2d-25           [2, 128, 32, 32]             256
             ReLU-26           [2, 128, 32, 32]               0
       BasicBlock-27           [2, 128, 32, 32]               0
           Conv2d-28           [2, 128, 32, 32]         147,456
      BatchNorm2d-29           [2, 128, 32, 32]             256
             ReLU-30           [2, 128, 32, 32]               0
           Conv2d-31           [2, 128, 32, 32]         147,456
      BatchNorm2d-32           [2, 128, 32, 32]             256
             ReLU-33           [2, 128, 32, 32]               0
       BasicBlock-34           [2, 128, 32, 32]               0
           Conv2d-35           [2, 256, 16, 16]         294,912
      BatchNorm2d-36           [2, 256, 16, 16]             512
             ReLU-37           [2, 256, 16, 16]               0
           Conv2d-38           [2, 256, 16, 16]         589,824
      BatchNorm2d-39           [2, 256, 16, 16]             512
           Conv2d-40           [2, 256, 16, 16]          32,768
      BatchNorm2d-41           [2, 256, 16, 16]             512
             ReLU-42           [2, 256, 16, 16]               0
       BasicBlock-43           [2, 256, 16, 16]               0
           Conv2d-44           [2, 256, 16, 16]         589,824
      BatchNorm2d-45           [2, 256, 16, 16]             512
             ReLU-46           [2, 256, 16, 16]               0
           Conv2d-47           [2, 256, 16, 16]         589,824
      BatchNorm2d-48           [2, 256, 16, 16]             512
             ReLU-49           [2, 256, 16, 16]               0
       BasicBlock-50           [2, 256, 16, 16]               0
           Conv2d-51             [2, 512, 8, 8]       1,179,648
      BatchNorm2d-52             [2, 512, 8, 8]           1,024
             ReLU-53             [2, 512, 8, 8]               0
           Conv2d-54             [2, 512, 8, 8]       2,359,296
      BatchNorm2d-55             [2, 512, 8, 8]           1,024
           Conv2d-56             [2, 512, 8, 8]         131,072
      BatchNorm2d-57             [2, 512, 8, 8]           1,024
             ReLU-58             [2, 512, 8, 8]               0
       BasicBlock-59             [2, 512, 8, 8]               0
           Conv2d-60             [2, 512, 8, 8]       2,359,296
      BatchNorm2d-61             [2, 512, 8, 8]           1,024
             ReLU-62             [2, 512, 8, 8]               0
           Conv2d-63             [2, 512, 8, 8]       2,359,296
      BatchNorm2d-64             [2, 512, 8, 8]           1,024
             ReLU-65             [2, 512, 8, 8]               0
       BasicBlock-66             [2, 512, 8, 8]               0
AdaptiveAvgPool2d-67             [2, 512, 1, 1]               0
           Linear-68                  [2, 1000]         513,000
================================================================
Total params: 11,689,512
Trainable params: 11,689,512
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 1.50
Forward/backward pass size (MB): 164.02
Params size (MB): 44.59
Estimated Total Size (MB): 210.12
----------------------------------------------------------------
```

- **from einops import rearrange, reduce, repeat**
  - einops的强项是把张量的维度操作具象化，让开发者“想出即写出” 
    - einops.rearrange()重新指定维度
    - einops.repeat()增加维度
    - einops.reduce()减少维度

```Python
from einops import rearrange

output_tensor = rearrange(input_tensor, 'h w c -> c h w')
```

用'h w c -> c h w'就完成了维度调换

```Python
import torch
 
a = torch.randn(3, 9, 9)  # [3, 9, 9]
output = rearrange(a, 'c (r p) w -> c r p w', p=3)
print(output.shape)   # [3, 3, 3, 9]
```

这是高级用法，把**中间维度看作r×p，然后给出p的数值，这样系统会自动把中间那个维度拆解成3×3**。这样就完成了[3, 9, 9] -> [3, 3, 3, 9]的维度转换

- **from torch.backends import cudnn**

```Python
from torch.backends import cudnn
cudnn.benchmark = False
cudnn.deterministic = True
```

- 设置 `torch.backends.cudnn.benchmark=True` 将会让程序在开始时花费一点额外时间，就是在每一个卷积层中测试 cuDNN 提供的所有卷积实现算法，然后选择最快的那个，为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。
- 适用场景是网络结构固定（不是动态变化的），网络的输入形状（batchsize，图片大小，输入通道等）是不变的，否则会导致程序不断调整优化，消耗时间。
- 设置 `torch.backends.cudnn.deterministic=True`，令每次返回的卷积算法是确定的，即默认算法。如果配合上设置 Torch 的随机种子为固定值的话，应该可以保证每次运行网络的时候相同输入的输出是固定的。