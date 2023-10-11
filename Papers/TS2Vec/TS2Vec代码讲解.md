### **先运行起来**

git clone 到本地，然后下载[UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip)，解压密码为 someone，放到 datasets 文件夹下，重命名为UCR，此时运行脚本

```Shell
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
# 比如
python train.py ECG200 testECG200 --loader UCR --eval
```

参数含义

| Parameter name | Description of parameter                                     |
| -------------- | ------------------------------------------------------------ |
| dataset_name   | The dataset name                                             |
| run_name       | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader         | The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart |
| batch_size     | The batch size (defaults to 8)                               |
| repr_dims      | The representation dimensions (defaults to 320)              |
| gpu            | The gpu no. used for training and inference (defaults to 0)  |
| eval           | Whether to perform evaluation after training                 |

### **当我们运行train.py时，都发生了什么**

首先是经典的添加命令行参数

```Python
import argparse

parser = argparse.ArgumentParser()
# 一系列的参数，挑几个有代表性的
parser.add_argument('--loader', type= , required=True, default= , help='')
parser.add_argument('--eval', action="store_true")# action = "store_true" 表示如果在命令行中输入了--eval，则将该选项的值设置为True。如果未输入--eval，则该选项的值将保持默认值False
parser.add_argument('--max-train-length', type=int, default=3000) # args.max_train_length = 3000，注意-和_的区别

args = parser.parse_args()
```

一段包含了固定随机种子、配置CUDNN库行为、是否启用TF32混合精度计算和控制多线程的代码（**比较固定**），函数返回一个device列表，如果不适用多线程就返回第一个device

```Python
def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

# 调用代码
device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
```

根据args.loader的值选择task_type，同时调用datautils.load_xxx(args.dataset)来返回train_data, train_labels, test_data, test_labels等

根据args.irregular的值选择正则化比例（只能在classification任务中修改args.irregular值，否则会报错），这属于原论文中的缺失值分类性能实验

![img](D:/Github_local_repo/ykzhang-eeis.github.io/images/-16970174308311.png)

将batch_size, lr, output_dims, max_train_length等参数以字典形式写入config变量

```Python
config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length
    )
```

根据args.save_every变量值选择几个epoch之后保存一下参数.pkl

```Python
parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback
  
if args.save_every is not None:
    unit = 'epoch' if args.epochs is not None else 'iter'
    config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)
```

创建保存权重的文件夹

加载模型

```Python
model = TS2Vec(
        input_dims=train_data.shape[-1],
        device=device,
        **config)
```

训练模型（使用的是`model.fit()`）

```Python
loss_log = model.fit(
    train_data,
    n_epochs=args.epochs,
    n_iters=args.iters,
    verbose=True # 控制是否在训练过程中输出详细信息。如果设置为 True，则会输出训练进度和损失值等信息
)
```

评估性能过程

```Python
if args.eval:
    if task_type == 'classification':
        out, eval_res = tasks.eval_classification(model, train_data, train_labels, test_data, test_labels, eval_protocol='svm')
    elif task_type == 'forecasting':
        out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols)
    elif task_type == 'anomaly_detection':
        out, eval_res = tasks.eval_anomaly_detection(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
    elif task_type == 'anomaly_detection_coldstart':
        out, eval_res = tasks.eval_anomaly_detection_coldstart(model, all_train_data, all_train_labels, all_train_timestamps, all_test_data, all_test_labels, all_test_timestamps, delay)
    else:
        assert False
    pkl_save(f'{run_dir}/out.pkl', out)
    pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
    print('Evaluation result:', eval_res)
```

### load_UCR()

选择数据路径->读取.csv文件->转化成矩阵->提取分类类别标签(第一列为分类类别标签)，转化成从0开始的连续整数->提取除第一列以外的矩阵数据，转化成np.float64格式的数据

标签编码：

- 在时间序列数据中，通常第一列包含类别标签，表示每个时间序列的类别。在这个步骤中，代码将这些原始标签映射到从0开始的连续整数。这个标签映射关系存储在字典 `transform` 中。然后，使用 `np.vectorize` 函数，将训练数据和测试数据中的原始标签替换为这些连续整数，分别保存在 `train_labels` 和 `test_labels` 中

```Python
train = train_array[:, 1:].astype(np.float64)
# 映射标签为从0开始的连续整数，比如(-1, 0, 3, -1)映射为(0, 1, 2, 0)
train_labels = np.vectorize(transform.get)(train_array[:, 0]) 
test = test_array[:, 1:].astype(np.float64)
test_labels = np.vectorize(transform.get)(test_array[:, 0])
```

`np.vectorize`的用法

```Python
import numpy as np

# 定义一个普通的 Python 函数
def my_function(x):
    # 这个函数对输入值 x 执行某些操作
    return x * 2

# 使用 np.vectorize 将函数向量化
vectorized_function = np.vectorize(my_function)

# 创建一个 NumPy 数组
arr = np.array([1, 2, 3, 4, 5])

# 使用向量化函数对整个数组进行操作
result = vectorized_function(arr)

# result 现在包含了 arr 中每个元素都经过 my_function 函数操作后的结果
# [2, 4, 6, 8, 10]
```

对于某些数据集，返回归一化后的数值，对于所有数据，返回的train和test矩阵都增加一个维度

```Python
return train[..., np.newaxis], train_labels, test[..., np.newaxis], test_labels
```

传入给网络模型的参数

```Python
config = dict(
    batch_size=args.batch_size,
    lr=args.lr,
    output_dims=args.repr_dims,
    max_train_length=args.max_train_length
)

model = TS2Vec(
    input_dims=train_data.shape[-1], # 对于单变量时间序列，input_dims的值应该为1
    device=device,
    **config
)
class TS2Vec:
    '''The TS2Vec model'''
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=16,
        max_train_length=None,
        temporal_unit=0,
        after_iter_callback=None,
        after_epoch_callback=None
    ):
        ''' 初始化一个 TS2Vec 模型。
        
        参数:
            input_dims (int): 输入维度。对于单变量时间序列，应将其设置为1。
            output_dims (int): 表示维度。
            hidden_dims (int): 编码器的隐藏维度。
            depth (int): 编码器中的隐藏残差块数量。
            device (int): 用于训练和推断的 GPU。
            lr (int): 学习率。
            batch_size (int): 批处理大小。
            max_train_length (Union[int, NoneType]): 训练的最大序列长度。对于长度大于 <max_train_length> 的序列，它将被裁剪成一些长度小于 <max_train_length> 的序列。
            temporal_unit (int): 执行时间对比的最小单位。在训练非常长的序列时，该参数有助于减少时间和内存成本。
            after_iter_callback (Union[Callable, NoneType]): 每次迭代后都会调用的回调函数。
            after_epoch_callback (Union[Callable, NoneType]): 每个时代后都会调用的回调函数。
        '''
```

__init__()函数里要重点看的几行代码

```Python
self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)
self.net = torch.optim.swa_utils.AveragedModel(self._net)
self.net.update_parameters(self._net)
```

1. `self._ne``t = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth).to(self.device)`
   1. 这一行创建了一个 `TSEncoder` 模型的实例，这是TS2Vec模型的核心组成部分。`TSEncoder` 模型通常是一个神经网络，用于将输入的时间序列数据映射到一个低维度的表示空间。参数 `input_dims` 表示输入维度，`output_dims` 表示输出维度，`hidden_dims` 表示隐藏层的维度，`depth` 表示模型的深度。
   2. `.to(self``.device)` 将创建的模型实例移到指定的计算设备，通常是GPU。这是为了加速模型的训练和推断。
2. `self.net = torch.optim.swa_utils.AveragedModel(self._net)`
   1. 这一行创建了一个用于参数平均的模型，通常用于一种称为 "Stochastic Weight Averaging (SWA)" 的训练策略。SWA 是一种用于改进深度学习模型的性能和鲁棒性的方法。`self._net` 是之前创建的 `TSEncoder` 模型的实例。
   2. `AveragedModel` 是 PyTorch 库中的一个类，用于创建参数平均的模型。参数平均可以帮助减小模型的方差，提高模型的性能。
3. `self.net.update_parameters(self._net)`
   1. 这一行用于将参数平均的模型 `self.net` 更新为 `self._net` 模型的参数。这是为了在训练过程中不断更新参数平均模型，以保持其对模型性能的影响。

在训练过程中，`self._net` 将用于实际的前向和反向传播，而 `self.net` 将用于参数平均。这有助于提高模型的泛化性能和稳定性。

`TS2Vec类中fit()`函数的作用

```Python
def fit(self, train_data, n_epochs=None, n_iters=None, verbose=False):
# 首先要确保输入的train_data的shape是(n_instance, n_timestamps, n_features)
assert train_data.ndim == 3

x = batch[0]
if self.max_train_length is not None and x.size(1) > self.max_train_length:
    window_offset = np.random.randint(x.size(1) - self.max_train_length + 1)
    x = x[:, window_offset : window_offset + self.max_train_length]
x = x.to(self.device)

ts_l = x.size(1)
crop_l = np.random.randint(low=2 ** (self.temporal_unit + 1), high=ts_l+1)
crop_left = np.random.randint(ts_l - crop_l + 1)
crop_right = crop_left + crop_l
crop_eleft = np.random.randint(crop_left + 1)
crop_eright = np.random.randint(low=crop_right, high=ts_l + 1)
crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0))
```

数据处理：

- 如果 `max_train_length`（在类属性中定义）不为 `None`，则根据 `max_train_length` 将时间序列数据切分成小片段，以减小训练数据的长度，以应对非常长的序列。
- 处理包含缺失值（NaN）的时间序列数据，以确保数据中没有全为NaN的时间戳。
- `ts_l = x.size(1)`：获取时间序列的长度。
- `crop_l`：随机生成一个子序列的长度，该长度介于 `2^(self.temporal_unit + 1)` 和 `ts_l` 之间。
- `crop_left` 和 `crop_right`：随机生成子序列的左边界和右边界，确保子序列在时间序列内。
- `crop_eleft` 和 `crop_eright`：随机生成子序列左边界和右边界相对于整个时间序列的偏移量，这些偏移量将用于确定子序列的位置。
- `crop_offset`：随机生成多个偏移值，这些偏移值将用于从整个时间序列中选择多个子序列。生成的偏移值的数量等于输入时间序列的数量（`x.size(0)`）。

```Python
class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels, channels, kernel_size):
        super().__init__()
        self.net = nn.Sequential(*[
            ConvBlock(
                channels[i-1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2**i,
                final=(i == len(channels)-1)
            )
            for i in range(len(channels))
        ])
        
    def forward(self, x):
        return self.net(x)
```

- `channels[i-1] if i > 0 else in_channels`：输入通道数。对于第一个卷积层，使用输入数据的通道数 `in_channels`，对于后续的卷积层，使用前一层的输出通道数。
- `channels[i]`：输出通道数，由 `channels` 列表中的相应元素指定。
- `kernel_size=kernel_size`：卷积核的大小，由构造函数传递的 `kernel_size` 参数指定。
- `dilation=2**i`：膨胀率，每个卷积层的膨胀率都是指数级增加的，以扩大感受野。
- `final=(i == len(channels)-1)`：一个布尔值，指示是否是最后一层卷积层。最后一层通常具有不同的处理方式。

```Python
class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)  # B x T x Ch
        
        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.repr_dropout(self.feature_extractor(x))  # B x Co x T
        x = x.transpose(1, 2)  # B x T x Co
        
        return x
```

1. `def forward(self, x, mask=None):`：定义了前向传播方法，用于指定数据在模型中的前向传播流程。这个方法接受输入数据 `x` 和可选的掩码 `mask`，并执行以下操作：
   1. `nan_mask = ~x.isnan().any(axis=-1)`：创建一个布尔掩码，用于检测输入数据中的缺失值，以便后续处理。
   2. `x[~nan_mask] = 0`：将输入数据中的缺失值用 0 填充。
   3. `x = self.input_fc(x)`：将输入数据通过线性全连接层进行线性变换，将输入维度从 `input_dims` 转换为 `hidden_dims`。
2. `if mask is None:`：如果没有提供掩码，根据模型是否处于训练模式来选择默认的掩码方式。
   1. 如果模型处于训练模式，使用 `self.mask_mode`。
   2. 否则，使用 `'all_true'` 作为默认掩码。
3. 接下来的代码根据所选的掩码方式生成相应的掩码：
   1. `'binomial'`：生成二项掩码，以随机掩盖一部分数据。
   2. `'continuous'`：生成连续掩码，以掩盖一部分数据。
   3. `'all_true'`：生成全为真的掩码，即不掩盖任何数据。
   4. `'all_false'`：生成全为假的掩码，即全部数据都被掩盖。
   5. `'mask_last'`：生成掩盖最后一个时间步的掩码。
4. 掩码操作应用于输入数据，以掩盖缺失值。此外，掩码也与之前生成的 `nan_mask` 结合使用。
5. 接下来是卷积编码器的处理：
   1. `x` 的维度进行转置操作，以适应卷积操作的输入要求。
   2. 通过 `self.feature_extractor` 处理输入数据，执行卷积编码操作。
   3. 最后，将卷积编码器的输出维度再次进行转置操作，以恢复原始维度。
6. 返回编码后的时间序列 `x` 作为输出。

### 重点看对比学习损失函数的写法

```Python
import torch
from torch import nn
import torch.nn.functional as F

def hierarchical_contrastive_loss(z1, z2, alpha=0.5, temporal_unit=0):
    loss = torch.tensor(0., device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            if 1 - alpha != 0:
                loss += (1 - alpha) * temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.size(1) == 1:
        if alpha != 0:
            loss += alpha * instance_contrastive_loss(z1, z2)
        d += 1
    return loss / d

def instance_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0)  # 2B x T x C
    z = z.transpose(0, 1)  # T x 2B x C
    sim = torch.matmul(z, z.transpose(1, 2))  # T x 2B x 2B
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # T x 2B x (2B-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss

def temporal_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss
```
