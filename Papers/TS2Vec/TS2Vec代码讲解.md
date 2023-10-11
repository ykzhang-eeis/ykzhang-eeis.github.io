# TS2Vec代码讲解

### 先运行起来

git clone 到本地，然后下载[UCR datasets](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip)，解压密码为 someone，放到 datasets 文件夹下，重命名为UCR，此时运行脚本

```shell
python train.py <dataset_name> <run_name> --loader <loader> --batch-size <batch_size> --repr-dims <repr_dims> --gpu <gpu> --eval
# 比如
python train.py ECG200 testECG200 --loader UCR --eval
```

参数含义

| Parameter name | Description of parameter                                     |
| -------------- | ------------------------------------------------------------ |
| dataset_name   | The dataset name                                             |
| run_name       | The folder name used to save model, output and evaluation metrics. This can be set to any word |
| loader         | The data loader used to load the experimental data. This can be set to `UCR`, `UEA`, `forecast_csv`, `forecast_csv_univar`, `anomaly`, or `anomaly_coldstart` |
| batch_size     | The batch size (defaults to 8)                               |
| repr_dims      | The representation dimensions (defaults to 320)              |
| gpu            | The gpu no. used for training and inference (defaults to 0)  |
| eval           | Whether to perform evaluation after training                 |

### 当我们运行train.py时，都发生了什么

首先是经典的添加命令行参数

```python
import argparse

parser = argparse.ArgumentParser()
# 一系列的参数，挑两个有代表性的
parser.add_argument('--loader', type= , required=True, default= , help='')
parser.add_argument('--eval', action="store_true")
# action = "store_true" 表示如果在命令行中输入了--eval，则将该选项的值设置为True。如果未输入--eval，则该选项的值将保持默认值False
args = parser.parse_args()
```

一段



























