import os
import shutil
import torchvision
from torchvision.datasets import CIFAR10
from PIL import Image

# 设置代理
os.environ['http_proxy'] = 'http://10.16.87.156:7890'
os.environ['https_proxy'] = 'http://10.16.87.156:7890'

# 定义目标目录
target_dir = './datasets/cifar10'

# 创建分类目录
for i in range(10):
    class_dir = os.path.join(target_dir, f'class{i}')
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

# 下载 CIFAR-10 数据集
trainset = CIFAR10(root='./temp_cifar10', train=True, download=True)
testset = CIFAR10(root='./temp_cifar10', train=False, download=True)

# 保存训练集图片
for idx, (img, label) in enumerate(trainset):
    class_dir = os.path.join(target_dir, f'class{label}')
    img_path = os.path.join(class_dir, f'train_{idx}.png')
    img.save(img_path)

# 保存测试集图片
for idx, (img, label) in enumerate(testset):
    class_dir = os.path.join(target_dir, f'class{label}')
    img_path = os.path.join(class_dir, f'test_{idx}.png')
    img.save(img_path)

# 删除临时目录
shutil.rmtree('./temp_cifar10')

print("CIFAR-10 数据集已下载并整理完成。")
