# ============================================================
# utils_shape/dataset.py
# 功能：定义数据加载器，训练时对图片实时执行随机增强
#
# 使用方式：
#   from utils_shape.dataset import get_train_loader
#   train_loader, num_classes = get_train_loader(data_dir, batch_size)
#
# 数据源：train_data_shape/（新版训练集，5:1划分后的训练部分，11010张）
# 在线增强：每个epoch训练时实时随机生成，不存到本地
# 倍率控制：通过 AUGMENT_MULTIPLY 参数控制等效数据量
#           设为10则等效 11010×10 = 110100张
#
# 环境切换说明：
#   本地Windows：num_workers=0,  pin_memory=False
#   云端GPU：    num_workers=4,  pin_memory=True
# ============================================================

import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import random


# ============================================================
# 全局配置：在线增强倍率
# ============================================================
# 含义：每张原始图在一个epoch内被重复采样的次数
# 效果：等效数据量 = 原始图片数 × AUGMENT_MULTIPLY
#       11010 × 10 = 110100（4090/5090显卡可轻松承受）
# 调节建议：
#   × 5  → 快速验证，约55050张，训练速度快
#   ×10  → 标准训练，约110100张（默认）
AUGMENT_MULTIPLY = 10


# ============================================================
# 辅助函数：添加高斯噪声
# 注意：必须定义在 TRAIN_TRANSFORM 之前，否则Lambda引用时报错
# ============================================================
def add_gaussian_noise(img, max_std=5):
    """
    给图片添加随机强度的高斯噪声

    参数：
        img     : PIL图片对象（RGB格式）
        max_std : 噪声最大标准差，越大噪点越明显
                  旧值：8，新值：5
                  原因：第一轮离线增强已包含足够变化，
                        在线噪声轻微扰动即可，避免破坏积木细节

    返回：
        PIL图片对象
    """
    std = random.uniform(0, max_std)

    # 噪声太小时直接跳过，节省计算
    if std < 1.0:
        return img

    img_array = np.array(img, dtype=np.float32)
    noise     = np.random.normal(0, std, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


# ============================================================
# 在线增强变换流水线（训练集）
# ============================================================
# 设计原则：
#   第一轮离线增强（data_augment_shape_v2.py）已做了较强的几何变换：
#     - 3种缩放比例、5种位置偏移、水平翻转、仿射倾斜、±15°旋转、8种背景色
#   因此在线增强的职责是：只做轻微扰动，增强鲁棒性，不再进行大幅几何破坏
#
# 与旧版的主要区别：
#   ① 删除 RandomRotation（离线已覆盖，避免叠加过度旋转+脏边问题）
#   ② RandomCrop padding：22 → 10（离线已做位置偏移，无需再大幅位移）
#   ③ ColorJitter 参数减半（离线已换背景色，在线轻微扰动即可）
#   ④ GaussianNoise max_std：8 → 5（降低噪声强度保护积木细节）
# ============================================================
TRAIN_TRANSFORM = transforms.Compose([

    # 第一步：保险resize，确保输入一定是224×224
    # 防止极少数尺寸不标准的图片导致后续操作报错
    transforms.Resize((224, 224)),

    # 第二步：随机裁剪
    # padding=10：先在四边各填充10像素（约4.5%），再随机裁回224×224
    # 效果：积木在画面中产生轻微随机位移
    # 旧值：padding=22（约10%）
    # 新值：padding=10（约4.5%）
    # 原因：第一轮离线增强已做了5种位置偏移，在线不需要再大幅移动
    transforms.RandomCrop(224, padding=10),

    # 第三步：随机水平翻转，概率50%
    # 注意：不做垂直翻转（乐高零件有方向性）
    transforms.RandomHorizontalFlip(p=0.5),

    # ── 旋转已取消 ──────────────────────────────────────────────
    # 旧版：RandomRotation(degrees=30, fill=128)
    # 取消原因：
    #   1. 第一轮离线增强已生成 ±15° 旋转的3个变体，角度多样性充足
    #   2. 旋转叠加：离线±15° + 在线±30° = 极端情况±45°，积木会歪得不自然
    #   3. 旋转后四角需要填充颜色，旧版 fill=128（灰色）与浅色训练集背景不匹配
    #      取消旋转后彻底消除这个隐患
    # ────────────────────────────────────────────────────────────

    # 第四步：颜色抖动（轻微版）
    # brightness=0.15  亮度随机变化：×0.85~1.15（旧值：0.3）
    # contrast=0.15    对比度随机变化：×0.85~1.15（旧值：0.3）
    # saturation=0.10  饱和度随机变化：×0.9~1.1（旧值：0.2）
    # hue=0.0          不做色调偏移（形状识别与颜色无关，保持原色）
    # 原因：第一轮离线增强已做了8种浅色背景替换，颜色多样性已够用
    #       在线只需轻微扰动模拟光照变化即可
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.0
    ),

    # 第五步：高斯噪声（自定义函数，轻微版）
    # 噪声标准差随机 0~5，模拟真实拍摄噪点
    # 旧值：max_std=8，新值：max_std=5
    # 原因：在线增强叠加已够强，降低噪声强度保护积木细节
    transforms.Lambda(lambda img: add_gaussian_noise(img, max_std=5)),

    # 第六步：转为 PyTorch Tensor
    # 同时把像素值从 0~255 归一化到 0.0~1.0
    transforms.ToTensor(),

    # 第七步：标准化
    # 使用 ImageNet 的均值和标准差，CNN训练的标准做法
    # 让每个通道数值分布接近均值0、标准差1，帮助模型更快收敛
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 自定义数据集类
# ============================================================
class LegoShapeDataset(Dataset):
    """
    乐高形状分类数据集

    自动读取 train_data_shape/ 下所有类别文件夹
    通过 multiply 参数控制每张图在一个epoch内重复采样的次数
    每次采样经过随机增强得到不同结果，等效扩充数据量

    例：11010张原图 × multiply=10 → 一个epoch等效110100张不同图片
    """

    def __init__(self, data_dir, transform=None, multiply=1):
        """
        参数：
            data_dir  : 数据根目录（train_data_shape/）
            transform : 图片变换操作（含在线增强）
            multiply  : 数据倍率，默认1（不重复）
                        传入 AUGMENT_MULTIPLY 即可启用倍率
        """
        self.data_dir  = data_dir
        self.transform = transform

        # 扫描所有类别文件夹，sorted保证每次顺序一致
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        # 类别名称 → 数字标签映射
        # 例：{'shape_00': 0, 'shape_01': 1, ..., 'shape_44': 44}
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # ── 扫描所有图片路径和对应标签 ───────────────────────────
        # 同时支持 .png 和 .jpg，防止混入其他格式被忽略
        base_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            label     = self.class_to_idx[class_name]

            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    base_samples.append((img_path, label))

        # ── 倍率扩展 ──────────────────────────────────────────────
        # 把 base_samples 列表重复 multiply 次
        # 例：原始11010条 × 10 = 110100条
        # DataLoader 每次取样时随机增强不同，等效110100张不同的图
        self.samples = base_samples * multiply

        print(f"数据集加载完成：")
        print(f"  类别数量：  {len(self.classes)}")
        print(f"  原始图片数：{len(base_samples)}")
        print(f"  增强倍率：  ×{multiply}")
        print(f"  等效总样本：{len(self.samples)}")

    def __len__(self):
        """返回等效总样本数（原始数量 × 倍率）"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        根据索引返回一张图片和标签
        DataLoader 会自动调用此方法

        参数：
            idx : 图片索引（0 到 len-1）

        返回：
            img   : 经过transform处理后的Tensor，shape(3,224,224)
            label : 整数标签（0~44）
        """
        img_path, label = self.samples[idx]

        # 读取图片，强制转为RGB（排除PNG透明通道干扰）
        img = Image.open(img_path).convert("RGB")

        # 执行增强变换（每次随机结果不同）
        if self.transform:
            img = self.transform(img)

        return img, label


# ============================================================
# 数据加载器构建函数
# ============================================================
def get_train_loader(
    data_dir,
    batch_size  = 32,
    num_workers = 0,
    pin_memory  = False,
    multiply    = AUGMENT_MULTIPLY   # 默认读取全局倍率配置
):
    """
    构建训练集 DataLoader

    参数：
        data_dir    : 数据根目录（train_data_shape/）
        batch_size  : 每批次图片数量
                      本地建议：32
                      云端GPU建议：64
        num_workers : 数据加载并行进程数
                      本地Windows：0
                      云端Linux：  4
        pin_memory  : 是否锁页内存，加速CPU→GPU传输
                      本地无GPU：False
                      云端有GPU：True
        multiply    : 数据增强倍率，默认使用全局 AUGMENT_MULTIPLY
                      也可以在调用时手动覆盖，例如：
                      get_train_loader(..., multiply=5)

    返回：
        train_loader : 训练集 DataLoader
        num_classes  : 类别总数（45）
    """

    dataset = LegoShapeDataset(
        data_dir  = data_dir,
        transform = TRAIN_TRANSFORM,
        multiply  = multiply
    )
    num_classes = len(dataset.classes)

    print(f"\n训练集构建：")
    print(f"  等效总样本：{len(dataset)}")
    print(f"  类别数：    {num_classes}")
    print(f"  批次大小：  {batch_size}")
    print(f"  num_workers：{num_workers}")
    print(f"  pin_memory： {pin_memory}")

    train_loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,    # 每个epoch打乱顺序，防止模型记住样本顺序
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,    # 丢弃最后不满一个batch的数据，保持batch大小一致
    )

    return train_loader, num_classes


# ============================================================
# 测试代码：直接运行此文件验证数据加载是否正常
# ============================================================
if __name__ == "__main__":

    # ── 本地测试配置 ──────────────────────────────────────────
    DATA_DIR    = r"C:\Users\lsbt\Desktop\lego_part_classifier\train_data_shape"
    BATCH_SIZE  = 32
    NUM_WORKERS = 0      # 本地Windows用0
    PIN_MEMORY  = False  # 本地无GPU用False
    # 云端使用时改为：NUM_WORKERS=4, PIN_MEMORY=True
    # ──────────────────────────────────────────────────────────

    print("=" * 50)
    print("测试数据加载器...")
    print("=" * 50 + "\n")

    train_loader, num_classes = get_train_loader(
        data_dir    = DATA_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
        # multiply 不传则自动使用全局 AUGMENT_MULTIPLY=10
    )

    # 取一批数据测试
    imgs, labels = next(iter(train_loader))

    print(f"\n✅ 测试成功！")
    print(f"  一批图片的shape：{imgs.shape}")
    print(f"  （{imgs.shape[0]}张图，"
          f"{imgs.shape[1]}通道，"
          f"{imgs.shape[2]}×{imgs.shape[3]}像素）")
    print(f"  标签示例：{labels[:8].tolist()}")
    print(f"  类别总数：{num_classes}")
    print(f"\n  像素值范围（标准化后）：")
    print(f"  最小值：{imgs.min():.3f}")
    print(f"  最大值：{imgs.max():.3f}")
    print(f"  均值：  {imgs.mean():.3f}")
