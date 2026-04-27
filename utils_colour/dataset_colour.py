# ============================================================
# utils_colour/dataset_colour.py
# 功能：颜色识别数据加载器，训练时对图片实时执行随机增强
#
# 使用方式：
#   from utils_colour.dataset_colour import get_train_loader_colour
#   train_loader, num_classes = get_train_loader_colour(data_dir, batch_size)
#
# 数据源：layer_one_colour/（9个子文件夹，每类约1112张）
# 在线增强：每个epoch训练时实时随机生成，不存到本地
# 倍率控制：通过 AUGMENT_MULTIPLY 参数控制等效数据量
#           设为3则等效 10008×3 = 30024张
#
# 与形状版(dataset.py)的核心差异：
#   1. AUGMENT_MULTIPLY 从10降为3（原始数据已更多，不需要高倍率）
#   2. 增强流水线删除随机裁剪（颜色识别与位置无关）
#   3. 增强流水线删除饱和度抖动（饱和度变化会改变颜色类别本身）
#   4. 亮度/对比度抖动幅度收窄（避免白/黑/灰互相混淆）
#   5. 类名从 shape_XX 改为 colour_XX
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
# 颜色识别原始数据量更多（每类约1112张），倍率设3已足够
# 等效数据量：10008 × 3 = 30024张
# 调节建议：
#   × 1 → 快速验证流程，不扩充
#   × 3 → 标准训练（默认）
#   × 5 → 数据更丰富，训练更慢
AUGMENT_MULTIPLY = 3


# ============================================================
# 辅助函数：添加高斯噪声（与形状版完全相同）
# ============================================================
def add_gaussian_noise(img, max_std=8):
    """
    给图片添加随机强度的高斯噪声

    参数：
        img     : PIL图片对象（RGB格式）
        max_std : 噪声最大标准差（默认8）

    返回：
        PIL图片对象
    """
    std = random.uniform(0, max_std)
    if std < 1.0:
        return img

    img_array = np.array(img, dtype=np.float32)
    noise     = np.random.normal(0, std, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


# ============================================================
# 在线增强变换流水线（颜色识别专用）
# ============================================================
TRAIN_TRANSFORM = transforms.Compose([

    # 第一步：保险resize，确保输入224×224
    transforms.Resize((224, 224)),

    # ── 与形状版的差异①：删除随机裁剪 ────────────────────────
    # 形状版有：transforms.RandomCrop(224, padding=22)
    # 颜色版删除原因：
    #   颜色识别依赖全局颜色统计，与零件在画面中的位置完全无关
    #   GlobalAvgPool已经做了全局平均，位置扰动没有意义
    #   且裁剪会把边缘的背景色引入更多，干扰颜色特征

    # 第二步：随机水平翻转，概率50%（与形状版相同）
    # 翻转不改变颜色，保留此操作增加多样性
    transforms.RandomHorizontalFlip(p=0.5),

    # 第三步：随机旋转 ±20°（与形状版略有缩减）
    # 形状版旋转±30°，颜色版缩减到±20°
    # 原因：旋转角度大时fill=128（中灰）填充区域增多，
    #       对颜色统计有轻微干扰，缩小旋转范围降低此影响
    transforms.RandomRotation(degrees=20, fill=128),

    # ── 与形状版的差异②：颜色抖动参数收窄，删除饱和度 ────────
    # 形状版：brightness=0.3, contrast=0.3, saturation=0.2, hue=0.0
    # 颜色版：brightness=0.15, contrast=0.15, saturation不做
    #
    # 删除饱和度的原因：
    #   饱和度降低会让彩色接近灰色，饱和度升高会让颜色过于鲜艳
    #   这会导致颜色类别边界模糊，例如低饱和度的红色可能被识别为灰色
    #
    # 亮度/对比度收窄到0.15的原因：
    #   白色零件亮度本来就高（约250），再增亮容易过曝
    #   黑色零件亮度本来就低（约40），再压暗细节完全消失
    #   收窄范围保护黑白灰三类不被过度扰动
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.0,   # 不抖动饱和度
        hue=0.0           # 不偏移色相
    ),

    # 第四步：高斯噪声（与形状版相同）
    # 轻微噪声模拟真实拍摄环境，提升泛化能力
    transforms.Lambda(lambda img: add_gaussian_noise(img, max_std=8)),

    # 第五步：转为 PyTorch Tensor（与形状版相同）
    transforms.ToTensor(),

    # 第六步：标准化（与形状版完全相同）
    # 使用相同的 ImageNet 均值和标准差
    # 保证两个模型的输入分布一致，方便未来联合使用
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 自定义数据集类
# ============================================================
class LegoColourDataset(Dataset):
    """
    乐高颜色分类数据集

    自动读取 layer_one_colour/ 下所有类别文件夹
    （colour_00 ~ colour_08，共9类）
    通过 multiply 参数控制每张图在一个epoch内重复采样次数

    例：10008张原图 × multiply=3 → 一个epoch等效30024张
    """

    def __init__(self, data_dir, transform=None, multiply=1):
        """
        参数：
            data_dir  : 数据根目录（layer_one_colour/）
            transform : 图片变换操作（含在线增强）
            multiply  : 数据倍率，默认1
        """
        self.data_dir  = data_dir
        self.transform = transform

        # 扫描所有类别文件夹，sorted保证每次顺序一致
        # colour_00→标签0，colour_01→标签1，...，colour_08→标签8
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])

        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # 扫描所有图片路径和对应标签
        base_samples = []
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            label     = self.class_to_idx[class_name]

            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    base_samples.append((img_path, label))

        # 倍率扩展：列表重复 multiply 次
        self.samples = base_samples * multiply

        print(f"颜色数据集加载完成：")
        print(f"  类别数量：  {len(self.classes)}")
        print(f"  类别列表：  {self.classes}")
        print(f"  原始图片数：{len(base_samples)}")
        print(f"  增强倍率：  ×{multiply}")
        print(f"  等效总样本：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================
# 数据加载器构建函数
# ============================================================
def get_train_loader_colour(
    data_dir,
    batch_size  = 32,
    num_workers = 0,
    pin_memory  = False,
    multiply    = AUGMENT_MULTIPLY
):
    """
    构建颜色识别训练集 DataLoader

    参数：
        data_dir    : 数据根目录（layer_one_colour/）
        batch_size  : 每批次图片数量
                      本地建议：32
                      云端GPU建议：64
        num_workers : 数据加载并行进程数
                      本地Windows：0
                      云端Linux：  4
        pin_memory  : 是否锁页内存
                      本地无GPU：False
                      云端有GPU：True
        multiply    : 数据增强倍率，默认使用全局 AUGMENT_MULTIPLY=3

    返回：
        train_loader : 训练集 DataLoader
        num_classes  : 类别总数（9）
    """
    dataset = LegoColourDataset(
        data_dir  = data_dir,
        transform = TRAIN_TRANSFORM,
        multiply  = multiply
    )
    num_classes = len(dataset.classes)

    print(f"\n颜色训练集构建：")
    print(f"  等效总样本：{len(dataset)}")
    print(f"  类别数：    {num_classes}")
    print(f"  批次大小：  {batch_size}")
    print(f"  num_workers：{num_workers}")
    print(f"  pin_memory： {pin_memory}")

    train_loader = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = num_workers,
        pin_memory  = pin_memory,
        drop_last   = True,
    )

    return train_loader, num_classes


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":

    # ── 本地测试配置 ──────────────────────────────────────────
    DATA_DIR    = r"C:\Users\lsbt\Desktop\lego_part_classifier\layer_one_colour"
    BATCH_SIZE  = 32
    NUM_WORKERS = 0
    PIN_MEMORY  = False
    # 云端改为：NUM_WORKERS=4, PIN_MEMORY=True
    # ──────────────────────────────────────────────────────────

    print("=" * 50)
    print("测试颜色数据加载器...")
    print("=" * 50 + "\n")

    train_loader, num_classes = get_train_loader_colour(
        data_dir    = DATA_DIR,
        batch_size  = BATCH_SIZE,
        num_workers = NUM_WORKERS,
        pin_memory  = PIN_MEMORY,
    )

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
