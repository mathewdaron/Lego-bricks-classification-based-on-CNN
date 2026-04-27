# ============================================================
# models_colour/model_v2_colour.py
# 颜色识别 Model V2：6层CNN，含BatchNorm
#
# 相比颜色V1的两处核心升级：
#   1. Block数量从2个增加到3个（4层→6层）
#   2. 每层卷积后加入BatchNorm（Conv→BN→ReLU）
#   3. 通道数扩展到128（V1最深只有64）
#
# 对比目的：验证BN和轻度加深对颜色识别的提升效果
# ============================================================

import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    """
    单个卷积单元：Conv2d → BatchNorm2d → ReLU
    （与形状V2完全相同的设计，复用相同思路）
    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False   # 有BN时不需要bias
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LegoColourCNN_V2(nn.Module):
    """
    6层CNN（颜色识别 Model V2），含BatchNorm

    架构概览：
    输入 (B, 3, 224, 224)
      ↓
    Block1: ConvBNReLU×2 (3→32)   + MaxPool → (B, 32, 112, 112)
      ↓
    Block2: ConvBNReLU×2 (32→64)  + MaxPool → (B, 64,  56,  56)
      ↓
    Block3: ConvBNReLU×2 (64→128) + MaxPool → (B, 128, 28,  28)
      ↓
    GlobalAvgPool → (B, 128)
      ↓
    FC(128→64) + ReLU + Dropout(0.3)
      ↓
    FC(64→9)
    输出 (B, 9)

    相比V1新增Block3的设计理由：
    白色和灰色之间、灰色和黑色之间的区分主要依赖亮度差异，
    多一层Block让网络有机会学习更细致的亮度梯度特征，
    有助于提升这几个易混淆类别的准确率。
    """

    def __init__(self, num_classes=9):
        super(LegoColourCNN_V2, self).__init__()

        # ── Block 1 ───────────────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 32, 112, 112)
        self.block1 = nn.Sequential(
            ConvBNReLU(3,  32),
            ConvBNReLU(32, 32),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Block 2 ───────────────────────────────────────────
        # 输入：(B, 32, 112, 112)
        # 输出：(B, 64, 56, 56)
        self.block2 = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── Block 3 ───────────────────────────────────────────
        # 输入：(B, 64, 56, 56)
        # 输出：(B, 128, 28, 28)
        self.block3 = nn.Sequential(
            ConvBNReLU(64,  128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 全局平均池化 ──────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 分类器 ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)        # (B,  32, 112, 112)
        x = self.block2(x)        # (B,  64,  56,  56)
        x = self.block3(x)        # (B, 128,  28,  28)
        x = self.avgpool(x)       # (B, 128,   1,   1)
        x = torch.flatten(x, 1)  # (B, 128)
        x = self.classifier(x)   # (B, 9)
        return x


def create_model_v2_colour(num_classes=9):
    model = LegoColourCNN_V2(num_classes=num_classes)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("颜色 Model V2：6层CNN（含BatchNorm）")
    print("=" * 50)
    print(f"总参数数：    {total_params:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print("=" * 50)
    print("Block结构：")
    print("  Block1: ConvBN×2 + MaxPool  (3   → 32 )")
    print("  Block2: ConvBN×2 + MaxPool  (32  → 64 )")
    print("  Block3: ConvBN×2 + MaxPool  (64  → 128)")
    print("  GlobalAvgPool → FC(128→64) → FC(64→9)")
    print("=" * 50 + "\n")

    return model


if __name__ == "__main__":
    print("测试 颜色 Model V2...")
    model       = create_model_v2_colour(num_classes=9)
    dummy_input = torch.randn(4, 3, 224, 224)
    output      = model(dummy_input)

    print(f"✅ 前向传播成功！")
    print(f"  输入shape：{dummy_input.shape}")
    print(f"  输出shape：{output.shape}")
    print(f"  （应该是 torch.Size([4, 9])）")

    print(f"\n各层输出尺寸验证：")
    x = dummy_input
    x = model.block1(x);  print(f"  Block1  输出：{tuple(x.shape)}")
    x = model.block2(x);  print(f"  Block2  输出：{tuple(x.shape)}")
    x = model.block3(x);  print(f"  Block3  输出：{tuple(x.shape)}")
    x = model.avgpool(x); print(f"  AvgPool 输出：{tuple(x.shape)}")
    x = torch.flatten(x, 1); print(f"  Flatten 输出：{tuple(x.shape)}")
