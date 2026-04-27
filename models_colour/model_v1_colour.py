# ============================================================
# models_colour/model_v1_colour.py
# 颜色识别 Model V1：4层浅层CNN，无BatchNorm
#
# 对比目的：颜色识别基础baseline
# 与形状V1的区别：
#   - 通道数大幅缩减：32→64（形状是64→512）
#   - 只有2个Block（形状有4个）
#   - 分类头缩减：FC(64→32→9)
#   - Dropout从0.5降到0.3（数据充足，过拟合风险低）
# ============================================================

import torch
import torch.nn as nn


class LegoColourCNN_V1(nn.Module):
    """
    4层浅层CNN（颜色识别 Model V1），无BatchNorm

    架构概览：
    输入 (B, 3, 224, 224)
      ↓
    Block1: Conv(3→32)×2 + ReLU + MaxPool  → (B, 32, 112, 112)
      ↓
    Block2: Conv(32→64)×2 + ReLU + MaxPool → (B, 64,  56,  56)
      ↓
    GlobalAvgPool → (B, 64)
      ↓
    FC(64→32) + ReLU + Dropout(0.3)
      ↓
    FC(32→9)
    输出 (B, 9)

    设计理由：
    颜色特征在极浅层就能被充分提取（色相/饱和度/亮度
    在第一个卷积Block就能感知），不需要深层网络。
    通道数32→64足够编码9种颜色的区分特征。
    """

    def __init__(self, num_classes=9):
        super(LegoColourCNN_V1, self).__init__()

        # ── Block 1 ───────────────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 32, 112, 112)
        self.block1 = nn.Sequential(
            nn.Conv2d(3,  32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224→112
        )

        # ── Block 2 ───────────────────────────────────────────
        # 输入：(B, 32, 112, 112)
        # 输出：(B, 64, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112→56
        )

        # ── 全局平均池化 ──────────────────────────────────────
        # (B, 64, 56, 56) → (B, 64, 1, 1)
        # 对颜色任务尤其合适：
        # 全局平均池化会把整张特征图的信息压缩成一个值，
        # 相当于统计整张图的"平均颜色特征"，
        # 正好符合颜色识别与位置无关的特性
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 分类器 ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),   # 颜色任务数据充足，0.3够用
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)        # (B, 32, 112, 112)
        x = self.block2(x)        # (B, 64,  56,  56)
        x = self.avgpool(x)       # (B, 64,   1,   1)
        x = torch.flatten(x, 1)  # (B, 64)
        x = self.classifier(x)   # (B, 9)
        return x


def create_model_v1_colour(num_classes=9):
    model = LegoColourCNN_V1(num_classes=num_classes)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("颜色 Model V1：4层浅层CNN（无BatchNorm）")
    print("=" * 50)
    print(f"总参数数：    {total_params:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print("=" * 50)
    print("Block结构：")
    print("  Block1: Conv×2 + ReLU + MaxPool  (3  → 32)")
    print("  Block2: Conv×2 + ReLU + MaxPool  (32 → 64)")
    print("  GlobalAvgPool → FC(64→32) → FC(32→9)")
    print("=" * 50 + "\n")

    return model


if __name__ == "__main__":
    print("测试 颜色 Model V1...")
    model       = create_model_v1_colour(num_classes=9)
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
    x = model.avgpool(x); print(f"  AvgPool 输出：{tuple(x.shape)}")
    x = torch.flatten(x, 1); print(f"  Flatten 输出：{tuple(x.shape)}")
