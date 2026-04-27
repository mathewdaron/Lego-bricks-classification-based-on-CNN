# ============================================================
# models_colour/model_v3_colour.py
# 颜色识别 Model V3：迷你ResNet，9层，含BN + 残差连接
#
# 与形状V3相比的简化：
#   - Stem用Conv3×3替代Conv7×7（颜色不需要大感受野）
#   - 每个Layer只有1个BasicBlock（形状V3每层2个）
#   - 通道数上限128（形状V3是512）
#   - Layer数量从4个减为3个
#
# 对比目的：验证残差结构在颜色识别这类简单任务上的效果
# ============================================================

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    基础残差块（与形状V3完全相同的设计）

    主路径：Conv3×3 → BN → ReLU → Conv3×3 → BN
    跳跃连接：
        通道/尺寸相同 → Identity（直接相加）
        通道/尺寸不同 → 1×1Conv + BN 对齐后相加
    最终输出：ReLU(主路径 + shortcut)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2  = nn.BatchNorm2d(out_channels)

        # 跳跃连接：需要对齐时用1×1卷积
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu(out)
        return out


class LegoColourCNN_V3(nn.Module):
    """
    迷你ResNet（颜色识别 Model V3），9层，含BN + 残差连接

    架构概览：
    输入 (B, 3, 224, 224)
      ↓
    Stem: Conv3×3(3→32, s=1) + BN + ReLU + MaxPool → (B, 32, 56, 56)
      ↓
    Layer1: BasicBlock×1 (32→32,  stride=1) → (B, 32,  56, 56)
      ↓
    Layer2: BasicBlock×1 (32→64,  stride=2) → (B, 64,  28, 28)
      ↓
    Layer3: BasicBlock×1 (64→128, stride=2) → (B, 128, 14, 14)
      ↓
    GlobalAvgPool → (B, 128)
      ↓
    FC(128→64) + ReLU + Dropout(0.3)
      ↓
    FC(64→9)
    输出 (B, 9)

    层数统计：
    Stem:   1层卷积
    Layer1: 1个Block × 2层 = 2层卷积
    Layer2: 1个Block × 2层 = 2层卷积
    Layer3: 1个Block × 2层 = 2层卷积
    合计：  1 + 3×2 = 7层卷积（+2层1×1投影卷积）≈ 9层

    Stem用Conv3×3而非形状V3的Conv7×7：
    颜色特征不依赖大范围空间关系，3×3感受野已经足够，
    小卷积核参数更少，计算更快
    """

    def __init__(self, num_classes=9):
        super(LegoColourCNN_V3, self).__init__()

        # ── Stem层 ────────────────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 32, 56, 56)
        # Conv3×3 stride=1 → 224→224
        # MaxPool stride=2 → 224→112
        # MaxPool stride=2 → 112→56
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224→112
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112→56
        )

        # ── Layer 1 ───────────────────────────────────────────
        # 输入：(B, 32, 56, 56)
        # 输出：(B, 32, 56, 56)  通道和尺寸不变，精炼特征
        self.layer1 = BasicBlock(32, 32,  stride=1)

        # ── Layer 2 ───────────────────────────────────────────
        # 输入：(B, 32, 56, 56)
        # 输出：(B, 64, 28, 28)  通道加倍，尺寸减半
        self.layer2 = BasicBlock(32, 64,  stride=2)

        # ── Layer 3 ───────────────────────────────────────────
        # 输入：(B, 64, 28, 28)
        # 输出：(B, 128, 14, 14) 通道加倍，尺寸减半
        self.layer3 = BasicBlock(64, 128, stride=2)

        # ── 全局平均池化 ──────────────────────────────────────
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 分类器 ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

        # ── 权重初始化 ────────────────────────────────────────
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu'
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias,   0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)          # (B,  32, 56, 56)
        x = self.layer1(x)        # (B,  32, 56, 56)
        x = self.layer2(x)        # (B,  64, 28, 28)
        x = self.layer3(x)        # (B, 128, 14, 14)
        x = self.avgpool(x)       # (B, 128,  1,  1)
        x = torch.flatten(x, 1)  # (B, 128)
        x = self.classifier(x)   # (B, 9)
        return x


def create_model_v3_colour(num_classes=9):
    model = LegoColourCNN_V3(num_classes=num_classes)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 55)
    print("颜色 Model V3：迷你ResNet（含BN + 残差连接）")
    print("=" * 55)
    print(f"总参数数：    {total_params:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print("=" * 55)
    print("架构结构：")
    print("  Stem:   Conv3×3(3→32) + BN + ReLU + MaxPool×2")
    print("  Layer1: BasicBlock (32→32,  stride=1)")
    print("  Layer2: BasicBlock (32→64,  stride=2)")
    print("  Layer3: BasicBlock (64→128, stride=2)")
    print("  GlobalAvgPool → FC(128→64) → Dropout → FC(64→9)")
    print("=" * 55 + "\n")

    return model


if __name__ == "__main__":
    print("测试 颜色 Model V3...")
    model       = create_model_v3_colour(num_classes=9)
    dummy_input = torch.randn(4, 3, 224, 224)
    output      = model(dummy_input)

    print(f"✅ 前向传播成功！")
    print(f"  输入shape：{dummy_input.shape}")
    print(f"  输出shape：{output.shape}")
    print(f"  （应该是 torch.Size([4, 9])）")

    print(f"\n各层输出尺寸验证：")
    x = dummy_input
    x = model.stem(x);    print(f"  Stem   输出：{tuple(x.shape)}")
    x = model.layer1(x);  print(f"  Layer1 输出：{tuple(x.shape)}")
    x = model.layer2(x);  print(f"  Layer2 输出：{tuple(x.shape)}")
    x = model.layer3(x);  print(f"  Layer3 输出：{tuple(x.shape)}")
    x = model.avgpool(x); print(f"  AvgPool输出：{tuple(x.shape)}")
    x = torch.flatten(x, 1); print(f"  Flatten输出：{tuple(x.shape)}")
