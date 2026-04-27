# ============================================================
# models_shape/model_v2.py
# Model V2：12层深度CNN，含BatchNorm
#
# 相比 V1 的两处核心升级：
#   1. 每个Block从2层卷积加深到3层（8层→12层）
#   2. 每层卷积后加入BatchNorm（Conv→BN→ReLU）
#
# 其余结构与V1完全一致，保证对比实验的控制变量原则：
#   - 相同的通道数progression：64→128→256→512→512
#   - 相同的GlobalAvgPool + 全连接分类器
#   - 相同的Dropout(0.5)
#
# 对比目的：验证"加深网络深度 + BatchNorm"对性能的提升效果
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 带 BatchNorm 的卷积单元（V2的基础构件）
# ============================================================
class ConvBNReLU(nn.Module):
    """
    单个卷积单元：Conv2d → BatchNorm2d → ReLU

    把这三个操作封装成一个小模块，有两个好处：
    1. 每个Block里重复用3次，避免写重复代码
    2. 结构清晰，和V1的"Conv→ReLU"形成鲜明对比

    BatchNorm的作用：
    - 对每个通道的输出做归一化，使数值分布稳定
    - 减少梯度消失/爆炸，让更深的网络也能稳定训练
    - 有轻微正则化效果，可以适当减少对Dropout的依赖
    - 加速收敛：通常比无BN的同等网络收敛更快
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        """
        参数：
            in_channels  : 输入通道数
            out_channels : 输出通道数
            kernel_size  : 卷积核大小（默认3×3）
            padding      : 填充（默认1，保持空间尺寸不变）
        """
        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False    # 使用BN时卷积层不需要bias
                          # 原因：BN内部有自己的偏移参数(beta)
                          # 保留bias是冗余的，去掉可减少参数量
        )
        self.bn   = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ============================================================
# 主模型类
# ============================================================
class LegoShapeCNN_V2(nn.Module):
    """
    12层深度CNN（Model V2），含BatchNorm

    架构概览：
    输入 (B, 3, 224, 224)
      ↓
    Block1: ConvBNReLU×3 (3→64→64→64)   + MaxPool → (B, 64,  112, 112)
      ↓
    Block2: ConvBNReLU×3 (64→128→128→128) + MaxPool → (B, 128,  56,  56)
      ↓
    Block3: ConvBNReLU×3 (128→256→256→256)+ MaxPool → (B, 256,  28,  28)
      ↓
    Block4: ConvBNReLU×3 (256→512→512→512)+ MaxPool → (B, 512,  14,  14)
      ↓
    Block5: ConvBNReLU×3 (512→512→512→512)+ MaxPool → (B, 512,   7,   7)
      ↓
    GlobalAvgPool → (B, 512)
      ↓
    FC(512→256) + ReLU + Dropout(0.5)
      ↓
    FC(256→45)
    输出 (B, 45)

    注意：V2比V1多一个Block5（512→512）
    原因：V1最后特征图是14×14，V2加深后需要额外一个Block
         让最终GlobalAvgPool前的特征图压缩到7×7，
         迫使网络学习更抽象的高层特征
    """

    def __init__(self, num_classes=45):
        """
        参数：
            num_classes : 输出类别数（形状识别为45）
        """
        super(LegoShapeCNN_V2, self).__init__()

        # ── Block 1 ───────────────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 64, 112, 112)
        # 第一层：3→64（通道扩展）
        # 第二、三层：64→64（深化特征，通道不变）
        self.block1 = nn.Sequential(
            ConvBNReLU(3,  64),   # 3→64，提取底层颜色/边缘特征
            ConvBNReLU(64, 64),   # 64→64，深化
            ConvBNReLU(64, 64),   # 64→64，深化
            nn.MaxPool2d(kernel_size=2, stride=2),  # 224→112
        )

        # ── Block 2 ───────────────────────────────────────────
        # 输入：(B, 64, 112, 112)
        # 输出：(B, 128, 56, 56)
        self.block2 = nn.Sequential(
            ConvBNReLU(64,  128),  # 64→128，通道加倍
            ConvBNReLU(128, 128),
            ConvBNReLU(128, 128),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 112→56
        )

        # ── Block 3 ───────────────────────────────────────────
        # 输入：(B, 128, 56, 56)
        # 输出：(B, 256, 28, 28)
        self.block3 = nn.Sequential(
            ConvBNReLU(128, 256),  # 128→256
            ConvBNReLU(256, 256),
            ConvBNReLU(256, 256),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 56→28
        )

        # ── Block 4 ───────────────────────────────────────────
        # 输入：(B, 256, 28, 28)
        # 输出：(B, 512, 14, 14)
        self.block4 = nn.Sequential(
            ConvBNReLU(256, 512),  # 256→512
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28→14
        )

        # ── Block 5 ───────────────────────────────────────────
        # 输入：(B, 512, 14, 14)
        # 输出：(B, 512, 7, 7)
        # V2新增的Block，V1没有
        # 作用：在GlobalAvgPool前再做一次特征整合
        #       让网络有机会学习更高层、更抽象的零件结构特征
        #       对于相似形状的细粒度区分（如1x2 vs 1x3）很有帮助
        self.block5 = nn.Sequential(
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            ConvBNReLU(512, 512),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14→7
        )

        # ── 全局平均池化 ──────────────────────────────────────
        # 把 (B, 512, 7, 7) 压缩成 (B, 512)
        # 对每个通道的7×7特征图求平均，得到512维特征向量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 全连接分类器 ──────────────────────────────────────
        # 与V1结构完全相同，保证分类头的控制变量一致
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        前向传播

        参数：
            x : 输入图片张量，shape (B, 3, 224, 224)

        返回：
            output : 45维类别得分，shape (B, 45)
        """
        x = self.block1(x)   # (B,  64, 112, 112)
        x = self.block2(x)   # (B, 128,  56,  56)
        x = self.block3(x)   # (B, 256,  28,  28)
        x = self.block4(x)   # (B, 512,  14,  14)
        x = self.block5(x)   # (B, 512,   7,   7)

        x = self.avgpool(x)       # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.classifier(x)   # (B, 45)

        return x


# ============================================================
# 模型初始化函数
# ============================================================
def create_model_v2(num_classes=45):
    """
    创建V2模型并打印参数统计信息

    返回：
        model : LegoShapeCNN_V2 实例
    """
    model = LegoShapeCNN_V2(num_classes=num_classes)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("Model V2：12层深度CNN（含BatchNorm）")
    print("=" * 50)
    print(f"总参数数：    {total_params:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print("=" * 50)
    print("Block结构：")
    print("  Block1: Conv×3 + BN + ReLU + MaxPool  (3   → 64 )")
    print("  Block2: Conv×3 + BN + ReLU + MaxPool  (64  → 128)")
    print("  Block3: Conv×3 + BN + ReLU + MaxPool  (128 → 256)")
    print("  Block4: Conv×3 + BN + ReLU + MaxPool  (256 → 512)")
    print("  Block5: Conv×3 + BN + ReLU + MaxPool  (512 → 512)")
    print("  GlobalAvgPool → FC(512→256) → FC(256→45)")
    print("=" * 50 + "\n")

    return model


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":

    print("测试 Model V2...")

    model = create_model_v2(num_classes=45)

    # 创建伪输入：批次大小4，RGB 3通道，224×224
    dummy_input = torch.randn(4, 3, 224, 224)

    # 前向传播测试
    output = model(dummy_input)

    print(f"✅ 前向传播成功！")
    print(f"  输入shape：{dummy_input.shape}")
    print(f"  输出shape：{output.shape}")
    print(f"  （应该是 torch.Size([4, 45])）")

    # 打印每层输出尺寸，方便验证
    print(f"\n各Block输出尺寸验证：")
    x = dummy_input
    for i, block in enumerate([
        model.block1, model.block2, model.block3,
        model.block4, model.block5
    ], start=1):
        x = block(x)
        print(f"  Block{i} 输出：{tuple(x.shape)}")
    x = model.avgpool(x)
    print(f"  AvgPool 输出：{tuple(x.shape)}")
    x = torch.flatten(x, 1)
    print(f"  Flatten 输出：{tuple(x.shape)}")
