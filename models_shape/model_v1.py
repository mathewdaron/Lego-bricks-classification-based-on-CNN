# ============================================================
# models/model_v1.py
# Model V1：8层浅层CNN，无BatchNorm
# 用作基础baseline，快速验证数据加载和训练流程是否正常
# ============================================================

import torch
import torch.nn as nn

class LegoShapeCNN_V1(nn.Module):
    """
    8层浅层CNN模型（Model V1）
    
    架构概览：
    输入(224×224×3)
      ↓
    Block1: Conv(3→64) + ReLU + MaxPool → (112×112×64)
      ↓
    Block2: Conv(64→128) + ReLU + MaxPool → (56×56×128)
      ↓
    Block3: Conv(128→256) + ReLU + MaxPool → (28×28×256)
      ↓
    Block4: Conv(256→512) + ReLU + MaxPool → (14×14×512)
      ↓
    全局平均池化 → (1×1×512)
      ↓
    全连接层 + Dropout → 45类概率
    
    核心特点：
    - 无BatchNorm，直接ReLU
    - 4个卷积块 → 共8层卷积
    - 逐步加深通道数：64→128→256→512
    - 逐步减小空间尺寸：224→112→56→28→14
    """

    def __init__(self, num_classes=45):
        """
        参数：
            num_classes : 输出类别数（形状识别为45）
        """
        super(LegoShapeCNN_V1, self).__init__()

        # ── 第一个卷积块 ────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 64, 112, 112)
        self.block1 = nn.Sequential(
            # 卷积层1：3个输入通道 → 64个输出通道
            # 卷积核大小 3×3，填充1保持空间尺寸
            nn.Conv2d(3, 64, kernel_size=3, padding=1),

            # 激活函数：ReLU（移除负值）
            nn.ReLU(inplace=True),

            # 卷积层2：64→64，继续特征提取
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 最大池化：2×2，步长2（空间尺寸减半）
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 第二个卷积块 ────────────────────────────────
        # 输入：(B, 64, 112, 112)
        # 输出：(B, 128, 56, 56)
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 第三个卷积块 ────────────────────────────────
        # 输入：(B, 128, 56, 56)
        # 输出：(B, 256, 28, 28)
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 第四个卷积块 ────────────────────────────────
        # 输入：(B, 256, 28, 28)
        # 输出：(B, 512, 14, 14)
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # ── 全局平均池化 ────────────────────────────────
        # 把 (B, 512, 14, 14) 变成 (B, 512)
        # 对每个通道求平均，得到512维特征向量
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 全连接分类器 ────────────────────────────────
        # 输入：512维特征
        # 输出：45维概率分布
        self.classifier = nn.Sequential(
            # 全连接层：512 → 256
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),

            # Dropout：训练时随机丢弃50%的神经元，防止过拟合
            nn.Dropout(p=0.5),

            # 全连接层：256 → 45
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        """
        前向传播

        参数：
            x : 输入图片张量，shape (B, 3, 224, 224)
              B=批次大小，3=RGB通道，224×224=图片尺寸

        返回：
            output : 45维的类别概率张量，shape (B, 45)
        """
        # 依次通过四个卷积块
        x = self.block1(x)   # (B, 64, 112, 112)
        x = self.block2(x)   # (B, 128, 56, 56)
        x = self.block3(x)   # (B, 256, 28, 28)
        x = self.block4(x)   # (B, 512, 14, 14)

        # 全局平均池化
        x = self.avgpool(x)  # (B, 512, 1, 1)

        # 展平成1维向量
        x = torch.flatten(x, 1)  # (B, 512)

        # 全连接分类
        x = self.classifier(x)   # (B, 45)

        return x

# ============================================================
# 模型初始化和参数统计
# ============================================================
def create_model_v1(num_classes=45):
    """
    创建V1模型并打印参数统计信息

    返回：
        model : LegoShapeCNN_V1 实例
    """
    model = LegoShapeCNN_V1(num_classes=num_classes)

    # 计算参数总数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 50)
    print("Model V1：8层浅层CNN（无BatchNorm）")
    print("=" * 50)
    print(f"总参数数：{total_params:,}")
    print(f"可训练参数：{trainable_params:,}")
    print("=" * 50 + "\n")

    return model

# ============================================================
# 测试代码：验证模型输出尺寸是否正确
# ============================================================
if __name__ == "__main__":

    print("测试Model V1...")

    model = create_model_v1(num_classes=45)

    # 创建伪输入：批次大小4，RGB 3通道，224×224
    dummy_input = torch.randn(4, 3, 224, 224)

    # 前向传播
    output = model(dummy_input)

    print(f"✅ 前向传播成功！")
    print(f"  输入shape：{dummy_input.shape}")
    print(f"  输出shape：{output.shape}")
    print(f"  （应该是 torch.Size([4, 45])）")
