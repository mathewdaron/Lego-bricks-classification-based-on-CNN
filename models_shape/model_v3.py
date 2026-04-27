# ============================================================
# models_shape/model_v3.py
# Model V3：迷你ResNet，约17层，含BatchNorm + 残差连接
#
# 相比 V2 的核心升级：
#   引入残差连接（Skip Connection）
#   梯度可通过跳跃路径直接回流，解决深层网络梯度消失问题
#
# 与V1/V2保持一致的部分（控制变量）：
#   - 通道数 progression：64→128→256→512
#   - BatchNorm（与V2相同）
#   - GlobalAvgPool + FC(512→256) + Dropout(0.5) + FC(256→45)
#
# 对比目的：在V2基础上，单独验证"残差连接"带来的性能提升
# ============================================================

import torch
import torch.nn as nn


# ============================================================
# 基础残差块（BasicBlock）
# ============================================================
class BasicBlock(nn.Module):
    """
    基础残差块，包含两层卷积和一条跳跃连接

    主路径（Main Path）：
        Conv3×3 → BN → ReLU → Conv3×3 → BN

    跳跃连接（Skip Connection / Shortcut）：
        情况A：输入输出通道相同，步长为1
               直接把输入 x 加到主路径输出上（identity shortcut）
        情况B：输入输出通道不同，或步长不为1（需要下采样）
               用 1×1 Conv + BN 对 x 做维度/尺寸对齐（projection shortcut）

    最终输出：ReLU(主路径输出 + shortcut输出)
    """

    def __init__(self, in_channels, out_channels, stride=1):
        """
        参数：
            in_channels  : 输入通道数
            out_channels : 输出通道数
            stride       : 第一层卷积的步长
                           stride=1：空间尺寸不变
                           stride=2：空间尺寸减半（下采样）
        """
        super(BasicBlock, self).__init__()

        # ── 主路径 ────────────────────────────────────────────
        # 第一层卷积：可能同时做通道扩展和空间下采样
        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1,
            bias=False   # 使用BN时不需要bias（BN有自己的beta参数）
        )
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)

        # 第二层卷积：通道数和空间尺寸都不变
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1,
            bias=False
        )
        self.bn2   = nn.BatchNorm2d(out_channels)

        # ── 跳跃连接 ──────────────────────────────────────────
        # 判断是否需要对输入x做维度对齐
        # 需要对齐的条件：步长不为1（空间尺寸变了）或通道数变了
        if stride != 1 or in_channels != out_channels:
            # Projection Shortcut：用1×1卷积对齐维度
            # 1×1卷积不改变空间位置，只调整通道数和空间尺寸
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            # Identity Shortcut：直接使用原始输入，无需任何操作
            # nn.Identity() 就是"什么都不做，直接返回输入"
            self.shortcut = nn.Identity()

    def forward(self, x):
        """
        前向传播

        参数：
            x : 输入特征图

        返回：
            加上残差后经过ReLU的特征图
        """
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加：主路径输出 + shortcut处理后的输入
        out = out + self.shortcut(x)

        # 最后统一做一次ReLU
        # 注意：ReLU在相加之后，这是标准ResNet做法
        out = self.relu(out)

        return out


# ============================================================
# 辅助函数：构建一个Layer（含多个BasicBlock）
# ============================================================
def make_layer(in_channels, out_channels, num_blocks, stride=1):
    """
    构建一个包含多个BasicBlock的层

    第一个Block负责：通道扩展 + 可能的空间下采样（stride控制）
    后续Block：通道数和空间尺寸均不变（stride固定为1）

    参数：
        in_channels  : 该层输入通道数
        out_channels : 该层输出通道数
        num_blocks   : Block数量（本项目每层2个）
        stride       : 第一个Block的步长（2=下采样，1=不变）

    返回：
        nn.Sequential 打包的多个BasicBlock
    """
    layers = []

    # 第一个Block：负责通道变换和下采样
    layers.append(BasicBlock(in_channels, out_channels, stride=stride))

    # 后续Block：输入输出通道相同，步长为1，使用Identity shortcut
    for _ in range(1, num_blocks):
        layers.append(BasicBlock(out_channels, out_channels, stride=1))

    return nn.Sequential(*layers)


# ============================================================
# 主模型类
# ============================================================
class LegoShapeCNN_V3(nn.Module):
    """
    迷你ResNet（Model V3），约17层，含BatchNorm + 残差连接

    架构概览：
    输入 (B, 3, 224, 224)
      ↓
    Stem:   Conv7×7(3→64, stride=2) + BN + ReLU + MaxPool
            → (B, 64, 56, 56)
      ↓
    Layer1: BasicBlock×2 (64→64,  stride=1) → (B, 64,  56, 56)
      ↓
    Layer2: BasicBlock×2 (64→128, stride=2) → (B, 128, 28, 28)
      ↓
    Layer3: BasicBlock×2 (128→256,stride=2) → (B, 256, 14, 14)
      ↓
    Layer4: BasicBlock×2 (256→512,stride=2) → (B, 512,  7,  7)
      ↓
    GlobalAvgPool → (B, 512)
      ↓
    FC(512→256) + ReLU + Dropout(0.5) + FC(256→45)
    输出 (B, 45)

    层数统计：
    Stem:   1层卷积
    Layer1: 2个Block × 2层 = 4层卷积
    Layer2: 2个Block × 2层 = 4层卷积
    Layer3: 2个Block × 2层 = 4层卷积
    Layer4: 2个Block × 2层 = 4层卷积
    合计：  1 + 4×4 = 17层卷积
    """

    def __init__(self, num_classes=45):
        """
        参数：
            num_classes : 输出类别数（形状识别为45）
        """
        super(LegoShapeCNN_V3, self).__init__()

        # ── Stem层 ────────────────────────────────────────────
        # 输入：(B, 3, 224, 224)
        # 输出：(B, 64, 56, 56)
        #
        # 为什么用7×7大卷积核：
        # 输入图像224×224，包含零件的整体轮廓信息
        # 7×7卷积核感受野更大，第一层就能捕捉较大范围的结构特征
        # stride=2 同时做一次空间下采样：224→112
        # 接MaxPool再下采样一次：112→56
        # 这样后续Layer处理的特征图尺寸合理，计算量可控
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 112→56
        )

        # ── Layer 1 ───────────────────────────────────────────
        # 输入：(B, 64, 56, 56)
        # 输出：(B, 64, 56, 56)
        # stride=1：空间尺寸不变，通道数也不变
        # 作用：在浅层特征上做残差精炼，不急于下采样
        self.layer1 = make_layer(
            in_channels=64, out_channels=64,
            num_blocks=2, stride=1
        )

        # ── Layer 2 ───────────────────────────────────────────
        # 输入：(B, 64,  56, 56)
        # 输出：(B, 128, 28, 28)
        # stride=2：空间尺寸减半，通道数加倍
        self.layer2 = make_layer(
            in_channels=64, out_channels=128,
            num_blocks=2, stride=2
        )

        # ── Layer 3 ───────────────────────────────────────────
        # 输入：(B, 128, 28, 28)
        # 输出：(B, 256, 14, 14)
        self.layer3 = make_layer(
            in_channels=128, out_channels=256,
            num_blocks=2, stride=2
        )

        # ── Layer 4 ───────────────────────────────────────────
        # 输入：(B, 256, 14, 14)
        # 输出：(B, 512,  7,  7)
        self.layer4 = make_layer(
            in_channels=256, out_channels=512,
            num_blocks=2, stride=2
        )

        # ── 全局平均池化 ──────────────────────────────────────
        # (B, 512, 7, 7) → (B, 512, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # ── 全连接分类器 ──────────────────────────────────────
        # 与V1/V2完全相同，保证分类头一致
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )

        # ── 权重初始化 ────────────────────────────────────────
        # ResNet引入残差后，合适的初始化能进一步加速收敛
        # 对卷积层用kaiming正态初始化，对BN层用标准初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """
        权重初始化
        kaiming_normal_ 专为ReLU激活函数设计，
        能保证前向传播时各层输出的方差稳定，避免梯度消失/爆炸
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode='fan_out',       # 保持前向传播方差稳定
                    nonlinearity='relu'   # 针对ReLU计算增益系数
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # BN的gamma初始化为1
                nn.init.constant_(m.bias,   0)  # BN的beta初始化为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        前向传播

        参数：
            x : 输入图片张量，shape (B, 3, 224, 224)

        返回：
            output : 45维类别得分，shape (B, 45)
        """
        x = self.stem(x)      # (B,  64, 56, 56)

        x = self.layer1(x)    # (B,  64, 56, 56)
        x = self.layer2(x)    # (B, 128, 28, 28)
        x = self.layer3(x)    # (B, 256, 14, 14)
        x = self.layer4(x)    # (B, 512,  7,  7)

        x = self.avgpool(x)       # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.classifier(x)   # (B, 45)

        return x


# ============================================================
# 模型初始化函数
# ============================================================
def create_model_v3(num_classes=45):
    """
    创建V3模型并打印参数统计信息

    返回：
        model : LegoShapeCNN_V3 实例
    """
    model = LegoShapeCNN_V3(num_classes=num_classes)

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\n" + "=" * 55)
    print("Model V3：迷你ResNet（含BatchNorm + 残差连接）")
    print("=" * 55)
    print(f"总参数数：    {total_params:,}")
    print(f"可训练参数：  {trainable_params:,}")
    print("=" * 55)
    print("架构结构：")
    print("  Stem:   Conv7×7(3→64, s=2) + BN + ReLU + MaxPool")
    print("  Layer1: BasicBlock×2 (64→64,   stride=1)")
    print("  Layer2: BasicBlock×2 (64→128,  stride=2)")
    print("  Layer3: BasicBlock×2 (128→256, stride=2)")
    print("  Layer4: BasicBlock×2 (256→512, stride=2)")
    print("  GlobalAvgPool → FC(512→256) → Dropout → FC(256→45)")
    print("=" * 55 + "\n")

    return model


# ============================================================
# 测试代码
# ============================================================
if __name__ == "__main__":

    print("测试 Model V3...")

    model = create_model_v3(num_classes=45)

    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)

    print(f"✅ 前向传播成功！")
    print(f"  输入shape：{dummy_input.shape}")
    print(f"  输出shape：{output.shape}")
    print(f"  （应该是 torch.Size([4, 45])）")

    # 逐层尺寸验证
    print(f"\n各层输出尺寸验证：")
    x = dummy_input
    x = model.stem(x)
    print(f"  Stem   输出：{tuple(x.shape)}")
    x = model.layer1(x)
    print(f"  Layer1 输出：{tuple(x.shape)}")
    x = model.layer2(x)
    print(f"  Layer2 输出：{tuple(x.shape)}")
    x = model.layer3(x)
    print(f"  Layer3 输出：{tuple(x.shape)}")
    x = model.layer4(x)
    print(f"  Layer4 输出：{tuple(x.shape)}")
    x = model.avgpool(x)
    print(f"  AvgPool输出：{tuple(x.shape)}")
    x = torch.flatten(x, 1)
    print(f"  Flatten输出：{tuple(x.shape)}")
