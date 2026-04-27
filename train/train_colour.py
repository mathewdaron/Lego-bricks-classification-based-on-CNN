# ============================================================
# train/train_colour.py
# 乐高颜色识别模型 —— 训练脚本（V1 / V2 / V3 通用）
#
# 用法（在项目根目录下运行）：
#   python train/train_colour.py --model v1
#   python train/train_colour.py --model v2
#   python train/train_colour.py --model v3
#
# V1 专用命令（降低学习率防止梯度爆炸）：
#   python train/train_colour.py --model v1 --lr 0.0002
#
# 与 train_shape.py 的差异：
#   1. 导入 utils_colour/dataset_colour.py
#   2. num_classes = 9（颜色类别）
#   3. num_epochs  = 30（颜色任务简单，收敛更快）
#   4. 数据目录改为 layer_one_colour
#   5. 模型从 models_colour/ 导入
#   6. 所有保存文件名加 colour_ 前缀，与形状结果严格区分
#   7. 梯度裁剪 max_norm 收紧为 1.0（颜色V1网络更浅）
# ============================================================

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
matplotlib.use('Agg')  # 云端无显示器环境必须加，否则画图报错
import matplotlib.pyplot as plt

# ── 把项目根目录加入Python路径 ────────────────────────────────
# 原因：train_colour.py 在 train/ 子文件夹里
# 如果不加这行，Python 找不到 utils_colour/ 和 models_colour/ 文件夹
# os.path.abspath(__file__)    → 本文件的绝对路径
# os.path.dirname(...) 两次    → 从 train/ 退到项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# ── 导入颜色识别数据加载器 ────────────────────────────────────
# 与形状版的唯一导入差异：来自 utils_colour 而非 utils_shape
from utils_colour.dataset_colour import get_train_loader_colour


# ============================================================
# 超参数配置
# 所有需要调整的数值都集中在这里，方便后续修改
# ============================================================
CONFIG = {
    # ── 与形状版不同的参数 ────────────────────────────────────
    "data_dir"    : "layer_one_colour",  # 颜色数据集（相对项目根目录）
    "num_classes" : 9,                   # 9种颜色（形状是45）
    "num_epochs"  : 30,                  # 颜色任务收敛快，30轮足够

    # ── 与形状版相同的参数 ────────────────────────────────────
    "checkpoint_dir" : "checkpoints",   # 模型权重保存目录
    "results_dir"    : "results",       # 曲线图保存目录
    "batch_size"     : 64,              # 云端GPU显存充足用64，不足改32
    "learning_rate"  : 0.001,           # Adam初始学习率（占位，运行时被覆盖）
    "num_workers"    : 4,               # 云端AutoDL设4；本地Windows设0
}

# ★ 新增1：各模型专属默认学习率 ★
# 颜色 V1（4层，无BN）：极浅网络在增强数据下同样面临梯度不稳定问题
#   → 学习率降低5倍，与形状V1保持相同策略
# 颜色 V2（6层，有BN）、V3（9层，有BN+残差）：BN保护，用标准学习率
MODEL_DEFAULT_LR = {
    'v1': 0.0002,   # 无BN模型，学习率要低5倍
    'v2': 0.001,    # 有BN，标准学习率
    'v3': 0.001,    # 有BN+残差，标准学习率
}


# ============================================================
# 模型加载函数
# 根据命令行传入的模型名称，动态导入对应颜色识别模型
# ============================================================
def load_model(model_name, num_classes):
    """
    根据名称加载对应颜色识别模型

    参数：
        model_name  : 'v1' / 'v2' / 'v3'
        num_classes : 输出类别数（9）

    返回：
        model : 对应的神经网络模型实例
    """
    if model_name == 'v1':
        from models_colour.model_v1_colour import create_model_v1_colour
        model = create_model_v1_colour(num_classes=num_classes)

    elif model_name == 'v2':
        from models_colour.model_v2_colour import create_model_v2_colour
        model = create_model_v2_colour(num_classes=num_classes)

    elif model_name == 'v3':
        from models_colour.model_v3_colour import create_model_v3_colour
        model = create_model_v3_colour(num_classes=num_classes)

    return model


# ============================================================
# 单个 epoch 训练逻辑
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    执行一轮完整训练

    参数：
        model     : 神经网络模型
        loader    : 训练集 DataLoader
        criterion : 损失函数
        optimizer : 优化器
        device    : 'cuda' 或 'cpu'

    返回：
        avg_loss : 本epoch平均损失
        accuracy : 本epoch训练准确率（百分比）
    """

    # 切换训练模式
    # 训练模式下 Dropout 随机丢弃神经元、BatchNorm 使用批次统计量
    # 和后面 eval() 模式区分，每epoch必须重新设置
    model.train()

    total_loss = 0.0  # 累计所有样本的损失总和
    correct    = 0    # 预测正确的样本数
    total      = 0    # 处理过的总样本数

    for images, labels in loader:

        # 把数据搬到GPU（或CPU）
        images = images.to(device)
        labels = labels.to(device)

        # 清空上一个batch残留的梯度
        # PyTorch梯度默认累加，每个batch必须手动清零
        optimizer.zero_grad()

        # 前向传播：图片 → 模型 → 各类别得分
        outputs = model(images)        # shape: (batch_size, 9)

        # 计算损失
        # CrossEntropyLoss 内部已包含 softmax，无需手动处理
        loss = criterion(outputs, labels)

        # 反向传播：自动计算所有参数的梯度
        loss.backward()

        # ★ 新增2：梯度裁剪 ★
        # 把所有参数的梯度范数（L2）限制在 max_norm 以内
        # 作用：防止某个 batch 出现异常大的梯度导致权重剧烈更新
        #
        # 颜色版使用 max_norm=1.0（形状版是2.0），原因：
        #   颜色 V1 只有4层，梯度从输出层到输入层传播路径极短
        #   梯度爆炸一旦发生，幅度会比深层网络更集中、更剧烈
        #   收紧阈值能更早拦截异常梯度，保护效果更好
        #   对 V2/V3（有BN）完全无害：
        #     BN 本已把激活值归一化，梯度通常远低于1.0
        #     clip_grad_norm_ 检测到范数未超阈值时什么都不做
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # 用 Adam 更新参数
        optimizer.step()

        # 统计损失（乘以batch大小，最后除以总样本数，得到真正的样本平均）
        total_loss += loss.item() * images.size(0)

        # 统计正确数
        # torch.max(outputs, dim=1) 返回每行最大值及其索引
        # 索引就是预测的类别编号
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()
        total   += labels.size(0)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy


# ============================================================
# 保存训练曲线图
# ============================================================
def save_curves(loss_list, acc_list, lr_list, model_name, results_dir):
    """
    保存 Loss / Accuracy / LearningRate 三张训练曲线图

    与形状版的差异：
        文件名格式从 model_vX_xxx.png
                  改为 colour_vX_xxx.png
        避免与形状模型的结果文件混淆

    参数：
        loss_list   : 每epoch的loss列表
        acc_list    : 每epoch的accuracy列表
        lr_list     : 每epoch的学习率列表
        model_name  : 'v1'/'v2'/'v3'
        results_dir : 图片保存目录
    """
    os.makedirs(results_dir, exist_ok=True)
    epochs = range(1, len(loss_list) + 1)

    # ── Loss 曲线 ─────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, loss_list, color='steelblue', marker='o',
             markersize=3, linewidth=1.5, label='Train Loss')
    plt.title(
        f'Colour Model {model_name.upper()} — Training Loss',
        fontsize=14
    )
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    loss_path = os.path.join(
        results_dir, f"colour_{model_name}_loss.png"   # colour_ 前缀
    )
    plt.savefig(loss_path, dpi=150)
    plt.close()
    print(f"  Loss 曲线已保存：{loss_path}")

    # ── Accuracy 曲线 ─────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, acc_list, color='tomato', marker='o',
             markersize=3, linewidth=1.5, label='Train Accuracy')
    plt.title(
        f'Colour Model {model_name.upper()} — Training Accuracy',
        fontsize=14
    )
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    acc_path = os.path.join(
        results_dir, f"colour_{model_name}_acc.png"
    )
    plt.savefig(acc_path, dpi=150)
    plt.close()
    print(f"  Acc  曲线已保存：{acc_path}")

    # ── 学习率曲线 ────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_list, color='forestgreen', marker='o',
             markersize=3, linewidth=1.5, label='Learning Rate')
    plt.title(
        f'Colour Model {model_name.upper()} — Learning Rate Schedule',
        fontsize=14
    )
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    # 学习率跨越多个数量级时，用对数坐标更清晰
    plt.yscale('log')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    lr_path = os.path.join(
        results_dir, f"colour_{model_name}_lr.png"
    )
    plt.savefig(lr_path, dpi=150)
    plt.close()
    print(f"  LR   曲线已保存：{lr_path}")


# ============================================================
# 主函数
# ============================================================
def main():

    # ── 解析命令行参数 ────────────────────────────────────────
    parser = argparse.ArgumentParser(description="乐高颜色识别模型训练脚本")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['v1', 'v2', 'v3'],
        help="选择训练的模型版本：v1 / v2 / v3"
    )

    # ★ 新增3：--lr 命令行参数 ★
    # 允许手动指定学习率，不传则根据模型版本自动选择最优值
    # 用法举例：
    #   python train/train_colour.py --model v1          → 自动用 0.0002
    #   python train/train_colour.py --model v2          → 自动用 0.001
    #   python train/train_colour.py --model v1 --lr 0.0001  → 手动覆盖
    parser.add_argument(
        '--lr',
        type=float,
        default=None,   # 默认 None，表示"用模型专属默认值"
        help="学习率（不传则根据模型版本自动选择："
             "V1=0.0002, V2/V3=0.001）"
    )

    args = parser.parse_args()
    model_name = args.model

    # 确定最终学习率：命令行传了就用命令行的，没传就用模型专属默认值
    if args.lr is not None:
        CONFIG["learning_rate"] = args.lr
    else:
        CONFIG["learning_rate"] = MODEL_DEFAULT_LR[model_name]

    print("\n" + "=" * 60)
    print(f"  乐高颜色识别 — 训练 Colour Model {model_name.upper()}")
    print("=" * 60)
    print(f"  学习率：{CONFIG['learning_rate']}（"
          f"{'命令行指定' if args.lr is not None else '模型默认值'}）")

    # ── 路径设置（全部基于ROOT_DIR，跨平台通用）──────────────────
    data_dir       = os.path.join(ROOT_DIR, CONFIG["data_dir"])
    checkpoint_dir = os.path.join(ROOT_DIR, CONFIG["checkpoint_dir"])
    results_dir    = os.path.join(ROOT_DIR, CONFIG["results_dir"])

    # 自动创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir,    exist_ok=True)

    # ── 设备检测 ──────────────────────────────────────────────
    # 自动检测是否有可用GPU，有则用GPU，没有则用CPU
    # AutoDL云端有GPU，会自动切换到 'cuda'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] 使用：{device}")
    if device.type == 'cuda':
        # 打印GPU型号，方便确认云端环境
        print(f"       GPU型号：{torch.cuda.get_device_name(0)}")

    # ── 加载数据 ──────────────────────────────────────────────
    # 完全交给 utils_colour/dataset_colour.py，train 本身零增强代码
    print(f"\n[数据] 加载中...")
    train_loader, num_classes = get_train_loader_colour(
        data_dir    = data_dir,
        batch_size  = CONFIG["batch_size"],
        num_workers = CONFIG["num_workers"],
    )
    print(f"[数据] 加载完成，类别数：{num_classes}")

    # ── 加载模型 ──────────────────────────────────────────────
    print(f"\n[模型] 初始化 Colour Model {model_name.upper()}...")
    model = load_model(model_name, num_classes=CONFIG["num_classes"])
    model = model.to(device)  # 把模型参数搬到GPU

    # ── 定义损失函数 ──────────────────────────────────────────
    # CrossEntropyLoss：多分类标准损失函数
    # 内部包含 softmax，输出层不需要再加 softmax
    criterion = nn.CrossEntropyLoss()

    # ── 定义优化器 ────────────────────────────────────────────
    # Adam：自适应学习率，收敛稳定，深度学习首选
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    # ── 定义学习率调度器 ──────────────────────────────────────
    # CosineAnnealingLR：余弦退火
    # 学习率从初始值按余弦曲线平滑下降到接近 0
    # 好处：前期学习快，后期微调精细，比固定学习率效果更好
    # T_max = num_epochs 表示在整个训练周期内完成一次余弦周期
    # 颜色版 T_max=30（形状版是50）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = CONFIG["num_epochs"],
        eta_min = 1e-6   # 学习率下降的最低值，不会降到0
    )

    # ── 开始训练循环 ──────────────────────────────────────────
    print(f"\n[训练] 开始，共 {CONFIG['num_epochs']} 个 epoch")
    print("-" * 60)

    loss_history = []  # 记录每epoch的loss，最后画图用
    acc_history  = []  # 记录每epoch的accuracy
    lr_history   = []  # 记录每epoch的学习率
    best_acc     = 0.0 # 记录历史最高准确率，用于保存最优模型

    for epoch in range(1, CONFIG["num_epochs"] + 1):

        epoch_start = time.time()  # 记录epoch开始时间

        # 执行一轮训练，返回本epoch的平均loss和准确率
        avg_loss, accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 更新学习率（每个epoch结束后调用一次）
        scheduler.step()

        # 记录历史
        loss_history.append(avg_loss)
        acc_history.append(accuracy)
        lr_history.append(optimizer.param_groups[0]['lr'])

        # 计算本epoch耗时
        epoch_time = time.time() - epoch_start

        # 获取当前学习率（用于打印，方便观察是否正常衰减）
        current_lr = optimizer.param_groups[0]['lr']

        # 打印本epoch信息
        print(
            f"Epoch [{epoch:>3}/{CONFIG['num_epochs']}] "
            f"Loss: {avg_loss:.4f}  "
            f"Acc: {accuracy:.2f}%  "
            f"LR: {current_lr:.6f}  "
            f"时间: {epoch_time:.1f}s"
        )

        # ── 保存最优模型 ──────────────────────────────────────
        # 每当出现更高的训练准确率，就保存一次权重
        # 文件名加 colour_ 前缀，与形状模型权重文件严格区分
        # 例：colour_v1_best.pth（形状版是 model_v1_best.pth）
        if accuracy > best_acc:
            best_acc  = accuracy
            save_path = os.path.join(
                checkpoint_dir, f"colour_{model_name}_best.pth"
            )
            # 只保存模型权重（state_dict），不保存整个模型对象
            # 好处：文件小，跨平台兼容性好
            torch.save(model.state_dict(), save_path)
            print(
                f"  ✅ 新最优模型已保存"
                f"（Acc: {best_acc:.2f}%）→ {save_path}"
            )

    # ── 训练结束 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  训练完成！最终最优准确率：{best_acc:.2f}%")
    print("=" * 60)

    # 保存训练曲线图
    print("\n[结果] 保存训练曲线图...")
    save_curves(
        loss_history, acc_history, lr_history,
        model_name, results_dir
    )

    print("\n全部完成！")


# ── 程序入口 ──────────────────────────────────────────────────
# 只有直接运行此脚本时才执行 main()
# 被其他文件 import 时不会自动执行
if __name__ == "__main__":
    main()
