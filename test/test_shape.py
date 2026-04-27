# ============================================================
# test/test_shape.py
# 乐高形状识别模型 —— 测试脚本（V1 / V2 / V3 通用）
#
# 用法（在项目根目录下运行）：
#   python test/test_shape.py --model v1
#   python test/test_shape.py --model v2
#   python test/test_shape.py --model v3
#
# 前提：
#   1. 对应模型已训练完毕，checkpoints/model_vX_best.pth 存在
#   2. 测试集已准备好，放在 test/test_dataset_shape/ 下
#      结构：每个类别一个子文件夹（shape_00 ~ shape_44）
#
# 输出（全部保存到 results/ 目录）：
#   混淆矩阵图         model_vX_confusion_matrix.png
#   各类别准确率图     model_vX_per_class_acc.png
#   错误样本可视化图   model_vX_error_samples.png
#   测试结果汇总       model_vX_test_result.txt（供compare脚本读取）
# ============================================================

import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── 把项目根目录加入Python路径 ────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 测试集路径（相对项目根目录）
    "test_data_dir"  : os.path.join("test", "test_dataset_shape"),
    "checkpoint_dir" : "checkpoints",
    "results_dir"    : "results",
    "num_classes"    : 45,
    "batch_size"     : 32,
    # 测试时固定用单进程，避免各种环境兼容问题
    "num_workers"    : 0,
}

# 45个类别的可读名称，用于坐标轴标签
# 顺序必须和 shape_00~shape_44 文件夹一一对应
CLASS_NAMES = [
    "Brick1x1",    "Brick1x2",    "Brick1x3",    "Brick1x4",
    "Brick1x6",    "Brick1x8",    "Brick1x10",   "Brick1x12",
    "Brick2x2C",   "Brick2x2",    "Brick2x3",    "Brick2x4",
    "Brick2x6",    "Brick2x8",    "Brick2x10",   "BrickRnd1x1",
    "BrickRnd2x2", "BrickDome",   "BrickStu1x1", "BrickStu1x2",
    "BrickStu1x4", "Plate1x1",    "Plate1x2",    "Plate1x3",
    "Plate1x4",    "Plate1x6",    "Plate1x8",    "Plate1x10",
    "Plate1x12",   "Plate2x2C",   "Plate2x2",    "Plate2x3",
    "Plate2x4",    "Plate2x6",    "Plate2x8",    "Plate2x10",
    "Plate3x3",    "Plate4x4",    "Plate6x6",    "PlateStu1x2",
    "PlateGroove", "PlateHdl",    "PlateRnd1x1", "PlateRnd2x2",
    "PlateRnd4x4",
]


# ============================================================
# 第一部分：测试集数据加载
# ============================================================
class TestDataset(Dataset):
    """
    测试集数据集类

    与训练集的 LegoShapeDataset 结构相同，但：
    1. 不做任何随机增强，只做标准预处理
    2. 额外保存原始图片路径，供错误样本可视化使用
    """

    # 测试集变换：无随机增强，仅标准化
    # 必须和训练集使用完全相同的 Normalize 参数
    # 否则模型看到的输入分布与训练时不一致，准确率会严重下降
    TRANSFORM = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    def __init__(self, data_dir):
        """
        参数：
            data_dir : 测试集根目录（test/test_dataset_shape/）
        """
        self.data_dir = data_dir

        # 扫描类别文件夹
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # 扫描所有图片，同时保存路径（用于错误可视化）
        self.samples = []   # (图片路径, 标签)
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            label     = self.class_to_idx[class_name]
            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, label))

        print(f"测试集加载完成：")
        print(f"  类别数：  {len(self.classes)}")
        print(f"  图片总数：{len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.TRANSFORM(img)
        return img, label

    def get_img_path(self, idx):
        """返回指定索引图片的原始路径（用于错误可视化）"""
        return self.samples[idx][0]


# ============================================================
# 第二部分：模型加载
# ============================================================
def load_model(model_name, num_classes, checkpoint_path, device):
    """
    加载模型结构并读取训练好的权重

    参数：
        model_name      : 'v1' / 'v2' / 'v3'
        num_classes     : 类别数（45）
        checkpoint_path : .pth 权重文件路径
        device          : 运行设备

    返回：
        model : 加载好权重的模型，已切换到eval模式
    """
    if model_name == 'v1':
        from models_shape.model_v1 import create_model_v1
        model = create_model_v1(num_classes=num_classes)
    elif model_name == 'v2':
        from models_shape.model_v2 import create_model_v2
        model = create_model_v2(num_classes=num_classes)
    elif model_name == 'v3':
        from models_shape.model_v3 import create_model_v3
        model = create_model_v3(num_classes=num_classes)

    # 加载训练好的权重
    # map_location 确保无论权重在GPU上保存，都能正确加载到当前设备
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 切换到评估模式
    # 评估模式下：Dropout 关闭（不随机丢弃），BatchNorm 使用训练期间
    # 统计的全局均值和方差，而不是当前batch的统计量
    # 必须调用，否则测试结果不稳定且偏低
    model.eval()
    model = model.to(device)

    print(f"  模型权重加载成功：{checkpoint_path}")
    return model


# ============================================================
# 第三部分：推理与结果收集
# ============================================================
def run_inference(model, loader, dataset, device, num_classes):
    """
    对整个测试集做推理，收集所有预测结果

    参数：
        model       : 已加载权重的模型（eval模式）
        loader      : 测试集 DataLoader
        dataset     : 测试集 Dataset（用于获取图片路径）
        device      : 运行设备
        num_classes : 类别数

    返回：
        all_preds   : 所有预测标签，numpy数组，shape (N,)
        all_labels  : 所有真实标签，numpy数组，shape (N,)
        all_paths   : 所有图片路径列表，长度N
        total_time  : 总推理耗时（秒）
        avg_time_ms : 单张图片平均推理耗时（毫秒）
    """
    all_preds  = []
    all_labels = []
    all_paths  = []

    total_time = 0.0

    # torch.no_grad()：测试时不需要计算梯度
    # 关闭梯度计算可以节省显存和加速推理（约快30%~50%）
    with torch.no_grad():

        sample_idx = 0  # 追踪当前处理到第几张图片

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 计时：只计算模型推理时间，不含数据加载
            t_start = time.time()
            outputs = model(images)
            # 如果用GPU，需要等待GPU计算完成再计时
            # 否则计时会在GPU还没算完时就结束，得到错误的时间
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.time()
            total_time += (t_end - t_start)

            # 取每行最大值的索引作为预测类别
            _, predicted = torch.max(outputs, dim=1)

            # 收集结果（移回CPU转numpy）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 收集对应的图片路径
            batch_size = images.size(0)
            for i in range(batch_size):
                all_paths.append(dataset.get_img_path(sample_idx + i))
            sample_idx += batch_size

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_samples      = len(all_labels)
    avg_time_ms    = (total_time / n_samples) * 1000

    return all_preds, all_labels, all_paths, total_time, avg_time_ms


# ============================================================
# 第四部分：各类可视化图生成函数
# ============================================================

def plot_confusion_matrix(all_labels, all_preds, class_names,
                          model_name, results_dir):
    """
    绘制并保存混淆矩阵

    混淆矩阵说明：
    - 行 = 真实类别，列 = 预测类别
    - 对角线 = 预测正确的数量（越亮越好）
    - 非对角线 = 预测错误，颜色越深说明这两类越容易混淆

    参数：
        all_labels  : 所有真实标签
        all_preds   : 所有预测标签
        class_names : 45个类别的可读名称列表
        model_name  : 'v1'/'v2'/'v3'
        results_dir : 图片保存目录
    """
    n = len(class_names)

    # 手动构建混淆矩阵
    # cm[i][j] = 真实类别为i，被预测为j的样本数
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # 归一化：每行除以该行总数，得到比例（方便看准确率）
    # 避免各类别样本数不同导致颜色偏差
    cm_norm = cm.astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    # 防止除以0（某类没有测试样本时）
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm_norm / row_sums

    # 绘图
    fig, ax = plt.subplots(figsize=(20, 18))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 坐标轴标签
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(
        f'Model {model_name.upper()} — Confusion Matrix',
        fontsize=14, pad=15
    )

    # 在每个格子里写数字（样本数少时可读性好）
    # 只在数值大于0时写，避免满屏0
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i][j] > 0:
                ax.text(j, i, str(cm[i][j]),
                        ha='center', va='center', fontsize=5,
                        color='white' if cm_norm[i][j] > thresh else 'black')

    plt.tight_layout()
    save_path = os.path.join(
        results_dir, f"model_{model_name}_confusion_matrix.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存：{save_path}")


def plot_per_class_accuracy(all_labels, all_preds, class_names,
                            model_name, results_dir):
    """
    绘制每个类别的单独准确率横向柱状图

    用途：直观看出哪些形状容易识别，哪些形状容易出错
    例如：Brick 1x6 和 Brick 1x8 可能因为外形相似而准确率低

    参数：同 plot_confusion_matrix
    """
    n = len(class_names)

    # 计算每类准确率
    per_class_acc = []
    for i in range(n):
        # 该类别的所有样本
        mask  = (all_labels == i)
        total = mask.sum()
        if total == 0:
            per_class_acc.append(0.0)
        else:
            correct = (all_preds[mask] == i).sum()
            per_class_acc.append(100.0 * correct / total)

    # 按准确率排序，方便阅读
    sorted_idx = np.argsort(per_class_acc)
    sorted_acc = [per_class_acc[i] for i in sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]

    # 用颜色区分高低准确率
    # 准确率>=80% → 绿色；60~80% → 橙色；<60% → 红色
    colors = []
    for acc in sorted_acc:
        if acc >= 80:
            colors.append('mediumseagreen')
        elif acc >= 60:
            colors.append('sandybrown')
        else:
            colors.append('tomato')

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 14))
    bars = ax.barh(range(n), sorted_acc, color=colors, edgecolor='white',
                   linewidth=0.5)

    # 在每个柱子末端写数值
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(
            min(acc + 1, 99),   # x位置：柱子末端右侧1个单位
            bar.get_y() + bar.get_height() / 2,
            f'{acc:.1f}%',
            va='center', ha='left', fontsize=7
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=8)
    ax.set_xlim(0, 108)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(
        f'Model {model_name.upper()} — Per-Class Accuracy',
        fontsize=14
    )
    ax.axvline(x=80, color='gray', linestyle='--',
               linewidth=1, alpha=0.6, label='80% 参考线')
    ax.legend(fontsize=9)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        results_dir, f"model_{model_name}_per_class_acc.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  类别准确率图已保存：{save_path}")

    return per_class_acc  # 返回供汇总txt使用


def plot_error_samples(all_labels, all_preds, all_paths,
                       class_names, model_name, results_dir,
                       max_errors=45):
    """
    错误样本可视化：每个类别最多展示1个预测错误的样本

    每个格子显示：
    - 图片本身
    - 标题：真实类别 → 预测类别（红色）

    参数：
        all_labels  : 所有真实标签
        all_preds   : 所有预测标签
        all_paths   : 所有图片路径
        class_names : 类别名称
        model_name  : 'v1'/'v2'/'v3'
        results_dir : 保存目录
        max_errors  : 最多展示的错误样本数（默认45，每类1个）
    """
    # 收集错误样本：每个类别只取第一个错误
    error_samples = {}  # {真实类别: (图片路径, 预测类别)}

    for path, true, pred in zip(all_paths, all_labels, all_preds):
        if true != pred and true not in error_samples:
            error_samples[true] = (path, pred)
        if len(error_samples) >= max_errors:
            break

    if len(error_samples) == 0:
        print("  ✅ 没有错误样本，模型在测试集上全部预测正确！")
        return

    n_errors = len(error_samples)
    print(f"  共发现 {n_errors} 个类别存在预测错误")

    # 计算子图布局：尽量接近正方形
    n_cols = min(9, n_errors)          # 每行最多9个
    n_rows = (n_errors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 2.2, n_rows * 2.8))

    # 统一成2D数组方便索引
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # 反标准化变换：把模型输入的Tensor还原成可显示的图片
    # 标准化：x' = (x - mean) / std
    # 反标准化：x  = x' * std + mean
    inv_mean = np.array([0.485, 0.456, 0.406])
    inv_std  = np.array([0.229, 0.224, 0.225])

    for plot_idx, (true_cls, (img_path, pred_cls)) in \
            enumerate(sorted(error_samples.items())):
        row = plot_idx // n_cols
        col = plot_idx  % n_cols
        ax  = axes[row][col]

        # 直接读取原始图片（不经过标准化，更清晰）
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))

        ax.imshow(img)
        ax.axis('off')

        # 标题：真实 → 预测（红色警示）
        true_name = class_names[true_cls]
        pred_name = class_names[pred_cls]
        ax.set_title(
            f'真: {true_name}\n预: {pred_name}',
            fontsize=6.5, color='red', pad=2
        )

    # 隐藏多余的子图格子
    total_slots = n_rows * n_cols
    for i in range(n_errors, total_slots):
        row = i // n_cols
        col = i  % n_cols
        axes[row][col].axis('off')

    fig.suptitle(
        f'Model {model_name.upper()} — Error Samples (每类最多1个)',
        fontsize=13, y=1.01
    )
    plt.tight_layout()
    save_path = os.path.join(
        results_dir, f"model_{model_name}_error_samples.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  错误样本图已保存：{save_path}")


# ============================================================
# 第五部分：保存汇总结果到txt（供compare脚本读取）
# ============================================================
def save_test_result(model_name, total_acc, per_class_acc,
                     avg_time_ms, results_dir):
    """
    把测试结果写入标准格式的txt文件

    格式固定，compare_models.py 会按行解析读取

    参数：
        model_name    : 'v1'/'v2'/'v3'
        total_acc     : 总体准确率（float，百分比）
        per_class_acc : 每类准确率列表（45个float）
        avg_time_ms   : 单张图片平均推理耗时（毫秒）
        results_dir   : 保存目录
    """
    save_path = os.path.join(
        results_dir, f"model_{model_name}_test_result.txt"
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"model={model_name}\n")
        f.write(f"total_accuracy={total_acc:.4f}\n")
        f.write(f"avg_inference_time_ms={avg_time_ms:.4f}\n")
        # 每类准确率写成一行，逗号分隔
        f.write(f"per_class_acc={','.join(f'{a:.4f}' for a in per_class_acc)}\n")

    print(f"  测试结果已保存：{save_path}")


# ============================================================
# 第六部分：主函数
# ============================================================
def main():

    # ── 命令行参数 ────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="乐高形状识别模型测试脚本")
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['v1', 'v2', 'v3'],
        help="选择测试哪个模型：v1 / v2 / v3"
    )
    args = parser.parse_args()
    model_name = args.model

    print("\n" + "=" * 60)
    print(f"  乐高形状识别 — 测试 Model {model_name.upper()}")
    print("=" * 60)

    # ── 路径设置 ──────────────────────────────────────────────
    test_data_dir  = os.path.join(ROOT_DIR, CONFIG["test_data_dir"])
    checkpoint_dir = os.path.join(ROOT_DIR, CONFIG["checkpoint_dir"])
    results_dir    = os.path.join(ROOT_DIR, CONFIG["results_dir"])
    checkpoint_path = os.path.join(
        checkpoint_dir, f"model_{model_name}_best.pth"
    )
    os.makedirs(results_dir, exist_ok=True)

    # ── 检查测试集和权重文件是否存在 ──────────────────────────
    if not os.path.exists(test_data_dir):
        print(f"\n❌ 错误：测试集目录不存在：{test_data_dir}")
        print(f"   请先制作测试集，结构参考：")
        print(f"   test/test_dataset_shape/")
        print(f"   ├── shape_00/  (若干张测试图)")
        print(f"   ├── shape_01/")
        print(f"   └── ...")
        return

    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 错误：模型权重不存在：{checkpoint_path}")
        print(f"   请先运行训练脚本：python train/train_shape.py --model {model_name}")
        return

    # ── 设备检测 ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] 使用：{device}")

    # ── 加载测试集 ────────────────────────────────────────────
    print(f"\n[数据] 加载测试集...")
    test_dataset = TestDataset(test_data_dir)
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,   # 测试集不打乱，保证路径索引对应正确
        num_workers = CONFIG["num_workers"],
        pin_memory  = (device.type == 'cuda'),
    )

    # ── 加载模型 ──────────────────────────────────────────────
    print(f"\n[模型] 加载 Model {model_name.upper()}...")
    model = load_model(
        model_name, CONFIG["num_classes"], checkpoint_path, device
    )

    # ── 推理 ──────────────────────────────────────────────────
    print(f"\n[推理] 开始推理，共 {len(test_dataset)} 张图片...")
    all_preds, all_labels, all_paths, total_time, avg_time_ms = \
        run_inference(model, test_loader, test_dataset, device,
                      CONFIG["num_classes"])

    # ── 打印总体结果 ──────────────────────────────────────────
    total_correct = (all_preds == all_labels).sum()
    total_samples = len(all_labels)
    total_acc     = 100.0 * total_correct / total_samples

    print(f"\n{'='*60}")
    print(f"  测试结果汇总")
    print(f"{'='*60}")
    print(f"  总样本数：    {total_samples}")
    print(f"  正确预测数：  {total_correct}")
    print(f"  总体准确率：  {total_acc:.2f}%")
    print(f"  总推理耗时：  {total_time:.2f} 秒")
    print(f"  单张平均耗时：{avg_time_ms:.2f} ms")
    print(f"{'='*60}")

    # ── 生成所有可视化图 ──────────────────────────────────────
    print(f"\n[结果] 生成可视化图表...")

    # 1. 混淆矩阵
    plot_confusion_matrix(
        all_labels, all_preds, CLASS_NAMES, model_name, results_dir
    )

    # 2. 各类别准确率
    per_class_acc = plot_per_class_accuracy(
        all_labels, all_preds, CLASS_NAMES, model_name, results_dir
    )

    # 3. 错误样本可视化
    plot_error_samples(
        all_labels, all_preds, all_paths,
        CLASS_NAMES, model_name, results_dir
    )

    # 4. 保存汇总txt（供compare脚本读取）
    save_test_result(
        model_name, total_acc, per_class_acc, avg_time_ms, results_dir
    )

    print(f"\n✅ 全部完成！所有结果已保存到：{results_dir}")


# ── 程序入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()
