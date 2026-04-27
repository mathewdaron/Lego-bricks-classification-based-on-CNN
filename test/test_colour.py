# ============================================================
# test/test_colour.py
# 乐高颜色识别模型 —— 测试脚本（V1 / V2 / V3 通用）
#
# 用法（在项目根目录下运行）：
#   python test/test_colour.py --model v1
#   python test/test_colour.py --model v2
#   python test/test_colour.py --model v3
#
# 前提：
#   1. 对应模型已训练完毕，checkpoints/colour_vX_best.pth 存在
#   2. 测试集已准备好，放在 test/test_dataset_colour/ 下
#      结构：每个类别一个子文件夹（colour_00 ~ colour_08）
#
# 输出（全部保存到 results/ 目录）：
#   混淆矩阵图         colour_vX_confusion_matrix.png
#   各类别准确率图     colour_vX_per_class_acc.png
#   错误样本可视化图   colour_vX_error_samples.png
#   测试结果汇总       colour_vX_test_result.txt（供compare脚本读取）
# ============================================================

import os
import sys
import time
import argparse

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')   # 非交互式后端，服务器/云端环境必须
import matplotlib.pyplot as plt

# ── 把项目根目录加入Python路径 ────────────────────────────────
# __file__ 是当前脚本路径（test/test_colour.py）
# dirname 两次 → 得到项目根目录
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ============================================================
# 配置
# ============================================================
CONFIG = {
    # 测试集路径（相对项目根目录）
    "test_data_dir"  : os.path.join("test", "test_dataset_colour"),
    "checkpoint_dir" : "checkpoints",
    "results_dir"    : "results",
    "num_classes"    : 9,
    "batch_size"     : 32,
    # 测试时固定单进程，避免各种环境兼容问题
    "num_workers"    : 0,
}

# 9个颜色类别的可读名称
# 顺序必须和 colour_00~colour_08 文件夹一一对应
CLASS_NAMES = [
    "Red",        # colour_00：大红（色相 0°）
    "Yellow",     # colour_01：明黄（色相 60°）
    "Green",      # colour_02：绿色（色相 120°）
    "Cyan",       # colour_03：青色（色相 180°）
    "Blue",       # colour_04：蓝色（色相 240°）
    "Rose",       # colour_05：玫瑰色（色相 300°）
    "Gray",       # colour_06：灰色（去饱和）
    "White",      # colour_07：白色（手动截图）
    "Black",      # colour_08：黑色（手动截图）
]


# ============================================================
# 第一部分：测试集数据加载
# ============================================================
class TestDatasetColour(Dataset):
    """
    颜色测试集数据集类

    与训练集的 LegoColourDataset 结构相同，但：
    1. 不做任何随机增强，只做标准预处理（Resize + ToTensor + Normalize）
    2. 额外保存原始图片路径，供错误样本可视化使用

    注意：Normalize 的 mean/std 必须和训练时完全一致！
    如果参数不同，模型看到的输入分布就和训练时不匹配，
    导致准确率大幅下降。
    """

    # 测试时的变换：无任何随机性，只做固定预处理
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
            data_dir : 测试集根目录（test/test_dataset_colour/）
        """
        self.data_dir = data_dir

        # 扫描所有类别文件夹（colour_00 ~ colour_08）
        # sorted() 保证顺序固定，和 CLASS_NAMES 对应
        self.classes = sorted([
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d))
        ])
        self.class_to_idx = {
            cls: idx for idx, cls in enumerate(self.classes)
        }

        # 扫描所有图片，同时保存路径（用于错误可视化）
        self.samples = []   # 每个元素：(图片路径, 标签整数)
        for class_name in self.classes:
            class_dir = os.path.join(data_dir, class_name)
            label     = self.class_to_idx[class_name]
            for img_file in sorted(os.listdir(class_dir)):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(class_dir, img_file)
                    self.samples.append((img_path, label))

        print(f"颜色测试集加载完成：")
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
    加载颜色识别模型结构并读取训练好的权重

    参数：
        model_name      : 'v1' / 'v2' / 'v3'
        num_classes     : 类别数（9）
        checkpoint_path : .pth 权重文件路径
        device          : 运行设备（cpu / cuda）

    返回：
        model : 加载好权重的模型，已切换到 eval 模式
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

    # 加载训练好的权重
    # map_location 确保无论权重在 GPU 上保存，都能加载到当前设备
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)

    # 切换到评估模式（eval mode）
    # 这一步必须做！作用：
    # - Dropout 层关闭（不再随机丢弃神经元）
    # - BatchNorm 层使用训练期间统计的全局均值/方差
    # 不调用的话，测试结果会不稳定且偏低
    model.eval()
    model = model.to(device)

    print(f"  颜色模型权重加载成功：{checkpoint_path}")
    return model


# ============================================================
# 第三部分：推理与结果收集
# ============================================================
def run_inference(model, loader, dataset, device, num_classes):
    """
    对整个测试集做推理，收集所有预测结果

    参数：
        model       : 已加载权重的模型（eval 模式）
        loader      : 测试集 DataLoader
        dataset     : 测试集 Dataset（用于获取图片路径）
        device      : 运行设备
        num_classes : 类别数（9）

    返回：
        all_preds   : 所有预测标签，numpy 数组，shape (N,)
        all_labels  : 所有真实标签，numpy 数组，shape (N,)
        all_paths   : 所有图片路径列表，长度 N
        total_time  : 总推理耗时（秒）
        avg_time_ms : 单张图片平均推理耗时（毫秒）
    """
    all_preds  = []
    all_labels = []
    all_paths  = []
    total_time = 0.0

    # torch.no_grad()：测试时关闭梯度计算
    # 好处：节省显存约30%，推理速度提升30%~50%
    with torch.no_grad():

        sample_idx = 0  # 追踪当前处理到第几张图片（用于取路径）

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # 计时：只计算模型前向推理时间，不含数据加载
            t_start = time.time()
            outputs = model(images)
            # GPU 异步执行，必须 synchronize 后再计时，否则时间偏短
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_end = time.time()
            total_time += (t_end - t_start)

            # 取 logits 最大值对应的索引作为预测类别
            _, predicted = torch.max(outputs, dim=1)

            # 收集结果（移回 CPU 转成 numpy）
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 收集对应图片路径
            batch_size = images.size(0)
            for i in range(batch_size):
                all_paths.append(dataset.get_img_path(sample_idx + i))
            sample_idx += batch_size

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_samples   = len(all_labels)
    avg_time_ms = (total_time / n_samples) * 1000

    return all_preds, all_labels, all_paths, total_time, avg_time_ms


# ============================================================
# 第四部分：各类可视化图生成函数
# ============================================================

def plot_confusion_matrix(all_labels, all_preds, class_names,
                          model_name, results_dir):
    """
    绘制并保存 9×9 混淆矩阵

    与形状版的主要区别：
    - figsize 更小（9×9 比 45×45 紧凑很多）
    - 字体更大（格子更大，可以写得清晰）
    - 格子内数字字号更大

    混淆矩阵读法：
    - 行 = 真实类别，列 = 预测类别
    - 对角线 = 预测正确（越亮越好）
    - 非对角线 = 混淆项（颜色越深代表越容易被误判成该列类别）
    """
    n = len(class_names)   # 9

    # 手动构建混淆矩阵
    # cm[i][j] = 真实类别为 i，被预测为 j 的样本数
    cm = np.zeros((n, n), dtype=int)
    for true, pred in zip(all_labels, all_preds):
        cm[true][pred] += 1

    # 行归一化：每行除以该行总数，得到各类别的识别比例
    # 消除各类样本数不同导致的颜色偏差
    cm_norm   = cm.astype(float)
    row_sums  = cm.sum(axis=1, keepdims=True)
    row_sums  = np.where(row_sums == 0, 1, row_sums)   # 防止除以0
    cm_norm   = cm_norm / row_sums

    # 颜色版矩阵只有 9×9，可以用更大的图和字体
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 坐标轴标签
    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    # 字体比形状版大（9个类，格子宽，字体12可以看清）
    ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=12)
    ax.set_yticklabels(class_names, fontsize=12)

    ax.set_xlabel('Predicted Label', fontsize=13)
    ax.set_ylabel('True Label', fontsize=13)
    ax.set_title(
        f'Colour Model {model_name.upper()} — Confusion Matrix',
        fontsize=14, pad=15
    )

    # 在每个格子里写实际样本数
    # 字号10，9×9 矩阵格子够大，数字清晰可读
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            if cm[i][j] > 0:
                ax.text(j, i, str(cm[i][j]),
                        ha='center', va='center', fontsize=10,
                        color='white' if cm_norm[i][j] > thresh else 'black')

    plt.tight_layout()
    # 文件名加 colour_ 前缀，与形状版严格区分
    save_path = os.path.join(
        results_dir, f"colour_{model_name}_confusion_matrix.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  混淆矩阵已保存：{save_path}")


def plot_per_class_accuracy(all_labels, all_preds, class_names,
                            model_name, results_dir):
    """
    绘制每个颜色类别的准确率横向柱状图

    颜色版与形状版区别：
    - 只有 9 个类别，图更紧凑（figsize 更小）
    - 字体更大，每根柱子更粗，数值更清晰
    - 同样用三色区分：绿(≥80%) / 橙(60~80%) / 红(<60%)
    """
    n = len(class_names)   # 9

    # 计算每个颜色类别的准确率
    per_class_acc = []
    for i in range(n):
        mask  = (all_labels == i)
        total = mask.sum()
        if total == 0:
            per_class_acc.append(0.0)
        else:
            correct = (all_preds[mask] == i).sum()
            per_class_acc.append(100.0 * correct / total)

    # 按准确率从低到高排序（从上往下读，最差的在最顶，最好的在底部）
    sorted_idx   = np.argsort(per_class_acc)
    sorted_acc   = [per_class_acc[i] for i in sorted_idx]
    sorted_names = [class_names[i] for i in sorted_idx]

    # 三色区分准确率高低
    colors = []
    for acc in sorted_acc:
        if acc >= 80:
            colors.append('mediumseagreen')   # 绿色：良好
        elif acc >= 60:
            colors.append('sandybrown')       # 橙色：一般
        else:
            colors.append('tomato')           # 红色：需关注

    # 颜色版只有9行，图可以矮一些
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(range(n), sorted_acc, color=colors,
                   edgecolor='white', linewidth=0.8, height=0.6)

    # 在每根柱子末端写准确率数值
    for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
        ax.text(
            min(acc + 1, 99),                    # x：柱子末端右侧一点
            bar.get_y() + bar.get_height() / 2,  # y：柱子中间
            f'{acc:.1f}%',
            va='center', ha='left', fontsize=10   # 比形状版字体更大
        )

    ax.set_yticks(range(n))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.set_xlim(0, 112)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title(
        f'Colour Model {model_name.upper()} — Per-Class Accuracy',
        fontsize=14
    )
    # 80% 参考线
    ax.axvline(x=80, color='gray', linestyle='--',
               linewidth=1, alpha=0.6, label='80% threshold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(
        results_dir, f"colour_{model_name}_per_class_acc.png"
    )
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  类别准确率图已保存：{save_path}")

    return per_class_acc   # 返回供汇总 txt 使用


def plot_error_samples(all_labels, all_preds, all_paths,
                       class_names, model_name, results_dir,
                       max_errors=9):
    """
    错误样本可视化：每个颜色类别最多展示 1 个预测错误的样本

    颜色版与形状版区别：
    - max_errors 从 45 降到 9（最多9个颜色类别）
    - n_cols 从 min(9, n) 改为 min(5, n)（图更宽松）
    - 格子更大，图片和标题更清晰

    每个格子显示：
    - 积木图片本身（直接读原始文件，无需反标准化）
    - 标题：真实颜色 → 预测颜色（红色字体）
    """
    # 收集错误样本：每个类别只取第一个错误
    error_samples = {}   # {真实类别idx: (图片路径, 预测类别idx)}

    for path, true, pred in zip(all_paths, all_labels, all_preds):
        if true != pred and true not in error_samples:
            error_samples[true] = (path, pred)
        if len(error_samples) >= max_errors:
            break

    if len(error_samples) == 0:
        print("  ✅ 没有错误样本，颜色模型在测试集上全部预测正确！")
        return

    n_errors = len(error_samples)
    print(f"  共发现 {n_errors} 个颜色类别存在预测错误")

    # 布局：最多每行5个（颜色版9个类，1行放不下时自动折两行）
    n_cols = min(5, n_errors)
    n_rows = (n_errors + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * 2.8, n_rows * 3.2))

    # 统一变成 2D 数组方便用 [row][col] 索引
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for plot_idx, (true_cls, (img_path, pred_cls)) in \
            enumerate(sorted(error_samples.items())):
        row = plot_idx // n_cols
        col = plot_idx  % n_cols
        ax  = axes[row][col]

        # 直接读取原始图片并缩放（不经过标准化变换，颜色更真实）
        img = Image.open(img_path).convert("RGB")
        img = img.resize((224, 224))

        ax.imshow(img)
        ax.axis('off')

        # 标题：真实颜色 → 预测颜色（红色字体提示错误）
        true_name = class_names[true_cls]
        pred_name = class_names[pred_cls]
        ax.set_title(
            f'真: {true_name}\n预: {pred_name}',
            fontsize=9, color='red', pad=3
        )

    # 隐藏多余的空格子
    total_slots = n_rows * n_cols
    for i in range(n_errors, total_slots):
        row = i // n_cols
        col = i  % n_cols
        axes[row][col].axis('off')

    fig.suptitle(
        f'Colour Model {model_name.upper()} — Error Samples（每类最多1个）',
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    save_path = os.path.join(
        results_dir, f"colour_{model_name}_error_samples.png"
    )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  错误样本图已保存：{save_path}")


# ============================================================
# 第五部分：保存汇总结果到 txt（供 compare 脚本读取）
# ============================================================
def save_test_result(model_name, total_acc, per_class_acc,
                     avg_time_ms, results_dir):
    """
    把颜色测试结果写入标准格式 txt

    格式与 test_shape.py 完全一致（compare_colour.py 按行解析读取）：
        model=v1
        total_accuracy=96.3333
        avg_inference_time_ms=0.5678
        per_class_acc=98.00,95.00,...（9个值，逗号分隔）

    参数：
        model_name    : 'v1'/'v2'/'v3'
        total_acc     : 总体准确率（float，百分比，例如 96.33）
        per_class_acc : 每类准确率列表（9个 float）
        avg_time_ms   : 单张图片平均推理耗时（毫秒）
        results_dir   : 保存目录
    """
    # 文件名加 colour_ 前缀，与形状版严格区分
    save_path = os.path.join(
        results_dir, f"colour_{model_name}_test_result.txt"
    )

    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(f"model={model_name}\n")
        f.write(f"total_accuracy={total_acc:.4f}\n")
        f.write(f"avg_inference_time_ms={avg_time_ms:.4f}\n")
        # 每类准确率写成一行，逗号分隔，保留4位小数
        f.write(f"per_class_acc={','.join(f'{a:.4f}' for a in per_class_acc)}\n")

    print(f"  测试结果已保存：{save_path}")


# ============================================================
# 第六部分：主函数
# ============================================================
def main():

    # ── 命令行参数解析 ────────────────────────────────────────
    parser = argparse.ArgumentParser(description="乐高颜色识别模型测试脚本")
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
    print(f"  乐高颜色识别 — 测试 Colour Model {model_name.upper()}")
    print("=" * 60)

    # ── 路径设置 ──────────────────────────────────────────────
    test_data_dir   = os.path.join(ROOT_DIR, CONFIG["test_data_dir"])
    checkpoint_dir  = os.path.join(ROOT_DIR, CONFIG["checkpoint_dir"])
    results_dir     = os.path.join(ROOT_DIR, CONFIG["results_dir"])
    # 权重文件名加 colour_ 前缀
    checkpoint_path = os.path.join(
        checkpoint_dir, f"colour_{model_name}_best.pth"
    )
    os.makedirs(results_dir, exist_ok=True)

    # ── 检查测试集和权重文件是否存在 ──────────────────────────
    if not os.path.exists(test_data_dir):
        print(f"\n❌ 错误：颜色测试集目录不存在：{test_data_dir}")
        print(f"   请先制作测试集，结构参考：")
        print(f"   test/test_dataset_colour/")
        print(f"   ├── colour_00/  (若干张红色测试图)")
        print(f"   ├── colour_01/  (若干张黄色测试图)")
        print(f"   └── ...（共9个子文件夹）")
        return

    if not os.path.exists(checkpoint_path):
        print(f"\n❌ 错误：颜色模型权重不存在：{checkpoint_path}")
        print(f"   请先运行训练脚本：python train/train_colour.py --model {model_name}")
        return

    # ── 设备检测 ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[设备] 使用：{device}")

    # ── 加载测试集 ────────────────────────────────────────────
    print(f"\n[数据] 加载颜色测试集...")
    test_dataset = TestDatasetColour(test_data_dir)
    test_loader  = DataLoader(
        test_dataset,
        batch_size  = CONFIG["batch_size"],
        shuffle     = False,    # 测试集不打乱，保证路径索引与图片一一对应
        num_workers = CONFIG["num_workers"],
        pin_memory  = (device.type == 'cuda'),
    )

    # ── 加载模型 ──────────────────────────────────────────────
    print(f"\n[模型] 加载 Colour Model {model_name.upper()}...")
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
    print(f"  颜色测试结果汇总")
    print(f"{'='*60}")
    print(f"  总样本数：    {total_samples}")
    print(f"  正确预测数：  {total_correct}")
    print(f"  总体准确率：  {total_acc:.2f}%")
    print(f"  总推理耗时：  {total_time:.2f} 秒")
    print(f"  单张平均耗时：{avg_time_ms:.2f} ms")
    print(f"{'='*60}")

    # ── 生成所有可视化图 ──────────────────────────────────────
    print(f"\n[结果] 生成可视化图表...")

    # 1. 混淆矩阵（9×9，字体更大更清晰）
    plot_confusion_matrix(
        all_labels, all_preds, CLASS_NAMES, model_name, results_dir
    )

    # 2. 各颜色类别准确率横向柱状图
    per_class_acc = plot_per_class_accuracy(
        all_labels, all_preds, CLASS_NAMES, model_name, results_dir
    )

    # 3. 错误样本可视化（每类最多1个，最多9张）
    plot_error_samples(
        all_labels, all_preds, all_paths,
        CLASS_NAMES, model_name, results_dir
    )

    # 4. 保存汇总 txt（格式与形状版完全一致，供 compare_colour.py 读取）
    save_test_result(
        model_name, total_acc, per_class_acc, avg_time_ms, results_dir
    )

    print(f"\n✅ 全部完成！所有颜色测试结果已保存到：{results_dir}")


# ── 程序入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()
