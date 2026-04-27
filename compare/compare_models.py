# ============================================================
# test/compare_models.py
# 三模型横向对比脚本
#
# 用法（在项目根目录下运行）：
#   python test/compare_models.py
#
# 前提：
#   三个模型都已完成测试，results/ 目录下存在：
#     model_v1_test_result.txt
#     model_v2_test_result.txt
#     model_v3_test_result.txt
#
# 输出（保存到 results/ 目录）：
#   model_compare_accuracy.png     三模型总体准确率对比柱状图
#   model_compare_per_class.png    三模型各类别准确率对比折线图
#   model_compare_speed.png        三模型推理速度对比柱状图
#   model_compare_summary.png      四图合一总览图
# ============================================================

import os
import sys

import numpy as np
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
    "results_dir" : os.path.join(ROOT_DIR, "results"),
    "num_classes" : 45,
}

# 三个模型的显示名称和颜色（统一风格）
MODEL_STYLES = {
    'v1': {
        'label' : 'V1  (8层CNN,  无BN)',
        'color' : 'steelblue',
        'marker': 'o',
    },
    'v2': {
        'label' : 'V2  (12层CNN, 含BN)',
        'color' : 'tomato',
        'marker': 's',
    },
    'v3': {
        'label' : 'V3  (迷你ResNet)',
        'color' : 'mediumseagreen',
        'marker': '^',
    },
}

# 45个类别的可读名称（与 test_shape.py 保持完全一致）
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
# 第一部分：读取测试结果txt
# ============================================================
def load_result(model_name, results_dir):
    """
    读取单个模型的测试结果txt文件

    txt格式（由 test_shape.py 生成）：
        model=v1
        total_accuracy=92.3456
        avg_inference_time_ms=1.2345
        per_class_acc=98.00,87.50,...（45个值，逗号分隔）

    参数：
        model_name  : 'v1' / 'v2' / 'v3'
        results_dir : results/ 目录路径

    返回：
        result : 字典，包含 total_accuracy / avg_time_ms / per_class_acc
                 若文件不存在，返回 None
    """
    txt_path = os.path.join(
        results_dir, f"model_{model_name}_test_result.txt"
    )

    if not os.path.exists(txt_path):
        return None

    result = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, value = line.split('=', 1)  # 只分割第一个等号

            if key == 'model':
                result['model'] = value

            elif key == 'total_accuracy':
                result['total_accuracy'] = float(value)

            elif key == 'avg_inference_time_ms':
                result['avg_time_ms'] = float(value)

            elif key == 'per_class_acc':
                result['per_class_acc'] = [
                    float(x) for x in value.split(',')
                ]

    return result


def load_all_results(results_dir):
    """
    加载所有可用模型的测试结果

    返回：
        results      : 字典 {'v1': {...}, 'v2': {...}, 'v3': {...}}
                       某个模型文件不存在时对应值为 None
        available    : 有效结果的模型名称列表，如 ['v1', 'v2', 'v3']
    """
    results   = {}
    available = []

    for name in ['v1', 'v2', 'v3']:
        r = load_result(name, results_dir)
        results[name] = r
        if r is not None:
            available.append(name)
            print(f"  ✅ {name}：总准确率 {r['total_accuracy']:.2f}%，"
                  f"推理耗时 {r['avg_time_ms']:.2f} ms/张")
        else:
            print(f"  ⚠️  {name}：结果文件不存在，跳过")

    return results, available


# ============================================================
# 第二部分：图1 —— 总体准确率对比柱状图
# ============================================================
def plot_accuracy_bar(results, available, results_dir):
    """
    三模型总体准确率并排柱状图

    直观体现：V1 → V2 → V3 的准确率提升幅度
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = np.arange(len(available))
    accs   = [results[m]['total_accuracy'] for m in available]
    colors = [MODEL_STYLES[m]['color']     for m in available]
    labels = [MODEL_STYLES[m]['label']     for m in available]

    bars = ax.bar(x_pos, accs, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.2)

    # 在柱子顶端标注具体数值
    for bar, acc in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Top-1 Accuracy (%)', fontsize=12)
    ax.set_title('Model Comparison — Overall Accuracy', fontsize=13)

    # y轴从最低准确率下方10%开始，突出差异
    min_acc = min(accs)
    ax.set_ylim(max(0, min_acc - 10), 102)

    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "model_compare_accuracy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  总体准确率对比图已保存：{save_path}")


# ============================================================
# 第三部分：图2 —— 各类别准确率对比折线图
# ============================================================
def plot_per_class_comparison(results, available, results_dir):
    """
    三模型在45个类别上的准确率折线对比图

    横轴：45个类别
    纵轴：准确率
    每个模型一条折线

    用途：
    - 看哪些类别在所有模型上都表现差（说明这类形状本身难区分）
    - 看哪些类别V3比V1明显提升（说明残差结构对这类有帮助）
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    x_pos = np.arange(len(CLASS_NAMES))

    for m in available:
        acc_list = results[m]['per_class_acc']
        style    = MODEL_STYLES[m]
        ax.plot(
            x_pos, acc_list,
            color=style['color'],
            marker=style['marker'],
            markersize=4,
            linewidth=1.2,
            label=style['label'],
            alpha=0.85
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=7)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 108)
    ax.set_title(
        'Model Comparison — Per-Class Accuracy', fontsize=13
    )
    ax.axhline(y=80, color='gray', linestyle='--',
               linewidth=1, alpha=0.5, label='80% 参考线')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "model_compare_per_class.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  各类别准确率对比图已保存：{save_path}")


# ============================================================
# 第四部分：图3 —— 推理速度对比柱状图
# ============================================================
def plot_speed_bar(results, available, results_dir):
    """
    三模型单张图片平均推理耗时对比柱状图

    用途：体现模型复杂度与速度的权衡
    V1最快（参数少），V2居中，V3最慢（但残差结构实际
    比参数量预期的更高效）
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = np.arange(len(available))
    times  = [results[m]['avg_time_ms'] for m in available]
    colors = [MODEL_STYLES[m]['color']  for m in available]
    labels = [MODEL_STYLES[m]['label']  for m in available]

    bars = ax.bar(x_pos, times, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.2)

    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f'{t:.2f} ms',
            ha='center', va='bottom', fontsize=11, fontweight='bold'
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel('Avg Inference Time (ms/image)', fontsize=12)
    ax.set_title('Model Comparison — Inference Speed', fontsize=13)
    ax.set_ylim(0, max(times) * 1.3)
    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "model_compare_speed.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  推理速度对比图已保存：{save_path}")


# ============================================================
# 第五部分：图4 —— 四图合一总览图
# ============================================================
def plot_summary(results, available, results_dir):
    """
    把准确率柱状图、类别折线图、速度柱状图、
    以及数值汇总表格合并成一张大图

    这张图适合直接放进 README 或 GitHub 展示
    """
    fig = plt.figure(figsize=(20, 14))

    # 用 GridSpec 灵活控制子图大小
    # 布局：上方一行两个小图（准确率+速度），下方两个大图（类别+表格）
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.32,
        height_ratios=[1, 1.2]
    )

    ax_acc   = fig.add_subplot(gs[0, 0])   # 左上：总体准确率
    ax_speed = fig.add_subplot(gs[0, 1])   # 右上：推理速度
    ax_cls   = fig.add_subplot(gs[1, 0])   # 左下：各类别折线
    ax_table = fig.add_subplot(gs[1, 1])   # 右下：数值汇总表格

    x_pos = np.arange(len(available))

    # ── 左上：总体准确率柱状图 ────────────────────────────────
    accs   = [results[m]['total_accuracy'] for m in available]
    colors = [MODEL_STYLES[m]['color']     for m in available]
    labels = [MODEL_STYLES[m]['label']     for m in available]

    bars = ax_acc.bar(x_pos, accs, color=colors, width=0.5,
                      edgecolor='white', linewidth=1.2)
    for bar, acc in zip(bars, accs):
        ax_acc.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.4,
            f'{acc:.2f}%',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax_acc.set_xticks(x_pos)
    ax_acc.set_xticklabels(labels, fontsize=8.5)
    ax_acc.set_ylabel('Accuracy (%)', fontsize=11)
    ax_acc.set_title('Overall Accuracy', fontsize=12)
    ax_acc.set_ylim(max(0, min(accs) - 10), 105)
    ax_acc.grid(axis='y', alpha=0.3)
    ax_acc.spines['top'].set_visible(False)
    ax_acc.spines['right'].set_visible(False)

    # ── 右上：推理速度柱状图 ──────────────────────────────────
    times = [results[m]['avg_time_ms'] for m in available]

    bars2 = ax_speed.bar(x_pos, times, color=colors, width=0.5,
                          edgecolor='white', linewidth=1.2)
    for bar, t in zip(bars2, times):
        ax_speed.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f'{t:.2f}ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax_speed.set_xticks(x_pos)
    ax_speed.set_xticklabels(labels, fontsize=8.5)
    ax_speed.set_ylabel('Avg Inference Time (ms)', fontsize=11)
    ax_speed.set_title('Inference Speed', fontsize=12)
    ax_speed.set_ylim(0, max(times) * 1.35)
    ax_speed.grid(axis='y', alpha=0.3)
    ax_speed.spines['top'].set_visible(False)
    ax_speed.spines['right'].set_visible(False)

    # ── 左下：各类别准确率折线图 ──────────────────────────────
    cls_x = np.arange(len(CLASS_NAMES))
    for m in available:
        style = MODEL_STYLES[m]
        ax_cls.plot(
            cls_x,
            results[m]['per_class_acc'],
            color=style['color'],
            marker=style['marker'],
            markersize=3,
            linewidth=1.0,
            label=style['label'],
            alpha=0.85
        )
    ax_cls.set_xticks(cls_x)
    ax_cls.set_xticklabels(CLASS_NAMES, rotation=90, fontsize=5.5)
    ax_cls.set_ylabel('Accuracy (%)', fontsize=11)
    ax_cls.set_ylim(0, 110)
    ax_cls.set_title('Per-Class Accuracy', fontsize=12)
    ax_cls.axhline(y=80, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)
    ax_cls.legend(fontsize=8, loc='lower right')
    ax_cls.grid(axis='y', alpha=0.3)
    ax_cls.spines['top'].set_visible(False)
    ax_cls.spines['right'].set_visible(False)

    # ── 右下：数值汇总表格 ────────────────────────────────────
    ax_table.axis('off')  # 表格不需要坐标轴

    # 构建表格数据
    col_labels = ['模型', '总准确率', '推理耗时', '最高类别Acc', '最低类别Acc']
    table_data = []
    for m in available:
        r          = results[m]
        pca        = r['per_class_acc']
        best_idx   = int(np.argmax(pca))
        worst_idx  = int(np.argmin(pca))
        table_data.append([
            MODEL_STYLES[m]['label'],
            f"{r['total_accuracy']:.2f}%",
            f"{r['avg_time_ms']:.2f} ms",
            f"{pca[best_idx]:.1f}% ({CLASS_NAMES[best_idx]})",
            f"{pca[worst_idx]:.1f}% ({CLASS_NAMES[worst_idx]})",
        ])

    table = ax_table.table(
        cellText    = table_data,
        colLabels   = col_labels,
        loc         = 'center',
        cellLoc     = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)   # 行高放大，更易阅读

    # 表头加背景色
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 数据行交替底色
    for i in range(1, len(table_data) + 1):
        bg = '#EBF0FA' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    ax_table.set_title('Metrics Summary', fontsize=12, pad=10)

    # ── 总标题 ────────────────────────────────────────────────
    fig.suptitle(
        'Lego Shape Classifier — Three Model Comparison',
        fontsize=16, fontweight='bold', y=1.01
    )

    save_path = os.path.join(results_dir, "model_compare_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  四图合一总览图已保存：{save_path}")


# ============================================================
# 主函数
# ============================================================
def main():

    results_dir = CONFIG["results_dir"]
    os.makedirs(results_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  乐高形状识别 — 三模型横向对比")
    print("=" * 60)

    # ── 读取测试结果 ──────────────────────────────────────────
    print(f"\n[读取] 加载测试结果...")
    results, available = load_all_results(results_dir)

    if len(available) == 0:
        print("\n❌ 没有找到任何测试结果文件，请先运行测试脚本：")
        print("   python test/test_shape.py --model v1")
        print("   python test/test_shape.py --model v2")
        print("   python test/test_shape.py --model v3")
        return

    if len(available) < 3:
        print(f"\n⚠️  只找到 {len(available)} 个模型的结果，"
              f"将仅对比已有模型：{available}")
        print("   建议三个模型都测试完毕后再运行本脚本")

    # ── 生成各对比图 ──────────────────────────────────────────
    print(f"\n[绘图] 生成对比图表...")

    plot_accuracy_bar(results, available, results_dir)
    plot_per_class_comparison(results, available, results_dir)
    plot_speed_bar(results, available, results_dir)
    plot_summary(results, available, results_dir)

    # ── 终端打印简要对比 ──────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  最终对比结果")
    print(f"{'='*60}")
    print(f"  {'模型':<25} {'总准确率':>10} {'推理耗时':>14}")
    print(f"  {'-'*50}")
    for m in available:
        r = results[m]
        print(
            f"  {MODEL_STYLES[m]['label']:<25} "
            f"{r['total_accuracy']:>9.2f}%"
            f"{r['avg_time_ms']:>12.2f} ms"
        )
    print(f"{'='*60}")
    print(f"\n✅ 全部完成！对比图已保存到：{results_dir}")


# ── 程序入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()
