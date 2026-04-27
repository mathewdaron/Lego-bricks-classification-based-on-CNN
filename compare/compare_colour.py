# ============================================================
# test/compare_colour.py
# 颜色识别三模型横向对比脚本
#
# 用法（在项目根目录下运行）：
#   python test/compare_colour.py
#
# 前提：
#   三个颜色模型都已完成测试，results/ 目录下存在：
#     colour_v1_test_result.txt
#     colour_v2_test_result.txt
#     colour_v3_test_result.txt
#   （由 python test/test_colour.py --model vX 生成）
#
# 输出（保存到 results/ 目录）：
#   colour_compare_accuracy.png     三模型总体准确率对比柱状图
#   colour_compare_per_class.png    三模型各颜色类别准确率对比折线图
#   colour_compare_speed.png        三模型推理速度对比柱状图
#   colour_compare_summary.png      四图合一总览图（最适合放README）
# ============================================================

import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')   # 非交互式后端，云端/服务器必须
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
    "num_classes" : 9,
}

# 三个颜色模型的显示名称和绘图样式（统一风格）
# label  : 图表中显示的文字
# color  : 折线/柱子颜色
# marker : 折线图的点形状
MODEL_STYLES = {
    'v1': {
        'label' : 'V1  (4层CNN,  无BN)',
        'color' : 'steelblue',
        'marker': 'o',
    },
    'v2': {
        'label' : 'V2  (6层CNN,  含BN)',
        'color' : 'tomato',
        'marker': 's',
    },
    'v3': {
        'label' : 'V3  (迷你ResNet)',
        'color' : 'mediumseagreen',
        'marker': '^',
    },
}

# 9个颜色类别的可读名称（必须与 test_colour.py 的 CLASS_NAMES 完全一致）
CLASS_NAMES = [
    "Red",     # colour_00：大红（0°）
    "Yellow",  # colour_01：明黄（60°）
    "Green",   # colour_02：绿色（120°）
    "Cyan",    # colour_03：青色（180°）
    "Blue",    # colour_04：蓝色（240°）
    "Rose",    # colour_05：玫瑰色（300°）
    "Gray",    # colour_06：灰色（去饱和）
    "White",   # colour_07：白色（手动截图）
    "Black",   # colour_08：黑色（手动截图）
]


# ============================================================
# 第一部分：读取测试结果 txt
# ============================================================
def load_result(model_name, results_dir):
    """
    读取单个颜色模型的测试结果 txt 文件

    txt 格式（由 test_colour.py 生成，与形状版完全一致）：
        model=v1
        total_accuracy=96.3333
        avg_inference_time_ms=0.5678
        per_class_acc=98.00,95.00,...（9个值，逗号分隔）

    参数：
        model_name  : 'v1' / 'v2' / 'v3'
        results_dir : results/ 目录路径

    返回：
        result : 字典，包含 total_accuracy / avg_time_ms / per_class_acc
                 若文件不存在，返回 None
    """
    # 颜色版文件名加 colour_ 前缀
    txt_path = os.path.join(
        results_dir, f"colour_{model_name}_test_result.txt"
    )

    if not os.path.exists(txt_path):
        return None

    result = {}
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 只分割第一个等号，避免值里含等号时出错
            key, value = line.split('=', 1)

            if key == 'model':
                result['model'] = value

            elif key == 'total_accuracy':
                result['total_accuracy'] = float(value)

            elif key == 'avg_inference_time_ms':
                result['avg_time_ms'] = float(value)

            elif key == 'per_class_acc':
                # 逗号分隔的9个浮点数
                result['per_class_acc'] = [
                    float(x) for x in value.split(',')
                ]

    return result


def load_all_results(results_dir):
    """
    加载三个颜色模型的测试结果

    返回：
        results   : 字典 {'v1': {...}, 'v2': {...}, 'v3': {...}}
                    某个模型文件不存在时对应值为 None
        available : 有效结果的模型名称列表，如 ['v1', 'v2', 'v3']
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

    直观体现：V1（浅层无BN）→ V2（加BN）→ V3（残差）的准确率变化

    颜色任务相对简单，三个模型准确率可能都很高且接近，
    所以 y 轴下限动态调整（从最低值下方10%开始），突出细微差异
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = np.arange(len(available))
    accs   = [results[m]['total_accuracy'] for m in available]
    colors = [MODEL_STYLES[m]['color']     for m in available]
    labels = [MODEL_STYLES[m]['label']     for m in available]

    bars = ax.bar(x_pos, accs, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.2)

    # 在柱子顶端标注具体百分比数值
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
    ax.set_title('Colour Model Comparison — Overall Accuracy', fontsize=13)

    # y轴动态下限：从最低准确率下方10%开始，放大差异便于阅读
    min_acc = min(accs)
    ax.set_ylim(max(0, min_acc - 10), 102)

    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    # 输出文件名加 colour_ 前缀
    save_path = os.path.join(results_dir, "colour_compare_accuracy.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  总体准确率对比图已保存：{save_path}")


# ============================================================
# 第三部分：图2 —— 各颜色类别准确率对比折线图
# ============================================================
def plot_per_class_comparison(results, available, results_dir):
    """
    三模型在9个颜色类别上的准确率折线对比图

    横轴：9个颜色类别（Red / Yellow / Green ...）
    纵轴：该类别的准确率
    每个模型一条折线

    颜色版与形状版的主要视觉差异：
    - 只有9个点（形状版45个），图更紧凑
    - 点更大（markersize=7），标签更大（fontsize=11）
    - 横轴标签不用旋转90°，45°倾斜就够了

    用途分析示例：
    - 如果 White/Black 三个模型都低 → 说明纯色积木本身难区分
    - 如果 V3 在 Gray 上比 V1 明显提升 → 说明残差结构更能捕捉低饱和度特征
    """
    # 颜色版图可以做得更紧凑（9个点 vs 45个点）
    fig, ax = plt.subplots(figsize=(9, 5))

    x_pos = np.arange(len(CLASS_NAMES))

    for m in available:
        acc_list = results[m]['per_class_acc']
        style    = MODEL_STYLES[m]
        ax.plot(
            x_pos, acc_list,
            color=style['color'],
            marker=style['marker'],
            markersize=7,       # 比形状版(4)更大，9个点看得清
            linewidth=1.5,
            label=style['label'],
            alpha=0.85
        )

    ax.set_xticks(x_pos)
    # 9个标签，45°倾斜就够（形状版45个要90°）
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        'Colour Model Comparison — Per-Class Accuracy', fontsize=13
    )
    ax.axhline(y=80, color='gray', linestyle='--',
               linewidth=1, alpha=0.5, label='80% threshold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "colour_compare_per_class.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  各类别准确率对比图已保存：{save_path}")


# ============================================================
# 第四部分：图3 —— 推理速度对比柱状图
# ============================================================
def plot_speed_bar(results, available, results_dir):
    """
    三模型单张图片平均推理耗时对比柱状图

    颜色模型参数量远小于形状模型（万级 vs 千万级），
    所以推理速度会快很多，时间单位仍是毫秒（ms），
    但数值会比形状版小很多，在README中体现"轻量级"特点
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos  = np.arange(len(available))
    times  = [results[m]['avg_time_ms'] for m in available]
    colors = [MODEL_STYLES[m]['color']  for m in available]
    labels = [MODEL_STYLES[m]['label']  for m in available]

    bars = ax.bar(x_pos, times, color=colors, width=0.5,
                  edgecolor='white', linewidth=1.2)

    # 在柱子顶端标注耗时数值
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
    ax.set_title('Colour Model Comparison — Inference Speed', fontsize=13)
    ax.set_ylim(0, max(times) * 1.3)
    ax.grid(axis='y', alpha=0.35)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    save_path = os.path.join(results_dir, "colour_compare_speed.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  推理速度对比图已保存：{save_path}")


# ============================================================
# 第五部分：图4 —— 四图合一总览图
# ============================================================
def plot_summary(results, available, results_dir):
    """
    把四张图合并成一张大图：
      左上：总体准确率柱状图
      右上：推理速度柱状图
      左下：各颜色类别折线图
      右下：数值汇总表格

    这张图最适合直接放进 README 展示，
    让招生官/访问者一眼看到三模型的完整对比信息

    颜色版与形状版的主要视觉差异：
    - 整体 figsize 略小（颜色类别少，折线图不需要那么宽）
    - 折线图标签字体更大、点更大
    - 表格中"最高/最低类别"显示颜色名称（而非形状名称）
    - 总标题改为 Colour Classifier
    """
    fig = plt.figure(figsize=(18, 12))

    # GridSpec 布局：2行2列
    # 上行：准确率柱状图 + 速度柱状图（各占一半）
    # 下行：类别折线图  + 数值汇总表格
    # height_ratios=[1, 1.2] → 下行比上行高20%（给折线图和表格更多空间）
    gs = gridspec.GridSpec(
        2, 2,
        figure=fig,
        hspace=0.42,
        wspace=0.30,
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

    # ── 左下：各颜色类别准确率折线图 ──────────────────────────
    # 颜色版只有9个点，标签和点都可以更大更清晰
    cls_x = np.arange(len(CLASS_NAMES))
    for m in available:
        style = MODEL_STYLES[m]
        ax_cls.plot(
            cls_x,
            results[m]['per_class_acc'],
            color=style['color'],
            marker=style['marker'],
            markersize=6,       # 比形状版(3)更大
            linewidth=1.2,
            label=style['label'],
            alpha=0.85
        )
    ax_cls.set_xticks(cls_x)
    # 颜色版标签短，45°倾斜就够（形状版45个要90°）
    ax_cls.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax_cls.set_ylabel('Accuracy (%)', fontsize=11)
    ax_cls.set_ylim(0, 110)
    ax_cls.set_title('Per-Class Accuracy', fontsize=12)
    ax_cls.axhline(y=80, color='gray', linestyle='--',
                   linewidth=0.8, alpha=0.5)
    ax_cls.legend(fontsize=9, loc='lower right')
    ax_cls.grid(axis='y', alpha=0.3)
    ax_cls.spines['top'].set_visible(False)
    ax_cls.spines['right'].set_visible(False)

    # ── 右下：数值汇总表格 ────────────────────────────────────
    ax_table.axis('off')   # 表格区域不需要坐标轴

    # 构建表格内容
    # 列：模型名 | 总准确率 | 推理耗时 | 最高准确率颜色 | 最低准确率颜色
    col_labels = ['模型', '总准确率', '推理耗时', '最高类别Acc', '最低类别Acc']
    table_data = []
    for m in available:
        r         = results[m]
        pca       = r['per_class_acc']
        best_idx  = int(np.argmax(pca))    # 准确率最高的颜色类别索引
        worst_idx = int(np.argmin(pca))    # 准确率最低的颜色类别索引
        table_data.append([
            MODEL_STYLES[m]['label'],
            f"{r['total_accuracy']:.2f}%",
            f"{r['avg_time_ms']:.2f} ms",
            # 括号里显示颜色名（如 "Green"），比形状版更直观
            f"{pca[best_idx]:.1f}% ({CLASS_NAMES[best_idx]})",
            f"{pca[worst_idx]:.1f}% ({CLASS_NAMES[worst_idx]})",
        ])

    table = ax_table.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = 'center',
        cellLoc   = 'center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)   # 行高放大，更易阅读

    # 表头：蓝色背景 + 白色加粗字体
    for j in range(len(col_labels)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # 数据行：交替白色/浅蓝色底色，提升可读性
    for i in range(1, len(table_data) + 1):
        bg = '#EBF0FA' if i % 2 == 0 else 'white'
        for j in range(len(col_labels)):
            table[i, j].set_facecolor(bg)

    ax_table.set_title('Metrics Summary', fontsize=12, pad=10)

    # ── 整张大图的总标题 ──────────────────────────────────────
    fig.suptitle(
        'Lego Colour Classifier — Three Model Comparison',
        fontsize=16, fontweight='bold', y=1.01
    )

    save_path = os.path.join(results_dir, "colour_compare_summary.png")
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
    print("  乐高颜色识别 — 三模型横向对比")
    print("=" * 60)

    # ── 读取三个颜色模型的测试结果 ────────────────────────────
    print(f"\n[读取] 加载颜色测试结果...")
    results, available = load_all_results(results_dir)

    # 没有任何结果 → 提示先跑测试脚本
    if len(available) == 0:
        print("\n❌ 没有找到任何颜色测试结果文件，请先运行测试脚本：")
        print("   python test/test_colour.py --model v1")
        print("   python test/test_colour.py --model v2")
        print("   python test/test_colour.py --model v3")
        return

    # 只有部分结果 → 警告但继续运行（对已有的模型做对比）
    if len(available) < 3:
        print(f"\n⚠️  只找到 {len(available)} 个颜色模型的结果，"
              f"将仅对比已有模型：{available}")
        print("   建议三个模型都测试完毕后再运行本脚本，对比更完整")

    # ── 生成四张对比图 ────────────────────────────────────────
    print(f"\n[绘图] 生成颜色对比图表...")

    plot_accuracy_bar(results, available, results_dir)         # 图1
    plot_per_class_comparison(results, available, results_dir) # 图2
    plot_speed_bar(results, available, results_dir)            # 图3
    plot_summary(results, available, results_dir)              # 图4（合一）

    # ── 终端打印简要对比表 ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  颜色识别最终对比结果")
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
    print(f"\n✅ 全部完成！颜色对比图已保存到：{results_dir}")


# ── 程序入口 ──────────────────────────────────────────────────
if __name__ == "__main__":
    main()
