# ============================================================
# visualize_models.py
# 模型结构图渲染脚本（Matplotlib 模块化分层框图）
#
# 用法（在项目根目录下运行）：
#   python visualize_models.py
#
# 无需训练、无需数据集，直接运行即可生成6张结构图
#
# 输出（保存到 results/model_diagrams/ 目录）：
#   shape_v1_architecture.png   形状V1：4块浅层CNN，无BN
#   shape_v2_architecture.png   形状V2：5块深层CNN，含BN
#   shape_v3_architecture.png   形状V3：迷你ResNet
#   colour_v1_architecture.png  颜色V1：2块浅层CNN，无BN
#   colour_v2_architecture.png  颜色V2：3块深层CNN，含BN
#   colour_v3_architecture.png  颜色V3：迷你ResNet
# ============================================================

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

# ── 输出目录 ──────────────────────────────────────────────────
OUTPUT_DIR = os.path.join("results", "model_diagrams")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 核心绘图引擎
# ============================================================

# 每种模块类型对应的颜色，统一6张图风格
MODULE_COLORS = {
    "input"      : "#AED6F1",   # 浅蓝：输入
    "conv_block" : "#A9DFBF",   # 浅绿：卷积块（无BN）
    "bn_block"   : "#A9DFBF",   # 浅绿：卷积块（含BN，加边框区分）
    "res_block"  : "#F9E79F",   # 浅黄：残差块
    "stem"       : "#D5F5E3",   # 薄荷绿：Stem层
    "pool"       : "#D2B4DE",   # 浅紫：全局池化
    "fc"         : "#FAD7A0",   # 浅橙：全连接层
    "output"     : "#F1948A",   # 浅红：输出
}


def draw_module(ax, x, y, width, height, label, sublabel,
                color, has_bn=False, is_res=False):
    """
    在坐标 (x, y) 处画一个模块方块

    参数：
        ax       : matplotlib 坐标轴
        x, y     : 方块左下角坐标
        width    : 方块宽度
        height   : 方块高度
        label    : 主标题（加粗大字，如 "Conv Block 1"）
        sublabel : 副标题（小字，如 "3→64, k3 ×2 + Pool"）
        color    : 填充颜色
        has_bn   : 是否含BN（True则画双线边框）
        is_res   : 是否是残差块（True则画虚线边框）
    """
    # 边框样式
    if is_res:
        # 残差块：黄色虚线边框，视觉上和普通块区分
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            linewidth=2, edgecolor="#E67E22",
            linestyle='--',
            facecolor=color, zorder=2
        )
    elif has_bn:
        # 含BN：深绿色双线边框
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            linewidth=2.5, edgecolor="#1E8449",
            facecolor=color, zorder=2
        )
    else:
        # 普通块：灰色细线边框
        rect = mpatches.FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.02",
            linewidth=1.5, edgecolor="#5D6D7E",
            facecolor=color, zorder=2
        )
    ax.add_patch(rect)

    # 主标签（加粗）
    cx = x + width  / 2
    cy = y + height / 2

    if sublabel:
        # 有副标签时，主标签靠上，副标签靠下
        ax.text(cx, cy + height * 0.12, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)
        ax.text(cx, cy - height * 0.15, sublabel,
                ha='center', va='center',
                fontsize=7.5, color='#2C3E50', zorder=3)
    else:
        ax.text(cx, cy, label,
                ha='center', va='center',
                fontsize=9, fontweight='bold', zorder=3)


def draw_arrow(ax, x_start, y_mid, x_end):
    """
    在两个模块之间画一条水平连接箭头

    参数：
        x_start : 起点 x（上一个模块右边缘）
        y_mid   : 箭头的 y 坐标（模块中线高度）
        x_end   : 终点 x（下一个模块左边缘）
    """
    ax.annotate(
        '', 
        xy=(x_end, y_mid),
        xytext=(x_start, y_mid),
        arrowprops=dict(
            arrowstyle='->', 
            color='#5D6D7E',
            lw=1.5
        ),
        zorder=1
    )


def render_architecture(modules, title, save_name,
                        fig_width=None, fig_height=3.8):
    """
    根据模块列表渲染一张完整的网络结构图

    参数：
        modules    : 模块定义列表，每个元素是一个字典：
                     {
                       'label'   : str,   主标题
                       'sublabel': str,   副标题（可为空字符串）
                       'color'   : str,   填充颜色（从 MODULE_COLORS 取）
                       'has_bn'  : bool,  是否含BN
                       'is_res'  : bool,  是否残差块
                       'width'   : float  模块宽度（相对单位，默认1.2）
                     }
        title      : 图表大标题
        save_name  : 保存文件名（不含路径，含扩展名）
        fig_width  : 图宽（None则自动计算）
        fig_height : 图高
    """
    # ── 布局参数 ──────────────────────────────────────────────
    MODULE_H    = 1.4    # 模块高度（固定）
    GAP         = 0.35   # 模块之间的间距（用于画箭头）
    Y_CENTER    = 1.0    # 所有模块的底边 y 坐标（水平一排）

    # 计算每个模块的 x 起始坐标
    x_positions = []
    x_cursor    = 0.2   # 起始留白
    for m in modules:
        x_positions.append(x_cursor)
        x_cursor += m.get('width', 1.2) + GAP

    total_width = x_cursor + 0.2   # 右侧留白

    if fig_width is None:
        fig_width = max(total_width * 1.05, 10)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlim(0, total_width)
    ax.set_ylim(0, Y_CENTER + MODULE_H + 1.0)
    ax.axis('off')   # 不显示坐标轴

    # ── 画所有模块 ────────────────────────────────────────────
    for i, (m, x) in enumerate(zip(modules, x_positions)):
        w = m.get('width', 1.2)
        draw_module(
            ax, x, Y_CENTER, w, MODULE_H,
            label    = m['label'],
            sublabel = m.get('sublabel', ''),
            color    = m['color'],
            has_bn   = m.get('has_bn', False),
            is_res   = m.get('is_res', False),
        )

    # ── 画模块间的箭头 ────────────────────────────────────────
    y_mid = Y_CENTER + MODULE_H / 2
    for i in range(len(modules) - 1):
        x_start = x_positions[i]     + modules[i].get('width', 1.2)
        x_end   = x_positions[i + 1]
        draw_arrow(ax, x_start, y_mid, x_end)

    # ── 图例（说明边框含义）──────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor='#A9DFBF', edgecolor='#5D6D7E',
                       linewidth=1.5, label='Conv Block (no BN)'),
        mpatches.Patch(facecolor='#A9DFBF', edgecolor='#1E8449',
                       linewidth=2.5, label='Conv Block (with BN)'),
        mpatches.Patch(facecolor='#F9E79F', edgecolor='#E67E22',
                       linewidth=2, linestyle='--', label='Residual Block'),
    ]
    ax.legend(
        handles=legend_elements, loc='upper right',
        fontsize=8, framealpha=0.85,
        bbox_to_anchor=(0.99, 0.99)
    )

    # ── 标题 ──────────────────────────────────────────────────
    ax.set_title(title, fontsize=13, fontweight='bold', pad=12)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ 已保存：{save_path}")


# ============================================================
# 6个模型的模块定义
# ============================================================

def get_shape_v1_modules():
    """
    形状 V1：4块浅层CNN，无BN
    结构：Input → Block1(3→64) → Block2(64→128) →
          Block3(128→256) → Block4(256→512) →
          GlobalAvgPool → FC(512→256) → FC(256→45) → Output
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'Conv Block 1',
            'sublabel': '3→64, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'Conv Block 2',
            'sublabel': '64→128, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'Conv Block 3',
            'sublabel': '128→256, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'Conv Block 4',
            'sublabel': '256→512, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '512×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'FC Layer 1',
            'sublabel': '512→256\n+ Dropout(0.5)',
            'color'   : MODULE_COLORS['fc'],
            'width'   : 1.35,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 256→45\n(45 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


def get_shape_v2_modules():
    """
    形状 V2：5块深层CNN，含BN
    结构：Input → Block1~5（每块3层Conv+BN+ReLU+MaxPool）→
          GlobalAvgPool → FC×2 → Output
    通道：64→128→256→512→512
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'BN Block 1',
            'sublabel': '3→64, k3×3\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 2',
            'sublabel': '64→128, k3×3\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 3',
            'sublabel': '128→256, k3×3\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 4',
            'sublabel': '256→512, k3×3\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 5',
            'sublabel': '512→512, k3×3\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '512×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'FC Layer 1',
            'sublabel': '512→256\n+ Dropout(0.5)',
            'color'   : MODULE_COLORS['fc'],
            'width'   : 1.35,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 256→45\n(45 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


def get_shape_v3_modules():
    """
    形状 V3：迷你ResNet（17层）
    结构：Input → Stem(Conv+BN+ReLU+MaxPool) →
          Layer1(ResBlock×2, 64→64) →
          Layer2(ResBlock×2, 64→128) →
          Layer3(ResBlock×2, 128→256) →
          Layer4(ResBlock×2, 256→512) →
          GlobalAvgPool → FC(512→45) → Output
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'Stem',
            'sublabel': '3→64, k7\nBN+Pool',
            'color'   : MODULE_COLORS['stem'],
            'has_bn'  : True,
            'width'   : 1.2,
        },
        {
            'label'   : 'Layer 1',
            'sublabel': '64→64\nResBlock×2',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'Layer 2',
            'sublabel': '64→128\nResBlock×2',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'Layer 3',
            'sublabel': '128→256\nResBlock×2',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'Layer 4',
            'sublabel': '256→512\nResBlock×2',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '512×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 512→45\n(45 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


def get_colour_v1_modules():
    """
    颜色 V1：2块浅层CNN，无BN
    结构：Input → Block1(3→32) → Block2(32→64) →
          GlobalAvgPool → FC(64→32) → FC(32→9) → Output
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'Conv Block 1',
            'sublabel': '3→32, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'Conv Block 2',
            'sublabel': '32→64, k3×2\n+ MaxPool',
            'color'   : MODULE_COLORS['conv_block'],
            'has_bn'  : False,
            'width'   : 1.4,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '64×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'FC Layer 1',
            'sublabel': '64→32\n+ Dropout(0.3)',
            'color'   : MODULE_COLORS['fc'],
            'width'   : 1.35,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 32→9\n(9 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


def get_colour_v2_modules():
    """
    颜色 V2：3块深层CNN，含BN
    结构：Input → Block1(3→32) → Block2(32→64) →
          Block3(64→128) → GlobalAvgPool →
          FC(128→64) → FC(64→9) → Output
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'BN Block 1',
            'sublabel': '3→32, k3×2\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 2',
            'sublabel': '32→64, k3×2\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'BN Block 3',
            'sublabel': '64→128, k3×2\nBN + MaxPool',
            'color'   : MODULE_COLORS['bn_block'],
            'has_bn'  : True,
            'width'   : 1.45,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '128×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'FC Layer 1',
            'sublabel': '128→64\n+ Dropout(0.3)',
            'color'   : MODULE_COLORS['fc'],
            'width'   : 1.35,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 64→9\n(9 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


def get_colour_v3_modules():
    """
    颜色 V3：迷你ResNet（9层）
    结构：Input → Stem(3→32, k3+BN+Pool) →
          Layer1(ResBlock×1, 32→32) →
          Layer2(ResBlock×1, 32→64) →
          Layer3(ResBlock×1, 64→128) →
          GlobalAvgPool → FC(128→9) → Output
    """
    return [
        {
            'label'   : 'Input',
            'sublabel': '3×224×224',
            'color'   : MODULE_COLORS['input'],
            'width'   : 1.1,
        },
        {
            'label'   : 'Stem',
            'sublabel': '3→32, k3\nBN+Pool',
            'color'   : MODULE_COLORS['stem'],
            'has_bn'  : True,
            'width'   : 1.2,
        },
        {
            'label'   : 'Layer 1',
            'sublabel': '32→32\nResBlock×1',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'Layer 2',
            'sublabel': '32→64\nResBlock×1',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'Layer 3',
            'sublabel': '64→128\nResBlock×1',
            'color'   : MODULE_COLORS['res_block'],
            'is_res'  : True,
            'width'   : 1.3,
        },
        {
            'label'   : 'GlobalAvgPool',
            'sublabel': '128×1×1',
            'color'   : MODULE_COLORS['pool'],
            'width'   : 1.3,
        },
        {
            'label'   : 'Output',
            'sublabel': 'FC 128→9\n(9 classes)',
            'color'   : MODULE_COLORS['output'],
            'width'   : 1.2,
        },
    ]


# ============================================================
# 主函数：渲染全部6张图
# ============================================================
def main():
    print("\n" + "=" * 55)
    print("  乐高积木分类器 — 模型结构图渲染")
    print("=" * 55)
    print(f"  输出目录：{OUTPUT_DIR}\n")

    # ── 形状模型三张 ──────────────────────────────────────────
    print("[形状模型]")

    render_architecture(
        modules   = get_shape_v1_modules(),
        title     = "Shape Model V1 — 8-Layer CNN (No BN) | 45 Classes",
        save_name = "shape_v1_architecture.png",
        fig_width = 16,
    )

    render_architecture(
        modules   = get_shape_v2_modules(),
        title     = "Shape Model V2 — 12-Layer CNN (With BN) | 45 Classes",
        save_name = "shape_v2_architecture.png",
        fig_width = 19,
    )

    render_architecture(
        modules   = get_shape_v3_modules(),
        title     = "Shape Model V3 — Mini ResNet (17-Layer) | 45 Classes",
        save_name = "shape_v3_architecture.png",
        fig_width = 16,
    )

    # ── 颜色模型三张 ──────────────────────────────────────────
    print("\n[颜色模型]")

    render_architecture(
        modules   = get_colour_v1_modules(),
        title     = "Colour Model V1 — 4-Layer CNN (No BN) | 9 Classes",
        save_name = "colour_v1_architecture.png",
        fig_width = 12,
    )

    render_architecture(
        modules   = get_colour_v2_modules(),
        title     = "Colour Model V2 — 6-Layer CNN (With BN) | 9 Classes",
        save_name = "colour_v2_architecture.png",
        fig_width = 14,
    )

    render_architecture(
        modules   = get_colour_v3_modules(),
        title     = "Colour Model V3 — Mini ResNet (9-Layer) | 9 Classes",
        save_name = "colour_v3_architecture.png",
        fig_width = 14,
    )

    print("\n" + "=" * 55)
    print("  ✅ 全部6张结构图生成完毕！")
    print(f"  保存位置：{os.path.abspath(OUTPUT_DIR)}")
    print("=" * 55)


if __name__ == "__main__":
    main()
