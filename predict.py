# ============================================================
# predict.py
# 乐高积木 —— 形状 × 颜色 联合预测演示脚本
#
# 用法（在项目根目录下运行）：
#   python predict.py
#
# 前提：
#   1. 形状最优模型权重存在（默认用 V3）：
#      checkpoints/model_v3_best.pth
#   2. 颜色最优模型权重存在（默认用 V3）：
#      checkpoints/colour_v3_best.pth
#   3. 待预测图片放入：input_images/ 文件夹
#      （支持 .png / .jpg / .jpeg）
#
# 输出（保存到 results_predict/ 目录）：
#   每张输入图片生成一张对应的带标注结果图
#   文件名：predict_原文件名.png
#   图上标注：
#     Shape : Brick 2x4  · #3001  (98.3%)
#     Color : ██ Blue              (96.1%)
# ============================================================

import os
import sys
import csv                          # ← 修改：用于读取 lego_mapping.csv

import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── 把项目根目录加入Python路径 ────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)


# ============================================================
# 配置（按需修改）
# ============================================================
CONFIG = {
    # 输入图片文件夹（把待预测图片放进来）
    "input_dir"           : os.path.join(ROOT_DIR, "input_images"),

    # 输出结果文件夹
    "output_dir"          : os.path.join(ROOT_DIR, "results_predict"),

    # 模型权重路径（默认用各任务最优的 V3）
    "shape_checkpoint"    : os.path.join(ROOT_DIR, "checkpoints", "model_v3_best.pth"),
    "colour_checkpoint"   : os.path.join(ROOT_DIR, "checkpoints", "colour_v3_best.pth"),

    # 如需换 V1/V2，改这里以及上方路径中的 v3
    "shape_model_version" : "v3",
    "colour_model_version": "v3",

    "num_shape_classes"   : 45,
    "num_colour_classes"  : 9,

    # lego_mapping.csv 路径                             # ← 修改
    "csv_path"            : os.path.join(ROOT_DIR, "lego_mapping.csv"),
}


# ============================================================
# 读取 lego_mapping.csv，构建两个并行列表               # ← 修改（新增整段）
# ============================================================
def load_lego_mapping(csv_path):
    """
    读取 lego_mapping.csv，按 shape_id 顺序（shape_00~shape_44）
    返回两个列表，索引与模型输出的类别索引完全对应：
        shape_names : ['Brick 1x1', 'Brick 1x2', ...]  45个形状全名
        lego_ids    : ['3005', '3004', ...]             45个乐高ID

    CSV 格式：
        shape_id, shape_name, lego_id, category
        shape_00, Brick 1x1,  3005,    Standard Brick
        ...

    为什么用 sorted() 按 shape_id 排序？
    因为 CSV 文件中行顺序理论上和文件夹顺序一致，
    但显式排序能保证万无一失，不依赖文件行序。
    """
    rows = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # shape_id 格式是 "shape_00"，取末两位数字转整数作为键
            idx = int(row['shape_id'].split('_')[1])
            rows[idx] = {
                'shape_name': row['shape_name'].strip(),
                'lego_id'   : row['lego_id'].strip(),
            }

    # 按索引 0→44 顺序组成列表
    shape_names = [rows[i]['shape_name'] for i in range(len(rows))]
    lego_ids    = [rows[i]['lego_id']    for i in range(len(rows))]

    return shape_names, lego_ids


# ── 在模块加载时就读取 CSV，全局共享 ─────────────────────────
# 这样推理函数直接用，不用每张图片都重复读文件
SHAPE_NAMES, LEGO_IDS = load_lego_mapping(CONFIG["csv_path"])   # ← 修改

# ── 9个颜色类别名（colour_00~colour_08 对应顺序）────────────
COLOUR_NAMES = [
    "Red",     # colour_00
    "Yellow",  # colour_01
    "Green",   # colour_02
    "Cyan",    # colour_03
    "Blue",    # colour_04
    "Rose",    # colour_05
    "Gray",    # colour_06
    "White",   # colour_07
    "Black",   # colour_08
]

# ── 预处理变换（必须和训练时完全一致）───────────────────────
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


# ============================================================
# 第一部分：加载模型
# ============================================================
def load_shape_model(version, checkpoint_path, num_classes, device):
    """
    加载形状识别模型

    参数：
        version         : 'v1' / 'v2' / 'v3'
        checkpoint_path : 权重文件路径
        num_classes     : 45
        device          : cpu / cuda
    """
    if version == 'v1':
        from models_shape.model_v1 import create_model_v1
        model = create_model_v1(num_classes=num_classes)
    elif version == 'v2':
        from models_shape.model_v2 import create_model_v2
        model = create_model_v2(num_classes=num_classes)
    elif version == 'v3':
        from models_shape.model_v3 import create_model_v3
        model = create_model_v3(num_classes=num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print(f"  ✅ 形状模型 {version.upper()} 加载成功")
    return model


def load_colour_model(version, checkpoint_path, num_classes, device):
    """
    加载颜色识别模型

    参数：
        version         : 'v1' / 'v2' / 'v3'
        checkpoint_path : 权重文件路径
        num_classes     : 9
        device          : cpu / cuda
    """
    if version == 'v1':
        from models_colour.model_v1_colour import create_model_v1_colour
        model = create_model_v1_colour(num_classes=num_classes)
    elif version == 'v2':
        from models_colour.model_v2_colour import create_model_v2_colour
        model = create_model_v2_colour(num_classes=num_classes)
    elif version == 'v3':
        from models_colour.model_v3_colour import create_model_v3_colour
        model = create_model_v3_colour(num_classes=num_classes)

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model = model.to(device)
    print(f"  ✅ 颜色模型 {version.upper()} 加载成功")
    return model


# ============================================================
# 第二部分：单张图片推理
# ============================================================
def predict_single(img_path, shape_model, colour_model, device):
    """
    对单张图片同时做形状和颜色预测

    返回：                                                # ← 修改
        shape_name   : 形状全名，如 "Brick 2x4"（来自CSV）
        lego_id      : 乐高ID，如 "3001"（来自CSV）
        shape_conf   : 形状预测置信度（0~100 百分比）
        colour_name  : 预测颜色名称，如 "Blue"
        colour_conf  : 颜色预测置信度（0~100 百分比）
    """
    img_pil    = Image.open(img_path).convert("RGB")
    img_tensor = TRANSFORM(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        # ── 形状预测 ──────────────────────────────────────────
        shape_logits = shape_model(img_tensor)
        shape_probs  = torch.softmax(shape_logits, dim=1)
        shape_conf_t, shape_idx_t = torch.max(shape_probs, dim=1)
        shape_idx  = shape_idx_t.item()
        shape_conf = shape_conf_t.item() * 100.0

        # ── 颜色预测 ──────────────────────────────────────────
        colour_logits = colour_model(img_tensor)
        colour_probs  = torch.softmax(colour_logits, dim=1)
        colour_conf_t, colour_idx_t = torch.max(colour_probs, dim=1)
        colour_idx  = colour_idx_t.item()
        colour_conf = colour_conf_t.item() * 100.0

    shape_name  = SHAPE_NAMES[shape_idx]     # ← 修改：来自CSV的全名
    lego_id     = LEGO_IDS[shape_idx]        # ← 修改：新增乐高ID
    colour_name = COLOUR_NAMES[colour_idx]

    return shape_name, lego_id, shape_conf, colour_name, colour_conf


# ============================================================
# 第三部分：生成单张可视化结果图
# ============================================================

# 每种颜色对应的色块十六进制颜色码
COLOUR_SWATCHES = {
    "Red"   : "#E74C3C",
    "Yellow": "#F1C40F",
    "Green" : "#27AE60",
    "Cyan"  : "#1ABC9C",
    "Blue"  : "#2980B9",
    "Rose"  : "#E91E8C",
    "Gray"  : "#95A5A6",
    "White" : "#F0F0F0",
    "Black" : "#2C3E50",
}


def save_result_image(img_path, shape_name, lego_id, shape_conf,  # ← 修改：加 lego_id
                      colour_name, colour_conf, output_dir):
    """
    生成一张带标注的预测结果图并保存

    图片布局：
    ┌──────────────────────────────────────┐
    │                                      │
    │           [ 积木原图 ]                │
    │                                      │
    ├──────────────────────────────────────┤
    │  Shape : Brick 2x4  · #3001  (98.3%)│  ← 形状名 + 乐高ID
    │  Color : ██ Blue           (96.1%)   │  ← 颜色小色块 + 颜色名
    └──────────────────────────────────────┘
    """
    img_pil = Image.open(img_path).convert("RGB")
    img_pil = img_pil.resize((320, 320))
    img_np  = np.array(img_pil)

    fig = plt.figure(figsize=(3.8, 4.8))
    gs  = plt.GridSpec(2, 1, figure=fig,
                       height_ratios=[4, 1.6], hspace=0.05)
    ax_img  = fig.add_subplot(gs[0])
    ax_text = fig.add_subplot(gs[1])

    # ── 上方：积木图片 ────────────────────────────────────────
    ax_img.imshow(img_np)
    ax_img.axis('off')

    # ── 下方：标注文字区 ──────────────────────────────────────
    ax_text.axis('off')
    ax_text.set_facecolor('#F8F9FA')

    # 分隔线
    ax_text.axhline(y=0.96, color='#BDC3C7', linewidth=0.8)

    # 形状行：名称 · #ID（置信度）                         # ← 修改
    # 例：Shape : Brick 2x4  · #3001  (98.3%)
    shape_line = f"Shape : {shape_name}  · #{lego_id}"
    ax_text.text(
        0.05, 0.72,
        shape_line,
        transform=ax_text.transAxes,
        fontsize=9, fontfamily='monospace',
        va='center', color='#2C3E50', fontweight='bold'
    )
    # 置信度写在右侧，颜色稍浅
    ax_text.text(
        0.95, 0.72,
        f"({shape_conf:.1f}%)",
        transform=ax_text.transAxes,
        fontsize=9, fontfamily='monospace',
        va='center', ha='right', color='#7F8C8D'
    )

    # 颜色行：小色块 + 颜色名 + 置信度
    swatch_color = COLOUR_SWATCHES.get(colour_name, "#AAAAAA")

    # 颜色小方块
    rect = mpatches.FancyBboxPatch(
        (0.05, 0.12), 0.07, 0.34,
        transform=ax_text.transAxes,
        boxstyle="square,pad=0",
        facecolor=swatch_color,
        edgecolor='#999999', linewidth=0.8,
        zorder=3
    )
    ax_text.add_patch(rect)

    # 颜色名称
    ax_text.text(
        0.145, 0.30,
        f"Color : {colour_name}",
        transform=ax_text.transAxes,
        fontsize=9, fontfamily='monospace',
        va='center', color='#2C3E50', fontweight='bold'
    )
    # 颜色置信度（右对齐）
    ax_text.text(
        0.95, 0.30,
        f"({colour_conf:.1f}%)",
        transform=ax_text.transAxes,
        fontsize=9, fontfamily='monospace',
        va='center', ha='right', color='#7F8C8D'
    )

    # ── 保存 ──────────────────────────────────────────────────
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path = os.path.join(output_dir, f"predict_{base_name}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='white')
    plt.close()
    return save_path


# ============================================================
# 第四部分：主函数
# ============================================================
def main():

    print("\n" + "=" * 60)
    print("  乐高积木 — 形状 × 颜色 联合预测演示")
    print("=" * 60)

    input_dir  = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # ── 检查 input_images 文件夹 ──────────────────────────────
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
        print(f"\n⚠️  已自动创建 input_images/ 文件夹")
        print(f"   请把待预测的积木图片放入：{input_dir}")
        return

    # ── 扫描图片 ──────────────────────────────────────────────
    valid_exts = ('.png', '.jpg', '.jpeg')
    img_files  = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(valid_exts)
    ])

    if len(img_files) == 0:
        print(f"\n⚠️  input_images/ 中没有图片，请放入后重新运行")
        return

    print(f"\n[输入] 发现 {len(img_files)} 张待预测图片")

    # ── 检查模型权重 ──────────────────────────────────────────
    shape_ckpt  = CONFIG["shape_checkpoint"]
    colour_ckpt = CONFIG["colour_checkpoint"]

    if not os.path.exists(shape_ckpt):
        print(f"\n❌ 形状模型权重不存在：{shape_ckpt}")
        return
    if not os.path.exists(colour_ckpt):
        print(f"\n❌ 颜色模型权重不存在：{colour_ckpt}")
        return

    # ── 设备检测 ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[设备] 使用：{device}")

    # ── 加载双模型 ────────────────────────────────────────────
    print(f"\n[模型] 加载形状与颜色双模型...")
    shape_model  = load_shape_model(
        CONFIG["shape_model_version"], shape_ckpt,
        CONFIG["num_shape_classes"],   device
    )
    colour_model = load_colour_model(
        CONFIG["colour_model_version"], colour_ckpt,
        CONFIG["num_colour_classes"],   device
    )

    # ── 批量推理 ──────────────────────────────────────────────
    print(f"\n[推理] 开始批量预测...\n")
    # 终端表头：加上乐高ID列                               # ← 修改
    print(f"  {'文件名':<22} {'形状名称':<20} {'乐高ID':>7}  "
          f"{'置信度':>7}   {'颜色':>8}  {'置信度':>7}")
    print(f"  {'-'*75}")

    for fname in img_files:
        img_path = os.path.join(input_dir, fname)

        shape_name, lego_id, shape_conf, colour_name, colour_conf = \
            predict_single(img_path, shape_model, colour_model, device)

        save_result_image(
            img_path,
            shape_name, lego_id, shape_conf,    # ← 修改：传入 lego_id
            colour_name, colour_conf,
            output_dir
        )

        # 终端打印：加乐高ID列                            # ← 修改
        print(f"  {fname:<22} {shape_name:<20} #{lego_id:<6}  "
              f"{shape_conf:>6.1f}%   {colour_name:>8}  {colour_conf:>6.1f}%")

    print(f"\n{'='*60}")
    print(f"  ✅ 全部完成！共处理 {len(img_files)} 张图片")
    print(f"  结果保存至：{output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
