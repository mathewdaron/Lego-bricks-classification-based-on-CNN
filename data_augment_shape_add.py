# ============================================================
# data_augment_shape_add.py
# 乐高形状识别 —— 新增刁钻视角图片的离线增强脚本
#
# 输入：raw_data_test/add/（45个子文件夹，每类1张刁钻视角原图）
# 输出：test/test_dataset_shape_add/（每张原图生成8张，共360张）
#
# 与 data_augment_shape_v2.py 的区别：
#   1. 倍率从36调整为8（背景×1, 位置×2, 翻转×1, 仿射×2, 旋转×2）
#   2. 输出路径改为 test/test_dataset_shape_add/
#   3. 输入路径改为 raw_data_test/add/
#   4. 输出直接作为测试集补充，不再需要5:1划分
#
# 增强逻辑（每张原图固定生成8张）：
#   基础框架（2种组合）：背景色8选1 × 位置5选2
#   形变乘数（×4）：水平翻转×1 → 仿射×2 → 旋转×2
#   总计：2 × 4 = 8张
#
# 运行方式（在项目根目录下）：
#   python data_augment_shape_add.py
# ============================================================

import os
import random
import math
import numpy as np
from PIL import Image

# ============================================================
# 路径配置
# ============================================================
RAW_ROOT    = r"C:\Users\lsbt\Desktop\lego_part_classifier\raw_data_test\add"
OUTPUT_ROOT = r"C:\Users\lsbt\Desktop\lego_part_classifier\test\test_dataset_shape_add"
CANVAS_SIZE = 224

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 参数配置（已按你的要求调整倍率）
# ============================================================

# ── 8种背景色（与v2一致）─────────────────────────────────────
ALL_BACKGROUNDS = [
    (255, 252, 245),   # 暖白
    (255, 248, 220),   # 浅米黄
    (255, 228, 196),   # 浅杏色
    (210, 245, 235),   # 浅薄荷绿
    (210, 235, 255),   # 浅天空蓝
    (255, 220, 215),   # 浅藕粉
    (230, 220, 255),   # 浅薰衣草紫
    (255, 250, 200),   # 浅鹅黄
]
N_BG = 1   # ★ 8选1（v2是8选2）

# ── 3种缩放档位 ──────────────────────────────────────────────
ALL_SCALES = [0.65, 0.80, 0.95]
N_SCALE = 1   # 3选1（不变）

# ── 5种位置方向 ──────────────────────────────────────────────
ALL_POSITIONS = [
    "center",
    "top_left",
    "bottom_left",
    "top_right",
    "bottom_right",
]
N_POS = 2   # ★ 5选2（v2是5选3）

POS_RATIO = 0.6

# ── 形变参数（与v2一致）──────────────────────────────────────
AFFINE_MAX_SHEAR_DEG = 5.0
AFFINE_MAX_SHIFT_RATIO = 0.03
ROTATE_MAX_DEG = 15.0

# ★ 形变倍率（与v2不同）：
#   翻转：×1（不变，随机执行或不执行）
#   仿射：×2（不变）
#   旋转：×2（v2是×3，这里改为2）
N_AFFINE = 2
N_ROTATE = 2

# 白色背景去除阈值
BG_THRESHOLD = 240


# ============================================================
# 以下所有函数与 data_augment_shape_v2.py 完全一致
# 复制自 v2 的对应函数，未做任何修改
# ============================================================

# ── 第一步：去白背景 ─────────────────────────────────────────
def remove_white_background(img, threshold=BG_THRESHOLD):
    """
    把原图的白色/浅色背景变为透明（RGBA格式）
    """
    img_array = np.array(img, dtype=np.uint8)

    # 检测三个通道都超过阈值的像素 → 背景
    white_mask = np.all(img_array >= threshold, axis=2)

    # 转为RGBA
    rgba = np.concatenate([
        img_array,
        np.full((*img_array.shape[:2], 1), 255, dtype=np.uint8)
    ], axis=2)

    # 将背景区域设为透明
    rgba[white_mask, 3] = 0

    result = Image.fromarray(rgba, mode='RGBA')
    return result


# ── 第二步：自适应缩放 + 位置 ────────────────────────────────
def place_on_transparent_canvas(fg_rgba, scale_ratio, position):
    """
    自适应缩放积木并放置到透明画布上的指定位置
    """
    W, H = CANVAS_SIZE, CANVAS_SIZE

    # 获取积木的边界框（去除透明区域）
    bbox = fg_rgba.getbbox()
    if bbox is None:
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))

    obj_w = bbox[2] - bbox[0]
    obj_h = bbox[3] - bbox[1]

    if obj_w == 0 or obj_h == 0:
        return Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # 裁剪出积木主体
    obj_img = fg_rgba.crop(bbox)

    # 自适应缩放：长边匹配画布的指定比例
    long_side = max(obj_w, obj_h)
    if long_side > 0:
        scale = (CANVAS_SIZE * scale_ratio) / long_side
    else:
        scale = CANVAS_SIZE

    new_w = max(1, int(obj_w * scale))
    new_h = max(1, int(obj_h * scale))

    obj_resized = obj_img.resize((new_w, new_h), Image.LANCZOS)

    # 计算可用偏移空间
    max_dx = W - new_w
    max_dy = H - new_h

    if max_dx <= 0 or max_dy <= 0:
        offset_x, offset_y = 0, 0
    else:
        if position == "center":
            offset_x = max_dx // 2
            offset_y = max_dy // 2
        elif position == "top_left":
            offset_x = 0
            offset_y = 0
        elif position == "bottom_left":
            offset_x = 0
            offset_y = max_dy
        elif position == "top_right":
            offset_x = max_dx
            offset_y = 0
        elif position == "bottom_right":
            offset_x = max_dx
            offset_y = max_dy

        # 应用偏移强度（向指定方向偏移可用空间的POS_RATIO）
        if position != "center":
            offset_x = int(offset_x * POS_RATIO)
            offset_y = int(offset_y * POS_RATIO)
        else:
            offset_x = random.randint(
                int(max_dx * (1 - POS_RATIO) / 2),
                int(max_dx * (1 + POS_RATIO) / 2)
            )
            offset_y = random.randint(
                int(max_dy * (1 - POS_RATIO) / 2),
                int(max_dy * (1 + POS_RATIO) / 2)
            )

    # 创建透明画布并贴上积木
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    canvas.paste(obj_resized, (offset_x, offset_y), obj_resized)

    return canvas


# ── 第三步：随机水平翻转 ─────────────────────────────────────
def random_flip(img_rgba):
    """
    随机决定是否水平翻转（50%概率）
    """
    if random.random() < 0.5:
        return img_rgba.transpose(Image.FLIP_LEFT_RIGHT)
    return img_rgba


# ── 第四步：仿射变换 ─────────────────────────────────────────
def apply_affine(img_rgba):
    """
    对透明图做轻微仿射变换（倾斜+平移）
    """
    W, H = CANVAS_SIZE, CANVAS_SIZE

    shear_x = math.tan(
        math.radians(random.uniform(-AFFINE_MAX_SHEAR_DEG,
                                     AFFINE_MAX_SHEAR_DEG))
    )
    shear_y = math.tan(
        math.radians(random.uniform(-AFFINE_MAX_SHEAR_DEG,
                                     AFFINE_MAX_SHEAR_DEG))
    )

    max_shift = int(CANVAS_SIZE * AFFINE_MAX_SHIFT_RATIO)
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)

    data = (
        1,       shear_x, -tx,
        shear_y, 1,       -ty
    )

    result = img_rgba.transform(
        (W, H),
        Image.AFFINE,
        data,
        resample=Image.BICUBIC,
        fillcolor=None
    )

    return result


# ── 第五步：小角度旋转 ──────────────────────────────────────
def apply_rotation(img_rgba):
    """
    对透明图做小角度随机旋转（±15°内）
    """
    angle = random.uniform(-ROTATE_MAX_DEG, ROTATE_MAX_DEG)

    result = img_rgba.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=None
    )

    return result


# ── 第六步：贴背景色 ─────────────────────────────────────────
def paste_on_background(fg_rgba, bg_color):
    """
    把透明积木图贴到纯色背景上，输出最终RGB图
    """
    background = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), bg_color)
    background.paste(fg_rgba, (0, 0), fg_rgba)
    return background


# ============================================================
# ★ 主流程：对单张原图生成8张增强变体（v2是36张） ★
# ============================================================
def augment_one_image(img_path, output_dir, base_name):
    """
    对一张刁钻视角原图生成8张增强变体并保存

    8张的生成逻辑：
    ┌─────────────────────────────────────────────────────────┐
    │ 步骤0：去白背景 → RGBA透明积木图                        │
    │                                                         │
    │ 步骤1：确定基础框架参数（共2种组合）：                   │
    │   背景色 1种（8选1）                                    │
    │   缩放   1种（3选1）                                    │
    │   位置   2种（5选2）                                    │
    │   组合：1 × 1 × 2 = 2种基础配置                        │
    │                                                         │
    │ 对每种基础配置（2次循环）：                              │
    │   步骤2：缩放+位置 → 透明画布积木图（1张）              │
    │   步骤3：随机翻转（执行或不执行）→ 1张                  │
    │   步骤4：仿射变换 × 2 → 2张仿射变体                    │
    │   步骤5：每张仿射变体 × 旋转2次 → 4张旋转变体          │
    │   步骤6：贴对应背景色 → 4张RGB最终图                   │
    │                                                         │
    │ 总计：2个配置 × 4张 = 8张                              │
    └─────────────────────────────────────────────────────────┘
    """
    # ── 读取原图 ──────────────────────────────────────────────
    original = Image.open(img_path).convert("RGB")

    # ── 步骤0：去除白色背景，转为RGBA透明图 ───────────────────
    fg_rgba_raw = remove_white_background(original, BG_THRESHOLD)

    # ── 步骤1：随机抽取基础框架参数 ───────────────────────────
    chosen_bgs = random.sample(ALL_BACKGROUNDS, N_BG)       # 8选1
    chosen_scale = random.choice(ALL_SCALES)                 # 3选1
    chosen_positions = random.sample(ALL_POSITIONS, N_POS)   # 5选2

    variant_idx = 0

    # ── 外层循环：遍历2种基础配置（背景×位置）─────────────────
    for bg_color in chosen_bgs:           # 1种背景
        for position in chosen_positions:  # 2种位置

            # ── 步骤2：缩放+位置 → 透明画布积木图 ─────────────
            placed = place_on_transparent_canvas(
                fg_rgba_raw, chosen_scale, position
            )

            # ── 步骤3：随机水平翻转（×1）──────────────────────
            flipped = random_flip(placed)

            # ── 步骤4：仿射变换 × 2 ────────────────────────────
            affine_variants = []
            for _ in range(N_AFFINE):   # 2次
                affine_img = apply_affine(flipped)
                affine_variants.append(affine_img)

            # ── 步骤5：旋转 × 2（对每个仿射变体）──────────────
            for affine_img in affine_variants:   # 2张仿射图
                for _ in range(N_ROTATE):        # 各旋转2次
                    rotated = apply_rotation(affine_img)

                    # ── 步骤6：贴背景色 ──────────────────────────
                    final_img = paste_on_background(rotated, bg_color)

                    # 保存
                    out_name = f"{base_name}_v{variant_idx:03d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    final_img.save(out_path)

                    variant_idx += 1

    # 每张原图应该恰好生成8张
    assert variant_idx == 8, \
        f"生成数量异常：{variant_idx}张（期望8张）"


# ============================================================
# 总控函数：遍历所有类别和所有原图
# ============================================================
def generate_all():
    """
    遍历 raw_data_test/add/ 下所有类别，对每张原图调用增强函数
    """
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # 扫描所有类别文件夹
    class_dirs = sorted([
        d for d in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, d))
    ])

    print(f"找到 {len(class_dirs)} 个类别文件夹")
    print(f"输入目录：{RAW_ROOT}")
    print(f"输出目录：{OUTPUT_ROOT}")
    print(f"每张原图生成：8 张变体")
    print()

    total_original = 0
    total_generated = 0

    for class_name in class_dirs:
        class_input_dir = os.path.join(RAW_ROOT, class_name)
        class_output_dir = os.path.join(OUTPUT_ROOT, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # 收集该类所有原图
        image_files = sorted([
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(image_files) == 0:
            print(f"  ⚠️  {class_name}：无图片，跳过")
            continue

        print(f"  {class_name}：{len(image_files)} 张原图 → ", end="")

        for img_file in image_files:
            img_path = os.path.join(class_input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            augment_one_image(img_path, class_output_dir, base_name)

            total_original += 1
            total_generated += 8

        # 确认生成了正确数量的文件
        generated_files = [
            f for f in os.listdir(class_output_dir)
            if f.lower().endswith('.png')
        ]
        expected = len(image_files) * 8
        print(f"{len(generated_files)} 张（期望 {expected} 张）")

    print()
    print("=" * 50)
    print(f"全部完成！")
    print(f"  处理原图：     {total_original} 张")
    print(f"  生成增强图：   {total_generated} 张")
    print(f"  输出到：       {OUTPUT_ROOT}")
    print("=" * 50)


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    generate_all()
