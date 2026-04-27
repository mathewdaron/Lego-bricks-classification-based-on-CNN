# ============================================================
# utils_colour/data_augment_colour_layer1.py
#
# 功能：颜色识别数据集第一层生成脚本
#
# 【阶段一】原始颜色图生成
#   输入：raw_data_shape/（139张大红色PNG）
#   处理：色相偏移×6（彩色）+ 去饱和（灰色）
#   输出：raw_data_colour/（共7个子文件夹）
#         colour_00~colour_05：6种彩色
#         colour_06：灰色
#
# 【阶段二】第一层增强（贴背景）
#   输入：raw_data_colour/
#   处理：贴8种固定背景色，零件居中，固定缩放0.75
#   输出：layer_one_colour/（共7个子文件夹）
#
# 白色和黑色原始图由手动截图获得，后续单独写脚本处理
#
# 运行方式（项目根目录下）：
#   python utils_colour/data_augment_colour_layer1.py
# ============================================================

import os
import sys
import numpy as np
from PIL import Image, ImageEnhance

# ── 把项目根目录加入路径 ──────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ============================================================
# 全局配置
# ============================================================

# ── 路径配置 ──────────────────────────────────────────────────
RAW_SHAPE_DIR  = os.path.join(ROOT_DIR, "raw_data_shape")
RAW_COLOUR_DIR = os.path.join(ROOT_DIR, "raw_data_colour")
LAYER_ONE_DIR  = os.path.join(ROOT_DIR, "layer_one_colour")

# ── 画布配置 ──────────────────────────────────────────────────
CANVAS_SIZE  = 224   # 输出图片尺寸（正方形）
OBJECT_SCALE = 0.75  # 零件占画布比例（固定居中）

# ── 7种颜色配置 ───────────────────────────────────────────────
# 格式：(文件夹名, 颜色类型, 参数)
# 颜色类型 'hue'  → 色相偏移，参数为偏移角度
# 颜色类型 'grey' → 去饱和，参数无意义（填0即可）
COLOUR_CONFIGS = [
    ("colour_00", "hue",  0),    # 大红（原色）
    ("colour_01", "hue",  60),   # 明黄
    ("colour_02", "hue",  120),  # 绿色
    ("colour_03", "hue",  180),  # 青色
    ("colour_04", "hue",  240),  # 蓝色
    ("colour_05", "hue",  300),  # 玫瑰色
    ("colour_06", "grey", 0),    # 灰色
    # colour_07（白色）：手动截图，后续单独处理
    # colour_08（黑色）：手动截图，后续单独处理
]

# ── 8种背景色（RGB） ──────────────────────────────────────────
BACKGROUND_COLOURS = [
    (255, 252, 245),  # bg_00 暖白
    (255, 248, 220),  # bg_01 浅米黄
    (255, 228, 196),  # bg_02 浅杏色
    (210, 245, 235),  # bg_03 浅薄荷绿
    (210, 235, 255),  # bg_04 浅天空蓝
    (255, 220, 215),  # bg_05 浅藕粉
    (230, 220, 255),  # bg_06 浅薰衣草紫
    (255, 250, 200),  # bg_07 浅鹅黄
]

# ── 去背景阈值 ────────────────────────────────────────────────
WHITE_THRESHOLD = 240


# ============================================================
# 工具函数
# ============================================================

def remove_white_background(img_rgb, threshold=WHITE_THRESHOLD):
    """
    去除白色背景，返回带透明通道的RGBA图片

    参数：
        img_rgb   : PIL RGB图片
        threshold : 背景判断阈值（三通道均超过此值视为背景）

    返回：
        PIL RGBA图片（背景透明）
    """
    img_rgba = img_rgb.convert("RGBA")
    data     = np.array(img_rgba, dtype=np.uint8)

    r, g, b  = data[:,:,0], data[:,:,1], data[:,:,2]
    bg_mask  = (r > threshold) & (g > threshold) & (b > threshold)
    data[:,:,3] = np.where(bg_mask, 0, 255)

    return Image.fromarray(data, "RGBA")


def shift_hue(img_rgba, degree):
    """
    对RGBA图片做色相偏移

    参数：
        img_rgba : PIL RGBA图片
        degree   : 色相偏移角度（0~360）

    返回：
        PIL RGBA图片
    """
    if degree == 0:
        return img_rgba

    r, g, b, a = img_rgba.split()
    img_rgb    = Image.merge("RGB", (r, g, b))

    data_hsv   = np.array(img_rgb.convert("HSV"), dtype=np.float32)

    # H通道范围0~255对应0~360°
    shift            = degree / 360.0 * 255.0
    data_hsv[:,:,0]  = (data_hsv[:,:,0] + shift) % 256

    shifted_rgb = Image.fromarray(data_hsv.astype(np.uint8), "HSV")
    shifted_rgb = shifted_rgb.convert("RGB")

    sr, sg, sb  = shifted_rgb.split()
    result      = Image.merge("RGBA", (sr, sg, sb, a))

    return result


def to_grey(img_rgba):
    """
    把彩色RGBA图片转为灰色（去饱和）

    原理：RGB转灰度后重新映射到[100,210]范围，
    保留零件明暗细节，整体呈中灰

    参数：
        img_rgba : PIL RGBA图片

    返回：
        PIL RGBA图片
    """
    r, g, b, a = img_rgba.split()
    img_rgb    = Image.merge("RGB", (r, g, b))

    grey       = np.array(img_rgb.convert("L"), dtype=np.float32)

    # 线性映射到 [100, 210]，保留明暗关系
    g_min = grey.min()
    g_max = grey.max()
    if g_max > g_min:
        out = (grey - g_min) / (g_max - g_min) * 110.0 + 100.0
    else:
        out = np.ones_like(grey) * 155.0

    out_uint8  = np.clip(out, 0, 255).astype(np.uint8)
    result_rgb = Image.fromarray(
        np.stack([out_uint8, out_uint8, out_uint8], axis=2), "RGB"
    )

    rr, rg, rb  = result_rgb.split()
    result_rgba = Image.merge("RGBA", (rr, rg, rb, a))

    return result_rgba


def place_on_background(img_rgba, bg_colour, canvas_size, object_scale):
    """
    把透明背景的RGBA零件图贴到纯色背景上，零件居中

    参数：
        img_rgba     : PIL RGBA图片（背景透明）
        bg_colour    : 背景RGB元组
        canvas_size  : 画布边长（像素）
        object_scale : 零件占画布比例

    返回：
        PIL RGB图片（224×224）
    """
    canvas   = Image.new("RGB", (canvas_size, canvas_size), bg_colour)
    obj_size = int(canvas_size * object_scale)

    img_resized = img_rgba.copy()
    img_resized.thumbnail((obj_size, obj_size), Image.LANCZOS)

    w, h    = img_resized.size
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2

    canvas.paste(img_resized, (paste_x, paste_y), mask=img_resized)

    return canvas


# ============================================================
# 阶段一：生成7种颜色原始图
# ============================================================
def stage1_generate_raw_colour():
    """
    139张大红色原始图 → 973张7色原始图（带透明背景）
    保存到 raw_data_colour/colour_00~colour_06/
    """
    print("\n" + "=" * 60)
    print("  阶段一：生成7种颜色原始图")
    print("=" * 60)

    for folder_name, _, _ in COLOUR_CONFIGS:
        os.makedirs(os.path.join(RAW_COLOUR_DIR, folder_name),
                    exist_ok=True)

    # 收集所有原始图路径
    raw_paths = []
    for shape_folder in sorted(os.listdir(RAW_SHAPE_DIR)):
        shape_dir = os.path.join(RAW_SHAPE_DIR, shape_folder)
        if not os.path.isdir(shape_dir):
            continue
        for img_file in sorted(os.listdir(shape_dir)):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                raw_paths.append(os.path.join(shape_dir, img_file))

    total_raw = len(raw_paths)
    print(f"  找到原始图：{total_raw} 张")
    print(f"  将生成：{total_raw} × 7 = {total_raw * 7} 张")

    generated = 0

    for img_idx, img_path in enumerate(raw_paths):

        img_rgb  = Image.open(img_path).convert("RGB")
        img_rgba = remove_white_background(img_rgb)

        shape_folder = os.path.basename(os.path.dirname(img_path))
        base_name    = os.path.splitext(os.path.basename(img_path))[0]
        file_stem    = f"{shape_folder}_{base_name}"

        for folder_name, colour_type, param in COLOUR_CONFIGS:

            if colour_type == "hue":
                result_rgba = shift_hue(img_rgba, degree=param)
            else:
                result_rgba = to_grey(img_rgba)

            out_path = os.path.join(
                RAW_COLOUR_DIR, folder_name,
                f"{file_stem}.png"
            )
            result_rgba.save(out_path)
            generated += 1

        if (img_idx + 1) % 20 == 0 or (img_idx + 1) == total_raw:
            print(f"  进度：{img_idx + 1}/{total_raw} 张已处理"
                  f"（已生成 {generated} 张）")

    print(f"\n  ✅ 阶段一完成！共生成 {generated} 张")
    print(f"  保存位置：{RAW_COLOUR_DIR}")


# ============================================================
# 阶段二：贴背景，生成 layer_one_colour
# ============================================================
def stage2_generate_layer_one():
    """
    973张原始颜色图 → 7784张带背景图（973×8）
    保存到 layer_one_colour/colour_00~colour_06/
    """
    print("\n" + "=" * 60)
    print("  阶段二：贴背景，生成 layer_one_colour")
    print("=" * 60)

    for folder_name, _, _ in COLOUR_CONFIGS:
        os.makedirs(os.path.join(LAYER_ONE_DIR, folder_name),
                    exist_ok=True)

    total_generated = 0

    for folder_name, _, _ in COLOUR_CONFIGS:

        colour_raw_dir = os.path.join(RAW_COLOUR_DIR, folder_name)
        colour_out_dir = os.path.join(LAYER_ONE_DIR,  folder_name)

        img_files     = sorted([
            f for f in os.listdir(colour_raw_dir)
            if f.lower().endswith('.png')
        ])

        colour_count = 0

        for img_file in img_files:
            img_path  = os.path.join(colour_raw_dir, img_file)
            img_rgba  = Image.open(img_path).convert("RGBA")
            base_name = os.path.splitext(img_file)[0]

            for bg_idx, bg_colour in enumerate(BACKGROUND_COLOURS):
                result_rgb = place_on_background(
                    img_rgba,
                    bg_colour    = bg_colour,
                    canvas_size  = CANVAS_SIZE,
                    object_scale = OBJECT_SCALE
                )
                out_name = f"{base_name}_bg{bg_idx:02d}.png"
                out_path = os.path.join(colour_out_dir, out_name)
                result_rgb.save(out_path)
                colour_count    += 1
                total_generated += 1

        print(f"  {folder_name}：{len(img_files)} 张"
              f" → {colour_count} 张（×8背景）")

    print(f"\n  ✅ 阶段二完成！共生成 {total_generated} 张")
    print(f"  保存位置：{LAYER_ONE_DIR}")


# ============================================================
# 主函数
# ============================================================
def main():

    print("\n" + "=" * 60)
    print("  颜色识别数据集第一层生成脚本（7种颜色版）")
    print("=" * 60)
    print(f"  原始图来源：{RAW_SHAPE_DIR}")
    print(f"  7种颜色：大红/明黄/绿/青/蓝/玫瑰/灰")
    print(f"  白色和黑色由手动截图后单独处理")
    print(f"  预计总输出：{139 * 7} × 8 = {139 * 7 * 8} 张")

    stage1_generate_raw_colour()
    stage2_generate_layer_one()

    print("\n" + "=" * 60)
    print("  全部完成！")
    print(f"  layer_one_colour/ 已包含7种颜色")
    print(f"  待手动截图完成后，运行白黑专用脚本补充剩余2类")
    print("=" * 60)


if __name__ == "__main__":
    main()
