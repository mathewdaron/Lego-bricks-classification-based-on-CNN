# ============================================================
# augment_test_colour.py
# 颜色测试集生成脚本
#
# 功能：
#   Part A：读取 raw_data_test/red/ 的225张红色图
#           通过色相旋转生成7种颜色（红/黄/绿/青/蓝/玫瑰/灰）
#           每种颜色 × 8种背景 = 1800张
#           输出到 colour_00 ~ colour_06
#
#   Part B：读取 raw_data_test/white/ 和 raw_data_test/black/ 的各139张
#           直接换背景（不改色相）× 8种背景 = 1112张
#           输出到 colour_07 / colour_08
#
# 用法（在项目根目录下运行）：
#   python utils_test/augment_test_colour.py
# ============================================================

import os
import random
import colorsys
import numpy as np
from PIL import Image

# ── 项目根目录 ────────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ============================================================
# 配置区（路径全部保留不动）
# ============================================================
CONFIG = {
    "input_red"   : os.path.join(ROOT_DIR, "raw_data_test", "red"),
    "input_white" : os.path.join(ROOT_DIR, "raw_data_test", "white"),
    "input_black" : os.path.join(ROOT_DIR, "raw_data_test", "black"),
    "output_dir"  : os.path.join(ROOT_DIR, "test", "test_dataset_colour"),
    "img_size"    : 224,
    "bg_threshold": 240,
}

# ── 8种背景色（与训练集完全相同）────────────────────────────
BACKGROUNDS = [
    (255, 252, 245),
    (255, 248, 220),
    (255, 228, 196),
    (210, 245, 235),
    (210, 235, 255),
    (255, 220, 215),
    (230, 220, 255),
    (255, 250, 200),
]

# ── 7种颜色定义（colour_00 ~ colour_06）────────────────────
COLOUR_CONFIGS = [
    ("colour_00", 0,    False),
    ("colour_01", 60,   False),
    ("colour_02", 120,  False),
    ("colour_03", 180,  False),
    ("colour_04", 240,  False),
    ("colour_05", 300,  False),
    ("colour_06", 0,    True ),
]


# ============================================================
# 核心功能函数
# ============================================================

def remove_white_background(img, threshold=240):
    """去除纯白背景，返回RGBA透明图"""
    img_arr = np.array(img.convert("RGB"))
    mask = (img_arr[:, :, 0] > threshold) & \
           (img_arr[:, :, 1] > threshold) & \
           (img_arr[:, :, 2] > threshold)

    rgba_arr = np.zeros(
        (img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8
    )
    rgba_arr[:, :, :3] = img_arr
    rgba_arr[:, :, 3]  = 255
    rgba_arr[mask, 3]  = 0

    return Image.fromarray(rgba_arr, mode="RGBA")


def shift_hue(img_rgba, hue_shift_deg):
    """对RGBA图片的非透明像素做色相旋转"""
    arr = np.array(img_rgba, dtype=np.float32)
    mask = arr[:, :, 3] > 0
    rgb_pixels = arr[mask, :3] / 255.0

    new_rgb = []
    hue_shift = hue_shift_deg / 360.0

    for r, g, b in rgb_pixels:
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        if s > 0.1:
            h = (h + hue_shift) % 1.0
        new_r, new_g, new_b = colorsys.hsv_to_rgb(h, s, v)
        new_rgb.append([new_r, new_g, new_b])

    new_rgb = np.array(new_rgb, dtype=np.float32)
    arr[mask, :3] = new_rgb * 255.0

    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


def to_gray(img_rgba):
    """
    把RGBA图片的积木部分转为标准灰色

    ⚠️ 修正说明（与训练集保持一致）：
        原方法：HSV去饱和（S=0，V保持原值）
        问题：大红色V值接近1.0，去饱和后颜色接近纯白
        
        正确方法：RGB→L灰度图，线性映射到[100, 210]
        效果：整体呈标准中灰，与训练集 to_grey() 完全一致
    """
    # 拆分RGBA，单独处理alpha通道
    r, g, b, a = img_rgba.split()
    img_rgb = Image.merge("RGB", (r, g, b))

    # RGB → L（单通道灰度），保留积木明暗立体感
    grey = np.array(img_rgb.convert("L"), dtype=np.float32)

    # 线性映射到 [100, 210]：
    #   最暗的像素映射到100（深灰），最亮的映射到210（浅灰）
    #   整体落在中灰区间，不会出现接近白色的情况
    g_min = grey.min()
    g_max = grey.max()
    if g_max > g_min:
        out = (grey - g_min) / (g_max - g_min) * 110.0 + 100.0
    else:
        # 极端情况：图片完全均匀，直接设为中灰155
        out = np.ones_like(grey) * 155.0

    out_uint8 = np.clip(out, 0, 255).astype(np.uint8)

    # 把灰度值同时赋给R/G/B三通道（灰色 = R=G=B）
    result_rgb = Image.fromarray(
        np.stack([out_uint8, out_uint8, out_uint8], axis=2), "RGB"
    )

    # 把原始alpha通道接回去（透明背景保留）
    rr, rg, rb = result_rgb.split()
    result_rgba = Image.merge("RGBA", (rr, rg, rb, a))

    return result_rgba


def place_on_background(fg_rgba, bg_color, canvas_size):
    """把透明背景的积木图贴到纯色背景上（居中，不做缩放/偏移）"""
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)

    fg_w, fg_h = fg_rgba.size
    base_size  = int(canvas_size * 0.80)
    ratio      = base_size / max(fg_w, fg_h)
    new_w      = max(int(fg_w * ratio), 10)
    new_h      = max(int(fg_h * ratio), 10)

    fg_resized = fg_rgba.resize((new_w, new_h), Image.LANCZOS)

    paste_x = (canvas_size - new_w) // 2
    paste_y = (canvas_size - new_h) // 2
    canvas.paste(fg_resized, (paste_x, paste_y), fg_resized)

    return canvas


# ============================================================
# Part A：处理红色图片 → 生成7种颜色
# ============================================================
def process_colour_from_red(input_dir, output_dir, img_size, threshold):
    """读取红色积木图，通过色相旋转生成7种颜色，各换8种背景"""
    print(f"\n{'─'*50}")
    print(f"  Part A：红色图 → 7种颜色（colour_00~06）")
    print(f"{'─'*50}")

    all_img_paths = []
    if not os.path.exists(input_dir):
        print(f"❌ 错误：红色图目录不存在：{input_dir}")
        return

    for sub in sorted(os.listdir(input_dir)):
        sub_path = os.path.join(input_dir, sub)
        if not os.path.isdir(sub_path):
            continue
        for f in sorted(os.listdir(sub_path)):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_img_paths.append(
                    (os.path.join(sub_path, f), sub, f)
                )

    print(f"  找到红色原图：{len(all_img_paths)} 张")

    for folder_name, hue_shift, is_gray in COLOUR_CONFIGS:

        colour_output_dir = os.path.join(output_dir, folder_name)
        os.makedirs(colour_output_dir, exist_ok=True)

        count = 0

        for img_path, shape_cls, img_file in all_img_paths:

            original = Image.open(img_path).convert("RGB")
            fg_rgba  = remove_white_background(original, threshold)

            if is_gray:
                fg_colored = to_gray(fg_rgba)       # ← 使用修正后的函数
            elif hue_shift == 0:
                fg_colored = fg_rgba
            else:
                fg_colored = shift_hue(fg_rgba, hue_shift)

            base_name = os.path.splitext(img_file)[0]
            for bg_idx, bg_color in enumerate(BACKGROUNDS):
                result = place_on_background(
                    fg_colored, bg_color, img_size
                )
                save_name = (
                    f"{shape_cls}_{base_name}_bg{bg_idx:02d}.png"
                )
                result.save(
                    os.path.join(colour_output_dir, save_name)
                )
                count += 1

        print(f"  ✅ {folder_name}（色相{hue_shift}°"
              f"{'，灰色' if is_gray else ''}）：生成 {count} 张")


# ============================================================
# Part B：处理白色/黑色图片 → 直接换背景
# ============================================================
def process_white_black(input_dir, output_folder,
                        output_dir, img_size, threshold):
    """读取白色或黑色积木图，去背景后换8种背景色"""
    print(f"\n{'─'*50}")
    print(f"  Part B：{output_folder}（{input_dir.split(os.sep)[-1]}）")
    print(f"{'─'*50}")

    colour_output_dir = os.path.join(output_dir, output_folder)
    os.makedirs(colour_output_dir, exist_ok=True)

    if not os.path.exists(input_dir):
        print(f"❌ 错误：目录不存在：{input_dir}")
        return

    img_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"  找到原图：{len(img_files)} 张")

    count = 0
    for img_file in img_files:
        img_path = os.path.join(input_dir, img_file)
        original = Image.open(img_path).convert("RGB")
        fg_rgba  = remove_white_background(original, threshold)

        base_name = os.path.splitext(img_file)[0]
        for bg_idx, bg_color in enumerate(BACKGROUNDS):
            result = place_on_background(fg_rgba, bg_color, img_size)
            save_name = f"{base_name}_bg{bg_idx:02d}.png"
            result.save(os.path.join(colour_output_dir, save_name))
            count += 1

    print(f"  ✅ {output_folder}：生成 {count} 张")


# ============================================================
# 主函数
# ============================================================
def main():

    output_dir = CONFIG["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  颜色测试集生成（修正版：灰色使用L灰度映射）")
    print(f"{'='*60}")
    print(f"  输出目录：{output_dir}")
    print(f"  预计生成：")
    print(f"    colour_00~06：225 × 8 × 7 = {225*8*7} 张")
    print(f"    colour_07~08：139 × 8 × 2 = {139*8*2} 张")
    print(f"    合计：{225*8*7 + 139*8*2} 张")
    print(f"{'='*60}")

    process_colour_from_red(
        input_dir  = CONFIG["input_red"],
        output_dir = output_dir,
        img_size   = CONFIG["img_size"],
        threshold  = CONFIG["bg_threshold"],
    )

    process_white_black(
        input_dir     = CONFIG["input_white"],
        output_folder = "colour_07",
        output_dir    = output_dir,
        img_size      = CONFIG["img_size"],
        threshold     = CONFIG["bg_threshold"],
    )

    process_white_black(
        input_dir     = CONFIG["input_black"],
        output_folder = "colour_08",
        output_dir    = output_dir,
        img_size      = CONFIG["img_size"],
        threshold     = CONFIG["bg_threshold"],
    )

    print(f"\n{'='*60}")
    print(f"  全部完成！输出目录：{output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
