# ============================================================
# utils_colour/add_white_black.py
#
# 功能：为手动截取的白色和黑色零件图生成第一层增强数据
#
# 输入：
#   raw_data_colour/colour_07/  ← 手动放入的白色零件截图
#   raw_data_colour/colour_08/  ← 手动放入的黑色零件截图
#
# 处理：
#   去除白色背景 → 贴8种固定背景色 → 居中缩放0.75
#
# 输出：
#   layer_one_colour/colour_07/  白色零件，每张原图×8背景
#   layer_one_colour/colour_08/  黑色零件，每张原图×8背景
#
# 注意：
#   本脚本只处理 colour_07 和 colour_08
#   不影响已生成的 colour_00~colour_06 数据
#
# 运行方式（项目根目录下）：
#   python utils_colour/add_white_black.py
# ============================================================

import os
import sys
import numpy as np
from PIL import Image

# ── 把项目根目录加入路径 ──────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)


# ============================================================
# 配置
# ============================================================

RAW_COLOUR_DIR = os.path.join(ROOT_DIR, "raw_data_colour")
LAYER_ONE_DIR  = os.path.join(ROOT_DIR, "layer_one_colour")

CANVAS_SIZE    = 224
OBJECT_SCALE   = 0.75
WHITE_THRESHOLD = 240

# 只处理这两个文件夹
TARGET_FOLDERS = ["colour_07", "colour_08"]

# 8种背景色（与 data_augment_colour_layer1.py 完全一致）
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


# ============================================================
# 工具函数（与主脚本完全一致）
# ============================================================

def remove_white_background(img_rgb, threshold=WHITE_THRESHOLD):
    """去除白色背景，返回RGBA透明图"""
    img_rgba = img_rgb.convert("RGBA")
    data     = np.array(img_rgba, dtype=np.uint8)
    r, g, b  = data[:,:,0], data[:,:,1], data[:,:,2]
    bg_mask  = (r > threshold) & (g > threshold) & (b > threshold)
    data[:,:,3] = np.where(bg_mask, 0, 255)
    return Image.fromarray(data, "RGBA")


def place_on_background(img_rgba, bg_colour, canvas_size, object_scale):
    """零件居中贴到纯色背景上，返回RGB图"""
    canvas      = Image.new("RGB", (canvas_size, canvas_size), bg_colour)
    obj_size    = int(canvas_size * object_scale)
    img_resized = img_rgba.copy()
    img_resized.thumbnail((obj_size, obj_size), Image.LANCZOS)
    w, h    = img_resized.size
    paste_x = (canvas_size - w) // 2
    paste_y = (canvas_size - h) // 2
    canvas.paste(img_resized, (paste_x, paste_y), mask=img_resized)
    return canvas


# ============================================================
# 主处理函数
# ============================================================

def process_folder(folder_name):
    """
    处理单个颜色文件夹（colour_07 或 colour_08）

    参数：
        folder_name : 'colour_07' 或 'colour_08'
    """
    raw_dir = os.path.join(RAW_COLOUR_DIR, folder_name)
    out_dir = os.path.join(LAYER_ONE_DIR,  folder_name)

    # 检查输入目录是否存在且有图片
    if not os.path.exists(raw_dir):
        print(f"  ⚠️  {folder_name}：原始图目录不存在，跳过")
        print(f"       请先把截图放入：{raw_dir}")
        return

    img_files = sorted([
        f for f in os.listdir(raw_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    if len(img_files) == 0:
        print(f"  ⚠️  {folder_name}：目录为空，跳过")
        print(f"       请先把截图放入：{raw_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)

    print(f"\n  处理 {folder_name}：{len(img_files)} 张原始图")

    count = 0
    for img_file in img_files:
        img_path  = os.path.join(raw_dir, img_file)
        base_name = os.path.splitext(img_file)[0]

        # 读取并去背景
        img_rgb  = Image.open(img_path).convert("RGB")
        img_rgba = remove_white_background(img_rgb)

        # 贴8种背景
        for bg_idx, bg_colour in enumerate(BACKGROUND_COLOURS):
            result_rgb = place_on_background(
                img_rgba,
                bg_colour    = bg_colour,
                canvas_size  = CANVAS_SIZE,
                object_scale = OBJECT_SCALE
            )
            out_name = f"{base_name}_bg{bg_idx:02d}.png"
            out_path = os.path.join(out_dir, out_name)
            result_rgb.save(out_path)
            count += 1

    print(f"  ✅ {folder_name}：生成 {count} 张"
          f"（{len(img_files)} × 8背景）")
    print(f"     保存位置：{out_dir}")


def main():

    print("\n" + "=" * 60)
    print("  白色/黑色零件第一层增强脚本")
    print("=" * 60)

    for folder_name in TARGET_FOLDERS:
        process_folder(folder_name)

    print("\n" + "=" * 60)
    print("  完成！layer_one_colour 现已包含9种颜色")
    print("=" * 60)


if __name__ == "__main__":
    main()
