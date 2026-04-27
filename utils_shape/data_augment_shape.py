# ============================================================
# data_augment_shape.py
# 功能：对 raw_data_shape 里的原始图片进行程序化变体生成
#
# 四步随机操作：
#   第一步：色相偏移  → 6种颜色中随机取3种
#   第二步：位置偏移  → 5种位置中随机取3种
#   第三步：大小缩放  → 3种大小中随机取2种
#   第四步：背景颜色  → 6种浅色背景中随机取3种
#
# 每张原始图生成：3×3×2×3 = 54张变体
# 新增：自动去除白色背景，解决背景被误上色问题
# ============================================================

from PIL import Image
import numpy as np
import os
import random

# ============================================================
# 路径配置
# ============================================================
RAW_ROOT    = r"C:\Users\lsbt\Desktop\lego_part_classifier\raw_data_shape"
LAYER1_ROOT = r"C:\Users\lsbt\Desktop\lego_part_classifier\layer_one_shape"
CANVAS_SIZE = (224, 224)

# 随机种子：保证每次运行结果一致，方便复现
random.seed(42)
np.random.seed(42)

# ============================================================
# 第一步：6种色相偏移角度，每张图随机取3种
# ============================================================
ALL_HUE_SHIFTS = [0, 60, 120, 180, 240, 300]
# 0°   = 保持原色（大红）
# 60°  = 偏黄
# 120° = 偏绿
# 180° = 偏青
# 240° = 偏蓝
# 300° = 偏品红

# ============================================================
# 第二步：5种位置配置，每张图随机取3种
# 格式：(水平偏移比例, 垂直偏移比例)
# 正值=向右/向下，负值=向左/向上
# ============================================================
ALL_POSITIONS = [
    ( 0.00,  0.00),   # 居中
    (-0.15, -0.15),   # 左上
    (-0.15,  0.15),   # 左下
    ( 0.15, -0.15),   # 右上
    ( 0.15,  0.15),   # 右下
]

# ============================================================
# 第三步：3种大小缩放比例，每张图随机取2种
# ============================================================
ALL_SCALES = [
    0.70,    # 标准大小（占画布70%）
    0.93,    # 放大1.33倍
    0.525,   # 缩小0.75倍
]

# ============================================================
# 第四步：6种浅色背景，每张图随机取3种
# ============================================================
ALL_BACKGROUNDS = [
    (255, 255, 255),   # 纯白
    (230, 230, 230),   # 浅灰
    (210, 230, 255),   # 浅蓝
    (255, 240, 220),   # 浅橙米白
    (220, 255, 220),   # 浅绿
    (255, 220, 240),   # 浅粉
]

# ============================================================
# 预处理函数：自动去除白色背景，转为透明
# ============================================================
def remove_white_background(img, threshold=240, edge_tolerance=10):
    """
    把图片中的白色/浅色背景替换为透明
    适用于背景为纯白或接近白色的PNG图片

    参数：
        img            : PIL图片对象
        threshold      : 判断为背景的亮度阈值（0~255）
                         高于此值的浅色像素视为背景
                         默认240，即非常接近白色才算背景
        edge_tolerance : 边缘容差，防止把积木边缘也变透明

    返回：
        RGBA格式的PIL图片（背景变透明）
    """
    # 转为RGBA格式，确保有透明度通道
    img_rgba = img.convert("RGBA")
    data = np.array(img_rgba, dtype=np.float32)

    R = data[..., 0]
    G = data[..., 1]
    B = data[..., 2]
    A = data[..., 3]

    # 判断哪些像素是"白色背景"
    # 条件1：R、G、B都高于阈值（足够亮）
    # 条件2：R、G、B差异很小（接近灰色/白色，不是彩色）
    is_white = (
        (R > threshold) &
        (G > threshold) &
        (B > threshold) &
        (np.abs(R - G) < edge_tolerance) &
        (np.abs(G - B) < edge_tolerance) &
        (np.abs(R - B) < edge_tolerance)
    )

    # 把白色背景像素的Alpha通道设为0（完全透明）
    A[is_white] = 0

    # 写回Alpha通道
    data[..., 3] = A
    result = data.astype(np.uint8)

    return Image.fromarray(result, 'RGBA')

# ============================================================
# 核心函数1：把图片放到指定背景的画布上
# ============================================================
def place_on_canvas(img, bg_color, scale, pos_offset):
    """
    把积木图片以指定大小和位置放到背景画布上

    参数：
        img        : PIL图片对象（已去除白色背景，RGBA格式）
        bg_color   : 背景颜色 (R, G, B)
        scale      : 积木缩放比例（相对于画布边长）
        pos_offset : 位置偏移 (水平偏移比例, 垂直偏移比例)

    返回：
        canvas     : 224×224 的 RGB PIL图片
    """
    offset_x_ratio, offset_y_ratio = pos_offset

    # 创建纯色背景画布
    canvas = Image.new("RGB", CANVAS_SIZE, bg_color)

    # 转为RGBA，保留透明度
    img_rgba = img.convert("RGBA")

    # 计算目标尺寸并缩放
    target_size = int(CANVAS_SIZE[0] * scale)
    target_size = max(target_size, 40)   # 最小40像素，防止太小
    img_rgba = img_rgba.resize((target_size, target_size), Image.LANCZOS)

    # 计算居中位置 + 偏移
    center_x = (CANVAS_SIZE[0] - img_rgba.width)  // 2
    center_y = (CANVAS_SIZE[1] - img_rgba.height) // 2
    offset_x = int(CANVAS_SIZE[0] * offset_x_ratio)
    offset_y = int(CANVAS_SIZE[1] * offset_y_ratio)

    paste_x = center_x + offset_x
    paste_y = center_y + offset_y

    # 用透明度通道作蒙版粘贴，保留积木轮廓
    canvas.paste(img_rgba, (paste_x, paste_y), img_rgba.split()[3])

    return canvas

# ============================================================
# 核心函数2：numpy向量化色相偏移（快速版）
# ============================================================
def shift_hue_fast(img, hue_shift_deg):
    """
    对图片做色相偏移，只改变有颜色的像素（饱和度>0.15）
    灰色/白色/黑色的高光和阴影保持不变

    参数：
        img           : PIL图片对象（RGB格式）
        hue_shift_deg : 色相偏移角度（0~360）

    返回：
        PIL图片对象
    """
    if hue_shift_deg == 0:
        return img.copy()

    # 转numpy数组，归一化到0~1
    arr = np.array(img, dtype=np.float32) / 255.0

    R, G, B = arr[..., 0], arr[..., 1], arr[..., 2]

    # RGB → HSV
    Cmax  = np.maximum(np.maximum(R, G), B)
    Cmin  = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    V = Cmax
    S = np.where(Cmax > 0, delta / Cmax, 0.0)

    H = np.zeros_like(R)
    mask_r = (Cmax == R) & (delta > 0)
    mask_g = (Cmax == G) & (delta > 0)
    mask_b = (Cmax == B) & (delta > 0)
    H[mask_r] = (((G - B) / delta) % 6)[mask_r]
    H[mask_g] = ((B - R) / delta + 2)[mask_g]
    H[mask_b] = ((R - G) / delta + 4)[mask_b]
    H = H / 6.0

    # 只对饱和度>0.15的像素做色相偏移
    hue_shift = hue_shift_deg / 360.0
    H = np.where(S > 0.15, (H + hue_shift) % 1.0, H)

    # HSV → RGB
    H6 = H * 6.0
    hi = np.floor(H6).astype(np.int32) % 6
    f  = H6 - np.floor(H6)
    p  = V * (1 - S)
    q  = V * (1 - f * S)
    t  = V * (1 - (1 - f) * S)

    R_new = np.select([hi==0,hi==1,hi==2,hi==3,hi==4,hi==5],[V,q,p,p,t,V])
    G_new = np.select([hi==0,hi==1,hi==2,hi==3,hi==4,hi==5],[t,V,V,q,p,p])
    B_new = np.select([hi==0,hi==1,hi==2,hi==3,hi==4,hi==5],[p,p,t,V,V,q])

    result = np.stack([R_new, G_new, B_new], axis=-1)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)

    return Image.fromarray(result)

# ============================================================
# 主函数：生成所有变体
# ============================================================
def generate_layer1():
    """
    对每张原始图：
    0. 预处理：自动去除白色背景
    1. 随机选3种色相
    2. 随机选3种位置
    3. 随机选2种大小
    4. 随机选3种背景
    组合生成 3×3×2×3 = 54 张变体
    """
    class_folders = sorted(os.listdir(RAW_ROOT))
    print(f"找到 {len(class_folders)} 个类别文件夹")
    print(f"每张原始图生成：3×3×2×3 = 54 张变体")
    print(f"新增：自动去除白色背景\n")

    total_generated = 0

    for class_name in class_folders:
        class_input_path  = os.path.join(RAW_ROOT,    class_name)
        class_output_path = os.path.join(LAYER1_ROOT, class_name)

        if not os.path.isdir(class_input_path):
            continue

        os.makedirs(class_output_path, exist_ok=True)

        image_files = [
            f for f in os.listdir(class_input_path)
            if f.lower().endswith('.png')
        ]

        print(f"  处理 {class_name}：{len(image_files)} 张原始图", end="")

        img_count = 0

        for img_file in image_files:
            img_path     = os.path.join(class_input_path, img_file)

            # 读取原始图片
            original_img = Image.open(img_path)

            # ★ 新增：自动去除白色背景，转为透明
            original_img = remove_white_background(original_img)

            base_name = os.path.splitext(img_file)[0]

            # ── 四步随机抽样 ──────────────────────────────
            chosen_hues   = random.sample(ALL_HUE_SHIFTS,  3)  # 6选3
            chosen_pos    = random.sample(ALL_POSITIONS,   3)  # 5选3
            chosen_scales = random.sample(ALL_SCALES,      2)  # 3选2
            chosen_bgs    = random.sample(ALL_BACKGROUNDS, 3)  # 6选3
            # ──────────────────────────────────────────────

            # 四层嵌套组合，生成54张
            variant_idx = 0
            for hue in chosen_hues:
                for pos in chosen_pos:
                    for scale in chosen_scales:
                        for bg in chosen_bgs:

                            # 放置到画布（背景+大小+位置）
                            canvas = place_on_canvas(
                                original_img, bg, scale, pos
                            )

                            # 色相偏移
                            final_img = shift_hue_fast(canvas, hue)

                            # 保存
                            out_name = f"{base_name}_v{variant_idx:03d}.png"
                            out_path = os.path.join(
                                class_output_path, out_name
                            )
                            final_img.save(out_path)
                            variant_idx += 1
                            img_count   += 1

        total_generated += img_count
        print(f" → 生成 {img_count} 张")

    print("\n" + "=" * 45)
    print(f"第一层变体生成完成！")
    print(f"总生成图片数：{total_generated} 张")
    print(f"输出目录：{LAYER1_ROOT}")

# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    generate_layer1()
