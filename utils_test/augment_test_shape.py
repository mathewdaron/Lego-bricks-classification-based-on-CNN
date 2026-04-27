# ============================================================
# augment_test_shape.py
# 形状测试集生成脚本
#
# 功能：
#   读取 raw_data_test/red/ 下的225张大红色积木图（45类各5张）
#   通过固定 + 随机增强，生成形状测试集
#   输出到 test/test_dataset_shape/shape_00/ ~ shape_44/
#
# 增强逻辑（每张原图生成48张）：
#   随机选4种背景色（共8种中选4）
#   随机选2种缩放比例（原尺寸/1.33倍/0.75倍 中选2）
#   随机选3种位置偏移（5种中选3）
#   以上组合：4 × 2 × 3 = 24张
#   每张再做一次水平翻转 → ×2 = 48张
#   所有48张各自做一次 ±30° 随机旋转（角度随机，数量不变）
#
# 用法（在项目根目录下运行）：
#   python augment_test_shape.py
# ============================================================

import os
import sys
import random
from PIL import Image, ImageFilter
import numpy as np

# ── 把项目根目录加入路径 ──────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ============================================================
# 配置区：所有参数集中在这里
# ============================================================
CONFIG = {
    # 输入：原始红色图片目录
    "input_dir"  : os.path.join(ROOT_DIR, "raw_data_test", "red"),
    # 输出：形状测试集目录
    "output_dir" : os.path.join(ROOT_DIR, "test", "test_dataset_shape"),
    # 输出图片尺寸（与训练集一致）
    "img_size"   : 224,
    # 随机种子（固定种子保证每次运行结果相同，方便复现）
    "random_seed": 42,
    # 白色背景去除阈值：RGB三通道都大于此值即视为背景
    # 纯白是255，设240是为了容忍轻微噪声和JPG压缩伪影
    "bg_threshold": 240,
    # 每张图随机选几种背景（从8种中选）
    "n_bg"       : 4,
    # 每张图随机选几种缩放（从3种中选）
    "n_scale"    : 2,
    # 每张图随机选几种位置（从5种中选）
    "n_position" : 3,
    # 随机旋转角度范围（度）
    "max_rotate" : 30,
}

# ── 8种背景色（与训练集完全相同）────────────────────────────
BACKGROUNDS = [
    (255, 252, 245),   # 暖白
    (255, 248, 220),   # 浅米黄
    (255, 228, 196),   # 浅杏色
    (210, 245, 235),   # 浅薄荷绿
    (210, 235, 255),   # 浅天空蓝
    (255, 220, 215),   # 浅藕粉
    (230, 220, 255),   # 浅薰衣草紫
    (255, 250, 200),   # 浅鹅黄
]

# ── 3种缩放比例 ───────────────────────────────────────────────
# 含义：积木在画布上占的比例
# 1.0 = 正常大小，1.33 = 放大，0.75 = 缩小
SCALES = [1.0, 1.33, 0.75]

# ── 5种位置偏移（相对于画布中心的偏移比例）──────────────────
# (dx, dy)：dx正=向右，dy正=向下；比例相对于画布尺寸
# 0.15 表示偏移量为画布尺寸的15%
POSITIONS = [
    (0.0,   0.0 ),   # 居中不偏移
    (-0.15, -0.15),  # 左上
    (-0.15,  0.15),  # 左下
    ( 0.15, -0.15),  # 右上
    ( 0.15,  0.15),  # 右下
]


# ============================================================
# 核心功能函数
# ============================================================

def remove_white_background(img, threshold=240):
    """
    去除纯白背景，返回带透明通道（RGBA）的图片

    原理：
        对每个像素，判断RGB三个通道是否都大于阈值
        满足条件的像素 → 透明（alpha=0）
        不满足的像素  → 保留原色（alpha=255）

    参数：
        img       : PIL Image，RGB模式
        threshold : 白色阈值，默认240
    返回：
        RGBA模式的PIL Image，背景已透明
    """
    # 转成numpy数组方便批量操作
    img_arr = np.array(img.convert("RGB"))
    # 判断每个像素是否为背景（三通道都超过阈值）
    # 结果是布尔数组，shape: (H, W)
    mask = (img_arr[:, :, 0] > threshold) & \
           (img_arr[:, :, 1] > threshold) & \
           (img_arr[:, :, 2] > threshold)

    # 创建RGBA图像（4通道，第4通道=透明度）
    rgba_arr = np.zeros(
        (img_arr.shape[0], img_arr.shape[1], 4), dtype=np.uint8
    )
    rgba_arr[:, :, :3] = img_arr          # 复制RGB通道
    rgba_arr[:, :, 3]  = 255              # 默认全不透明
    rgba_arr[mask, 3]  = 0               # 背景像素设为透明

    return Image.fromarray(rgba_arr, mode="RGBA")


def place_on_background(fg_rgba, bg_color, canvas_size,
                        scale, position):
    """
    把透明背景的积木图贴到纯色背景上

    参数：
        fg_rgba     : RGBA模式的积木图（已去背景）
        bg_color    : 背景颜色 (R, G, B)
        canvas_size : 输出画布大小（正方形边长，如224）
        scale       : 缩放比例（1.0=原始比例适应画布）
        position    : (dx, dy) 偏移比例

    返回：
        RGB模式的PIL Image，尺寸为 canvas_size × canvas_size
    """
    # 创建纯色背景画布
    canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)

    # 计算积木缩放后的目标尺寸
    # 先把积木等比缩放到适应画布大小（留一点边距），再乘以scale
    fg_w, fg_h = fg_rgba.size
    # 等比缩放：让积木的长边适应画布的80%（留20%边距）
    base_size = int(canvas_size * 0.80)
    ratio = base_size / max(fg_w, fg_h)
    new_w = int(fg_w * ratio * scale)
    new_h = int(fg_h * ratio * scale)

    # 防止缩放后尺寸为0
    new_w = max(new_w, 10)
    new_h = max(new_h, 10)

    # 缩放积木（LANCZOS是高质量插值算法）
    fg_resized = fg_rgba.resize((new_w, new_h), Image.LANCZOS)

    # 计算贴图位置：画布中心 + 偏移
    dx_pixels = int(position[0] * canvas_size)
    dy_pixels = int(position[1] * canvas_size)

    # 贴图左上角坐标
    paste_x = (canvas_size - new_w) // 2 + dx_pixels
    paste_y = (canvas_size - new_h) // 2 + dy_pixels

    # 把积木贴到背景上（第三个参数是透明度mask，实现透明贴图）
    canvas.paste(fg_resized, (paste_x, paste_y), fg_resized)

    return canvas


def random_rotate(img, max_angle=30):
    """
    对图片做随机旋转

    参数：
        img       : RGB模式PIL Image
        max_angle : 最大旋转角度（正负范围），默认±30°

    返回：
        旋转后的PIL Image

    注意：
        旋转后边角会出现空白，用白色填充（与测试脚本的Normalize一致）
        expand=False：保持画布尺寸不变（不裁剪也不扩大）
    """
    angle = random.uniform(-max_angle, max_angle)
    # fillcolor：旋转后的空白区域用接近白色填充
    # 因为后续模型输入时会做Normalize，白色区域不影响特征提取
    rotated = img.rotate(
        angle,
        resample=Image.BICUBIC,   # 双三次插值，旋转质量好
        expand=False,              # 保持原尺寸
        fillcolor=(255, 255, 255)  # 空白处填白色
    )
    return rotated


# ============================================================
# 主流程
# ============================================================
def main():

    # 固定随机种子，保证每次运行生成的是完全相同的测试集
    random.seed(CONFIG["random_seed"])
    np.random.seed(CONFIG["random_seed"])

    input_dir  = CONFIG["input_dir"]
    output_dir = CONFIG["output_dir"]
    img_size   = CONFIG["img_size"]

    # 检查输入目录
    if not os.path.exists(input_dir):
        print(f"❌ 错误：输入目录不存在：{input_dir}")
        return

    # 获取所有类别文件夹（shape_00 ~ shape_44），排序保证一致
    class_folders = sorted([
        d for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])

    if len(class_folders) == 0:
        print(f"❌ 错误：在 {input_dir} 下没有找到任何子文件夹")
        print(f"   请确认结构为：raw_data_test/red/shape_00/ ~ shape_44/")
        return

    print(f"\n{'='*60}")
    print(f"  形状测试集生成")
    print(f"{'='*60}")
    print(f"  输入目录：{input_dir}")
    print(f"  输出目录：{output_dir}")
    print(f"  类别数量：{len(class_folders)}")
    print(f"  每张原图生成数量：")
    print(f"    背景 {CONFIG['n_bg']} 种 × "
          f"缩放 {CONFIG['n_scale']} 种 × "
          f"位置 {CONFIG['n_position']} 种 × "
          f"翻转 2（原图+翻转）= "
          f"{CONFIG['n_bg']*CONFIG['n_scale']*CONFIG['n_position']*2} 张")
    print(f"  所有图片额外做 ±{CONFIG['max_rotate']}° 随机旋转")
    print(f"{'='*60}\n")

    total_generated = 0  # 统计总生成数

    for class_name in class_folders:

        class_input_dir  = os.path.join(input_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)

        # 获取该类所有图片
        img_files = sorted([
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        class_count = 0  # 该类已生成数量

        for img_file in img_files:

            img_path = os.path.join(class_input_dir, img_file)
            # 读取原图
            original = Image.open(img_path).convert("RGB")

            # Step1：去白色背景 → RGBA透明图
            fg_rgba = remove_white_background(
                original, CONFIG["bg_threshold"]
            )

            # Step2：随机选背景、缩放、位置的组合
            # 每次固定抽取，保证可复现（已设置随机种子）
            chosen_bgs       = random.sample(BACKGROUNDS, CONFIG["n_bg"])
            chosen_scales    = random.sample(SCALES,      CONFIG["n_scale"])
            chosen_positions = random.sample(POSITIONS,   CONFIG["n_position"])

            # Step3：三重循环，生成所有组合
            img_index = 0  # 该原图的输出序号

            for bg_color in chosen_bgs:
                for scale in chosen_scales:
                    for position in chosen_positions:

                        # 生成贴图后的正常图（未翻转）
                        composed = place_on_background(
                            fg_rgba, bg_color, img_size, scale, position
                        )

                        # 水平翻转版
                        flipped = composed.transpose(
                            Image.FLIP_LEFT_RIGHT
                        )

                        # 对两张各自做随机旋转（角度独立随机）
                        rotated_normal  = random_rotate(
                            composed, CONFIG["max_rotate"]
                        )
                        rotated_flipped = random_rotate(
                            flipped,  CONFIG["max_rotate"]
                        )

                        # 保存原图（旋转后）
                        base_name = os.path.splitext(img_file)[0]
                        save_name_n = (
                            f"{base_name}_aug{img_index:03d}_n.png"
                        )
                        rotated_normal.save(
                            os.path.join(class_output_dir, save_name_n)
                        )

                        # 保存翻转版（旋转后）
                        save_name_f = (
                            f"{base_name}_aug{img_index:03d}_f.png"
                        )
                        rotated_flipped.save(
                            os.path.join(class_output_dir, save_name_f)
                        )

                        img_index  += 1
                        class_count += 2

        total_generated += class_count
        print(f"  ✅ {class_name}：生成 {class_count} 张")

    print(f"\n{'='*60}")
    print(f"  全部完成！")
    print(f"  总生成图片数：{total_generated}")
    print(f"  输出目录：{output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
