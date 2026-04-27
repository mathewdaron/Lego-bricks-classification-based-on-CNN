# ============================================================
# data_augment_shape_v2.py
# 乐高形状识别 —— 第一轮离线数据增强脚本（新版）
#
# 输入：raw_data_shape/（45个子文件夹，每类约8张原图，共约360张）
# 输出：layer_one_shape/（每张原图生成36张，总计约12960张）
#
# 增强逻辑（每张原图固定生成36张）：
#   基础框架（6种组合）：背景色8选2 × 缩放3选1 × 位置5选3
#   形变乘数（×6）：水平翻转×1 → 仿射×2 → 旋转×3（串联叠加）
#   总计：6 × 6 = 36张
#
# 关键设计：
#   全程在RGBA透明图上做几何变换，最后才贴背景色
#   这样仿射/旋转产生的空白区域 = 透明，而非白色
#   避免白色残留干扰背景替换
#
# 运行方式（在项目根目录下）：
#   python data_augment_shape_v2.py
# ============================================================

import os
import random
import math
import numpy as np
from PIL import Image

# ============================================================
# 路径配置
# ============================================================
RAW_ROOT    = r"C:\Users\lsbt\Desktop\lego_part_classifier\raw_data_shape"
OUTPUT_ROOT = r"C:\Users\lsbt\Desktop\lego_part_classifier\layer_one_shape"
CANVAS_SIZE = 224   # 输出图片尺寸（正方形）

# 固定随机种子，保证每次运行结果完全相同（可复现）
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ============================================================
# 参数配置：所有增强选项集中在这里
# ============================================================

# ── 8种背景色（浅色系，与测试集保持一致）────────────────────
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
N_BG = 2   # 每张原图从8种中随机选2种（组合，不是顺序）

# ── 3种缩放档位（积木长边占画布的比例）──────────────────────
# 0.65 = 小尺寸，0.80 = 标准尺寸，0.95 = 大尺寸
# 自适应机制保证缩放后积木不超出画布
ALL_SCALES = [0.65, 0.80, 0.95]
N_SCALE = 1   # 每张原图从3种中随机选1种

# ── 5种位置方向（相对于画布的偏移方向）──────────────────────
# 格式：(水平方向, 垂直方向)，含义见下面的注释
# 实际偏移量由自适应机制动态计算，保证积木不超出画布
ALL_POSITIONS = [
    "center",      # 居中
    "top_left",    # 左上
    "bottom_left", # 左下
    "top_right",   # 右上
    "bottom_right",# 右下
]
N_POS = 3   # 每张原图从5种中随机选3种

# 位置偏移强度（相对于可用空白空间的比例）
# 0.6 表示使用可移动空间的60%，保留40%作为安全边距
POS_RATIO = 0.6

# ── 形变参数 ──────────────────────────────────────────────────
# 水平翻转：随机二选一（翻 / 不翻）
# 仿射变换：生成2个变体，每次随机独立采样参数
AFFINE_MAX_SHEAR_DEG = 5.0    # 仿射最大倾斜角度（度）
AFFINE_MAX_SHIFT_RATIO = 0.03  # 仿射最大平移（画布尺寸的比例）

# 旋转：生成3个变体，每次在[-15, 15]°内随机采样
ROTATE_MAX_DEG = 15.0

# 白色背景去除阈值（原图中RGB三通道都高于此值视为背景）
BG_THRESHOLD = 240


# ============================================================
# 第一步：去除白色背景
# ============================================================
def remove_white_background(img, threshold=BG_THRESHOLD):
    """
    把原图的白色/浅色背景变为透明（RGBA格式）

    原理：对每个像素，若RGB三通道都超过阈值，认为是背景，
    将其Alpha通道设为0（完全透明），积木主体保留。

    参数：
        img       : PIL Image，RGB模式
        threshold : 白色判定阈值，默认240
    返回：
        RGBA模式的PIL Image，背景已透明
    """
    # 先确保是RGB格式，统一处理
    img_rgb = img.convert("RGB")
    arr = np.array(img_rgb, dtype=np.uint8)

    R = arr[:, :, 0]
    G = arr[:, :, 1]
    B = arr[:, :, 2]

    # 判断哪些像素是白色背景
    # 三通道都高于阈值 → 视为背景
    is_bg = (R > threshold) & (G > threshold) & (B > threshold)

    # 构建RGBA数组
    rgba = np.zeros((arr.shape[0], arr.shape[1], 4), dtype=np.uint8)
    rgba[:, :, :3] = arr        # 复制RGB
    rgba[:, :, 3] = 255         # 默认全不透明
    rgba[is_bg, 3] = 0          # 背景像素设为完全透明

    return Image.fromarray(rgba, mode="RGBA")


# ============================================================
# 第二步：自适应缩放 + 位置放置（全程RGBA透明图）
# ============================================================
def place_on_transparent_canvas(fg_rgba, scale, position):
    """
    把积木透明图按指定缩放和位置，放置到透明画布上

    关键：输入和输出都是RGBA透明图，还没有贴背景色！
    这样后续的仿射、旋转操作产生的空白 = 透明，而非白色。

    自适应机制：
        目标尺寸 = 画布边长 × scale
        若积木实际宽高比不是正方形，按长边缩放，保持比例
        偏移量 = 剩余空白空间 × POS_RATIO（保证不超界）

    参数：
        fg_rgba  : RGBA透明积木图
        scale    : 积木长边占画布的目标比例（如0.80）
        position : 位置字符串（"center"/"top_left"等）
    返回：
        CANVAS_SIZE × CANVAS_SIZE 的RGBA透明画布
    """
    # 创建透明画布（Alpha=0，全透明）
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))

    fg_w, fg_h = fg_rgba.size

    # 计算目标尺寸：按长边缩放到 CANVAS_SIZE × scale
    target_long = int(CANVAS_SIZE * scale)
    # 防止尺寸过小（最小40像素）
    target_long = max(target_long, 40)

    # 等比缩放：保持积木宽高比
    if fg_w >= fg_h:
        new_w = target_long
        new_h = max(int(fg_h * target_long / fg_w), 1)
    else:
        new_h = target_long
        new_w = max(int(fg_w * target_long / fg_h), 1)

    # 缩放积木（LANCZOS：高质量插值，适合缩小）
    fg_resized = fg_rgba.resize((new_w, new_h), Image.LANCZOS)

    # 计算居中位置（积木在画布正中间的左上角坐标）
    cx = (CANVAS_SIZE - new_w) // 2
    cy = (CANVAS_SIZE - new_h) // 2

    # 计算可偏移的最大像素数（自适应：基于实际剩余空间）
    # 剩余空白 = 居中时积木到边缘的距离
    max_dx = int(cx * POS_RATIO)   # 水平最大偏移量
    max_dy = int(cy * POS_RATIO)   # 垂直最大偏移量

    # 根据位置方向计算实际偏移量
    if position == "center":
        dx, dy = 0, 0
    elif position == "top_left":
        dx, dy = -max_dx, -max_dy
    elif position == "bottom_left":
        dx, dy = -max_dx, +max_dy
    elif position == "top_right":
        dx, dy = +max_dx, -max_dy
    elif position == "bottom_right":
        dx, dy = +max_dx, +max_dy
    else:
        dx, dy = 0, 0

    # 计算最终粘贴坐标
    paste_x = cx + dx
    paste_y = cy + dy

    # 将积木贴到透明画布上（用自身Alpha通道作蒙版）
    canvas.paste(fg_resized, (paste_x, paste_y), fg_resized)

    return canvas


# ============================================================
# 第三步：水平翻转（在透明图上操作）
# ============================================================
def random_flip(img_rgba):
    """
    随机决定是否水平翻转（二选一：翻 / 不翻）

    纯镜像操作，不产生任何空白区域，全程RGBA透明图。

    参数：
        img_rgba : RGBA透明图
    返回：
        翻转后（或未翻转）的RGBA透明图
    """
    if random.random() < 0.5:
        # FLIP_LEFT_RIGHT：水平镜像
        return img_rgba.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        return img_rgba.copy()


# ============================================================
# 第四步：小幅仿射变换（在透明图上操作，生成2个变体）
# ============================================================
def apply_affine(img_rgba):
    """
    对透明图做轻微仿射变换（倾斜+微平移）

    仿射变换说明：
        使用PIL的transform函数，仿射矩阵形式：
        [a, b, c]    a,d控制缩放/旋转，b,e控制倾斜，c,f控制平移
        [d, e, f]

        这里只做轻微倾斜（≤5°）+ 微小平移（≤3%画布），
        不做缩放，不破坏积木整体结构。

    空白区域处理：
        全程RGBA透明图，仿射后边缘空白 = 透明（Alpha=0）
        不会产生白色残留。

    参数：
        img_rgba : RGBA透明图（CANVAS_SIZE × CANVAS_SIZE）
    返回：
        仿射后的RGBA透明图
    """
    W, H = img_rgba.size  # 应该是 (224, 224)

    # 随机生成倾斜角度（弧度）
    # 水平倾斜 shear_x：让图像向左/右倾斜
    # 限制在 ±AFFINE_MAX_SHEAR_DEG 度内
    shear_x = math.tan(
        math.radians(random.uniform(-AFFINE_MAX_SHEAR_DEG,
                                     AFFINE_MAX_SHEAR_DEG))
    )
    shear_y = math.tan(
        math.radians(random.uniform(-AFFINE_MAX_SHEAR_DEG,
                                     AFFINE_MAX_SHEAR_DEG))
    )

    # 随机微小平移（像素数）
    max_shift = int(CANVAS_SIZE * AFFINE_MAX_SHIFT_RATIO)
    tx = random.randint(-max_shift, max_shift)
    ty = random.randint(-max_shift, max_shift)

    # PIL的transform仿射矩阵（逆变换格式）：
    # 输出像素(x,y) ← 输入像素(a*x + b*y + c, d*x + e*y + f)
    # 纯倾斜：a=1, b=shear_x, c=0, d=shear_y, e=1, f=0
    # 加平移：c=-tx, f=-ty（负号因为是逆变换）
    data = (
        1,       shear_x, -tx,
        shear_y, 1,       -ty
    )

    # 执行仿射变换
    # BICUBIC：双三次插值，质量好，适合几何变换
    # fillcolor=None → 边缘填充为透明（因为是RGBA图）
    result = img_rgba.transform(
        (W, H),
        Image.AFFINE,
        data,
        resample=Image.BICUBIC,
        fillcolor=None   # RGBA图透明填充
    )

    return result


# ============================================================
# 第五步：小角度旋转（在透明图上操作，生成3个变体）
# ============================================================
def apply_rotation(img_rgba):
    """
    对透明图做小角度随机旋转（±15°内）

    旋转后四角出现的空白区域 = 透明（因为是RGBA图）
    不会产生白色角落。

    参数：
        img_rgba : RGBA透明图
    返回：
        旋转后的RGBA透明图
    """
    # 在±15°内均匀随机采样旋转角度
    angle = random.uniform(-ROTATE_MAX_DEG, ROTATE_MAX_DEG)

    # PIL的rotate：逆时针旋转
    # expand=False：保持画布尺寸不变
    # fillcolor=None → 空白处填充透明（RGBA图）
    result = img_rgba.rotate(
        angle,
        resample=Image.BICUBIC,
        expand=False,
        fillcolor=None   # RGBA图透明填充
    )

    return result


# ============================================================
# 第六步：贴背景色，输出最终RGB图
# ============================================================
def paste_on_background(fg_rgba, bg_color):
    """
    把透明积木图贴到纯色背景上，输出最终RGB图

    这是流程的最后一步，此时fg_rgba中：
    - 积木像素：RGB有值，Alpha=255（不透明）
    - 空白区域：Alpha=0（透明）→ 会被背景色填充

    参数：
        fg_rgba  : RGBA透明图（含积木+透明空白区域）
        bg_color : 背景颜色 (R, G, B)
    返回：
        CANVAS_SIZE × CANVAS_SIZE 的RGB图
    """
    # 创建纯色背景
    background = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), bg_color)

    # 把透明图贴到背景上，透明区域自动填充为背景色
    # 第三个参数：用fg_rgba的Alpha通道作为蒙版
    background.paste(fg_rgba, (0, 0), fg_rgba)

    return background


# ============================================================
# 主流程：对单张原图生成36张增强变体
# ============================================================
def augment_one_image(img_path, output_dir, base_name):
    """
    对一张原始图生成完整的36张增强变体并保存

    36张的生成逻辑（理解A：串联叠加）：
    ┌─────────────────────────────────────────────────────┐
    │ 步骤0：去白背景 → RGBA透明积木图                    │
    │                                                     │
    │ 步骤1：确定基础框架参数（共6种组合）：               │
    │   背景色 2种（8选2）                                │
    │   缩放   1种（3选1）                                │
    │   位置   3种（5选3）                                │
    │   组合：2 × 1 × 3 = 6种基础配置                    │
    │                                                     │
    │ 对每种基础配置（6次循环）：                          │
    │   步骤2：缩放+位置 → 透明画布积木图（1张）           │
    │   步骤3：随机翻转（执行或不执行）→ 1张               │
    │   步骤4：仿射变换 × 2 → 2张仿射变体                 │
    │   步骤5：每张仿射变体 × 旋转3次 → 6张旋转变体       │
    │   步骤6：贴对应背景色 → 6张RGB最终图                │
    │                                                     │
    │ 总计：6个配置 × 6张 = 36张                          │
    └─────────────────────────────────────────────────────┘

    参数：
        img_path   : 原图绝对路径
        output_dir : 该类别的输出文件夹路径
        base_name  : 原图文件名（不含扩展名），用于命名输出文件
    """
    # ── 读取原图 ──────────────────────────────────────────
    original = Image.open(img_path).convert("RGB")

    # ── 步骤0：去除白色背景，转为RGBA透明图 ───────────────
    fg_rgba_raw = remove_white_background(original, BG_THRESHOLD)

    # ── 步骤1：随机抽取基础框架参数 ───────────────────────
    # 背景色：8选2（random.sample保证不重复）
    chosen_bgs = random.sample(ALL_BACKGROUNDS, N_BG)
    # 缩放：3选1
    chosen_scale = random.choice(ALL_SCALES)
    # 位置：5选3
    chosen_positions = random.sample(ALL_POSITIONS, N_POS)

    # 用计数器给输出文件编号（0~35）
    variant_idx = 0

    # ── 外层循环：遍历6种基础配置（背景×位置）─────────────
    # 注意：背景色信息记录在配置里，最后一步才用
    for bg_color in chosen_bgs:           # 2种背景
        for position in chosen_positions:  # 3种位置

            # ── 步骤2：缩放+位置 → 透明画布积木图 ─────────
            # 此时是RGBA透明图，背景色只是记录，还没有贴
            placed = place_on_transparent_canvas(
                fg_rgba_raw, chosen_scale, position
            )

            # ── 步骤3：随机水平翻转（×1，执行或不执行）────
            flipped = random_flip(placed)
            # flipped 仍是RGBA透明图

            # ── 步骤4：仿射变换 × 2 ────────────────────────
            # 对翻转后的图生成2个独立仿射变体
            affine_variants = []
            for _ in range(2):
                affine_img = apply_affine(flipped)
                # affine_img 仍是RGBA透明图（空白处=透明）
                affine_variants.append(affine_img)

            # ── 步骤5：旋转 × 3（对每个仿射变体）──────────
            for affine_img in affine_variants:   # 2张仿射图
                for _ in range(3):               # 各旋转3次
                    rotated = apply_rotation(affine_img)
                    # rotated 仍是RGBA透明图（旋转角落=透明）

                    # ── 步骤6：最后一步，贴背景色 ──────────
                    # 此刻才把透明区域填充为目标背景色
                    # 输出最终RGB图，干净无白色残留
                    final_img = paste_on_background(rotated, bg_color)

                    # 保存
                    out_name = f"{base_name}_v{variant_idx:03d}.png"
                    out_path = os.path.join(output_dir, out_name)
                    final_img.save(out_path)

                    variant_idx += 1

    # 每张原图应该恰好生成36张
    assert variant_idx == 36, \
        f"生成数量异常：{variant_idx}张（期望36张）"


# ============================================================
# 总控函数：遍历所有类别和所有原图
# ============================================================
def generate_all():
    """
    遍历 raw_data_shape/ 下所有类别，对每张原图调用增强函数
    """
    # 检查输入目录
    if not os.path.exists(RAW_ROOT):
        print(f"❌ 错误：原始数据目录不存在：{RAW_ROOT}")
        return

    # 获取所有类别文件夹，排序保证一致
    class_folders = sorted([
        d for d in os.listdir(RAW_ROOT)
        if os.path.isdir(os.path.join(RAW_ROOT, d))
    ])

    print(f"\n{'='*60}")
    print(f"  乐高形状 —— 第一轮离线数据增强（新版）")
    print(f"{'='*60}")
    print(f"  输入目录  ：{RAW_ROOT}")
    print(f"  输出目录  ：{OUTPUT_ROOT}")
    print(f"  类别数量  ：{len(class_folders)}")
    print(f"  每张原图  ：36张变体")
    print(f"  增强流程  ：去背景→缩放+位置→翻转→仿射×2→旋转×3→贴背景")
    print(f"  随机种子  ：{RANDOM_SEED}")
    print(f"{'='*60}\n")

    total_imgs   = 0   # 原图总数
    total_output = 0   # 输出总数

    for class_name in class_folders:
        class_input_dir  = os.path.join(RAW_ROOT,    class_name)
        class_output_dir = os.path.join(OUTPUT_ROOT, class_name)

        # 创建输出子文件夹（已存在则跳过）
        os.makedirs(class_output_dir, exist_ok=True)

        # 获取该类所有原始图片
        img_files = sorted([
            f for f in os.listdir(class_input_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        class_output_count = 0

        for img_file in img_files:
            img_path  = os.path.join(class_input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]

            augment_one_image(img_path, class_output_dir, base_name)
            class_output_count += 36

        total_imgs   += len(img_files)
        total_output += class_output_count

        print(f"  ✅ {class_name}：{len(img_files)} 张原图 "
              f"→ {class_output_count} 张增强图")

    print(f"\n{'='*60}")
    print(f"  全部完成！")
    print(f"  原始图片总数：{total_imgs} 张")
    print(f"  输出图片总数：{total_output} 张")
    print(f"  输出目录    ：{OUTPUT_ROOT}")
    print(f"{'='*60}\n")


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    generate_all()
