# ============================================================
# split_dataset_shape.py
# 乐高形状识别 —— 训练集 / 测试集划分脚本
#
# 输入：layer_one_shape/（45个子文件夹，共约13212张增强图）
# 输出：
#   train_data_shape/（训练集，约5/6）
#   test/test_dataset_shape/（测试集，约1/6）
#
# 划分逻辑：
#   对每个类别，将该类所有图片随机打乱
#   取前 1/6（向下取整）作为测试集，其余全部作为训练集
#   不严格要求整除，零头全部归入训练集
#   按文件复制方式输出，不移动/删除原文件
#
# 随机种子固定为42，保证可复现
#
# 运行方式（在项目根目录下）：
#   python split_dataset_shape.py
# ============================================================

import os
import random
import shutil

# ============================================================
# 路径配置
# ============================================================
SOURCE_ROOT = r"C:\Users\lsbt\Desktop\lego_part_classifier\layer_one_shape"
TRAIN_ROOT  = r"C:\Users\lsbt\Desktop\lego_part_classifier\train_data_shape"
TEST_ROOT   = r"C:\Users\lsbt\Desktop\lego_part_classifier\test\test_dataset_shape"

# 随机种子，固定保证可复现
RANDOM_SEED = 42

# 测试集比例：约1/6
TEST_RATIO = 1 / 6


# ============================================================
# 主函数
# ============================================================
def split_dataset():

    random.seed(RANDOM_SEED)

    # 检查输入目录
    if not os.path.exists(SOURCE_ROOT):
        print(f"❌ 错误：输入目录不存在：{SOURCE_ROOT}")
        return

    # 获取所有类别文件夹，排序保证一致
    class_folders = sorted([
        d for d in os.listdir(SOURCE_ROOT)
        if os.path.isdir(os.path.join(SOURCE_ROOT, d))
    ])

    print(f"\n{'='*60}")
    print(f"  乐高形状 —— 训练集 / 测试集划分")
    print(f"{'='*60}")
    print(f"  输入目录  ：{SOURCE_ROOT}")
    print(f"  训练集    ：{TRAIN_ROOT}")
    print(f"  测试集    ：{TEST_ROOT}")
    print(f"  划分比例  ：训练 5/6，测试 1/6（非严格整除，零头归训练）")
    print(f"  随机种子  ：{RANDOM_SEED}")
    print(f"  类别数量  ：{len(class_folders)}")
    print(f"{'='*60}\n")

    total_train = 0
    total_test  = 0

    for class_name in class_folders:

        src_class_dir   = os.path.join(SOURCE_ROOT, class_name)
        train_class_dir = os.path.join(TRAIN_ROOT,  class_name)
        test_class_dir  = os.path.join(TEST_ROOT,   class_name)

        # 创建输出子文件夹
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(test_class_dir,  exist_ok=True)

        # 获取该类所有图片，排序后打乱（排序保证不同系统下顺序一致）
        all_files = sorted([
            f for f in os.listdir(src_class_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        # 随机打乱
        random.shuffle(all_files)

        # 计算测试集数量（向下取整，零头归训练集）
        n_total = len(all_files)
        n_test  = int(n_total * TEST_RATIO)   # 向下取整
        n_train = n_total - n_test            # 零头全归训练集

        # 划分文件列表
        test_files  = all_files[:n_test]
        train_files = all_files[n_test:]

        # 复制测试集文件
        for fname in test_files:
            src  = os.path.join(src_class_dir,  fname)
            dst  = os.path.join(test_class_dir, fname)
            shutil.copy2(src, dst)

        # 复制训练集文件
        for fname in train_files:
            src  = os.path.join(src_class_dir,   fname)
            dst  = os.path.join(train_class_dir, fname)
            shutil.copy2(src, dst)

        total_train += n_train
        total_test  += n_test

        print(f"  ✅ {class_name}：共{n_total:4d}张 "
              f"→ 训练{n_train:4d}张 / 测试{n_test:3d}张")

    print(f"\n{'='*60}")
    print(f"  划分完成！")
    print(f"  训练集总量：{total_train} 张")
    print(f"  测试集总量：{total_test}  张")
    print(f"  实际比例  ：{total_train/(total_train+total_test)*100:.1f}%"
          f" : {total_test/(total_train+total_test)*100:.1f}%")
    print(f"{'='*60}\n")


# ============================================================
# 程序入口
# ============================================================
if __name__ == "__main__":
    split_dataset()
