import os
import shutil
from tqdm import tqdm

def copy_images_with_renaming(src_dir, dest_dir):
    """
    将源文件夹下所有子文件夹的图片拷贝到目标文件夹中，重命名为原名称加上当前第几张图片。
    如果目标文件夹不存在，则创建。
    显示拷贝进度。
    
    :param src_dir: 源文件夹路径
    :param dest_dir: 目标文件夹路径
    """
    # 确保目标文件夹存在
    os.makedirs(dest_dir, exist_ok=True)

    # 遍历源文件夹下的所有文件及子文件夹
    all_images = []
    for root, _, files in os.walk(src_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 支持的图片格式
                all_images.append(os.path.join(root, file))

    # 复制文件并重命名
    for i, img_path in enumerate(tqdm(all_images, desc="Copying Images")):
        # 提取原文件名和扩展名
        file_name, ext = os.path.splitext(os.path.basename(img_path))
        # 生成新文件名：原文件名_序号.扩展名
        new_name = f"{file_name}_{i + 61}{ext}"
        dest_path = os.path.join(dest_dir, new_name)
        # 拷贝文件到目标路径
        shutil.copy(img_path, dest_path)

    print(f"拷贝完成！总共处理了 {len(all_images)} 张图片。")

# 示例使用
src_folder = r"D:\BaiduNetdiskDownload\kolektor缺陷数据集\kos07"  # 替换为你的源文件夹路径
dest_folder = r"F:\python\AI\DefectDetection\2025.1.5\Data\marbled\all"  # 替换为你的目标文件夹路径

copy_images_with_renaming(src_folder, dest_folder)