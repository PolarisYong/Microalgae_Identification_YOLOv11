import os
import re
import shutil


def organize_images(source_dir, target_parent, channel_num):
    """
    整理图片文件：将同名图片放入同一文件夹，并按源文件夹序号重命名

    参数:
        source_dir: 包含99个子文件夹的源目录
        target_parent: 目标父目录，将在其中创建以图片名为基础的子文件夹
    """
    # 正则表达式匹配源文件夹名称，提取括号中的数字（如(001)中的001）
    folder_pattern = r'.*\((\d{3})\)$'

    # 遍历源目录中的所有子文件夹
    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)

        if os.path.isdir(folder_path):
            # 提取源文件夹编号（如001）
            match = re.match(folder_pattern, folder)
            if not match:
                print(f"跳过不符合命名规则的文件夹: {folder}")
                continue

            source_num = match.group(1).lstrip('0') or '0'
            print(f"\n处理文件夹: {folder} (编号: {source_num})")

            # 检查image子文件夹
            image_dir = os.path.join(folder_path, "images")
            if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
                print(f"警告: {folder}中未找到images文件夹，已跳过")
                continue

            # 处理image文件夹中的图片
            for img_file in os.listdir(image_dir):
                if not img_file.lower().endswith('.tif'):
                    continue
                pattern = r'IMG\d+x(\d{3})\.tif$'
                match = re.match(pattern, img_file)
                if match:
                    chamber_num = match.group(1).lstrip('0') or '0'
                else:
                    chamber_num = "error"
                # 构建新文件名
                new_filename = f"CH{channel_num}_CB{chamber_num}_H{source_num}.tif"
                # 构建目标路径
                folder_name = os.path.splitext(img_file)[0]
                target_folder = os.path.join(target_parent, folder_name)
                # 确保目标文件夹存在（关键修正：每次处理都检查并创建）
                os.makedirs(target_folder, exist_ok=True)

                source_path = os.path.join(image_dir, img_file)
                target_path = os.path.join(target_folder, new_filename)

                # 复制文件
                shutil.copy2(source_path, target_path)
                print(f"复制: {img_file} -> {target_folder}/{new_filename}")

    print("\n所有文件处理完成")


if __name__ == "__main__":
    # 源文件夹路径（包含99个子文件夹的目录）
    source_directory = r"F:\Microalgae_Photoes\20260520\样本[3]-L120-B-D-E.COM.3"  # 替换为实际源目录路径

    # 目标父目录（将在其中创建以图片名为基础的子文件夹）
    target_parent_directory = r"F:\Microalgae_Photoes\20260520\CH9"  # 替换为实际目标目录路径

    channel_num = "9"
    # 验证源文件夹
    if not os.path.exists(source_directory) or not os.path.isdir(source_directory):
        print(f"错误: 源文件夹不存在或不是有效的目录 - {source_directory}")
    else:
        # 创建目标父目录（如果不存在）
        os.makedirs(target_parent_directory, exist_ok=True)
        organize_images(source_directory, target_parent_directory, channel_num)
