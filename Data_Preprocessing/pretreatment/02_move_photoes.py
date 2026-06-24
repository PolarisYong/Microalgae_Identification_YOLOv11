import os
import re
import shutil


def organize_images(source_dir, target_parent, channel_num):
    """
    整理图片文件：将同名图片放入同一文件夹，并按源文件夹序号重命名

    参数:
        source_dir: 包含99个子文件夹的源目录（HEIDSTAR.COM.x.y）
        target_parent: 目标父目录（CHz）
        channel_num: 通道编号（z值）
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
                # 确保目标文件夹存在
                os.makedirs(target_folder, exist_ok=True)

                source_path = os.path.join(image_dir, img_file)
                target_path = os.path.join(target_folder, new_filename)

                # 复制文件
                shutil.copy2(source_path, target_path)
                print(f"复制: {img_file} -> {target_folder}/{new_filename}")

    print("\n当前通道处理完成")


def batch_process_root_directory(root_dir):
    """
    批处理根目录下的所有HEIDSTAR.COM.x.y格式子文件夹

    参数:
        root_dir: 根目录（F:\Microalgae_Photoes\20260609）
    """
    # 正则匹配HEIDSTAR.COM.x.y格式的文件夹名，提取x和y
    folder_name_pattern = r'HEIDSTAR\.COM\.(\d+)\.(\d+)$'

    # 遍历根目录下的所有子文件夹
    for sub_folder in os.listdir(root_dir):
        sub_folder_path = os.path.join(root_dir, sub_folder)

        # 仅处理目录，且符合命名格式
        if not os.path.isdir(sub_folder_path):
            continue

        match = re.match(folder_name_pattern, sub_folder)
        if not match:
            print(f"\n跳过不符合HEIDSTAR.COM.x.y格式的文件夹: {sub_folder}")
            continue

        # 提取x和y并转换为整数
        x = int(match.group(1))
        y = int(match.group(2))

        # 计算对应的通道号z
        z = 3 * (x - 1) + y
        channel_num = str(z)

        # 构造目标父目录（CHz）
        target_parent = os.path.join(root_dir, f"CH{z}")

        print(f"\n=====================================")
        print(f"开始处理: {sub_folder} -> CH{z} (x={x}, y={y}, z={z})")
        print(f"=====================================")

        # 确保目标目录存在
        os.makedirs(target_parent, exist_ok=True)

        # 调用整理图片函数
        organize_images(sub_folder_path, target_parent, channel_num)

    print("\n所有文件夹批处理完成！")


if __name__ == "__main__":
    # 根目录（包含所有HEIDSTAR.COM.x.y子文件夹的目录）
    root_directory = r"F:\Microalgae_Photoes\20260531"

    # 验证根目录有效性
    if not os.path.exists(root_directory) or not os.path.isdir(root_directory):
        print(f"错误: 根目录不存在或不是有效的目录 - {root_directory}")
    else:
        batch_process_root_directory(root_directory)