import os
import shutil


def find_tif_in_subfolders(image_root, target_filename):
    """在图片根目录及其所有子目录中查找目标tif文件"""
    for root, dirs, files in os.walk(image_root):
        # 跳过隐藏文件夹
        if os.path.basename(root).startswith('.'):
            continue
        if target_filename in files:
            return os.path.join(root, target_filename)
    return None


def copy_matching_tif_files():
    # 配置文件夹路径
    json_folder = r"F:\Microalgae_Photoes\20251002\dataset_labelme\CH6"  # 存放json文件的文件夹
    image_root_folder = r"F:\Microalgae_Photoes\20251002\Processed\006_L100_500"  # 图片根文件夹（请替换为实际路径）
    photoes_folder = os.path.join(json_folder, "photoes")  # 目标存放文件夹

    # 创建photoes文件夹（如果不存在）
    os.makedirs(photoes_folder, exist_ok=True)
    print(f"图片将被复制到: {photoes_folder}")

    # 检查json文件夹是否存在
    if not os.path.exists(json_folder) or not os.path.isdir(json_folder):
        print(f"错误: 文件夹 '{json_folder}' 不存在或不是有效的目录")
        return

    # 检查图片根文件夹是否存在
    if not os.path.exists(image_root_folder) or not os.path.isdir(image_root_folder):
        print(f"错误: 图片根文件夹 '{image_root_folder}' 不存在或不是有效的目录")
        return

    # 获取所有json文件的基础名称（不含扩展名）
    json_files = [f for f in os.listdir(json_folder)
                  if f.lower().endswith('.json') and not f.startswith('.')]

    if not json_files:
        print(f"在 '{json_folder}' 中未找到任何json文件")
        return

    print(f"找到 {len(json_files)} 个json文件，开始查找对应的tif图片...\n")

    # 记录复制结果
    copied = []
    not_found = []

    # 遍历每个json文件，查找并复制对应的tif
    for json_file in json_files:
        # 获取基础名称（不含扩展名）
        base_name = os.path.splitext(json_file)[0]
        tif_filename = f"{base_name}.tif"

        # 在图片文件夹及其子文件夹中查找
        source_path = find_tif_in_subfolders(image_root_folder, tif_filename)

        if source_path:
            # 目标路径
            dest_path = os.path.join(photoes_folder, tif_filename)

            # 复制文件（保留元数据）
            shutil.copy2(source_path, dest_path)
            copied.append((tif_filename, os.path.dirname(source_path)))
            print(f"已复制: {tif_filename} (来自: {os.path.dirname(source_path)})")
        else:
            not_found.append(tif_filename)
            print(f"未找到: {tif_filename}")

    # 输出总结报告
    print("\n" + "=" * 50)
    print("复制结果总结")
    print("=" * 50)
    print(f"总处理json文件数: {len(json_files)}")
    print(f"成功复制tif文件数: {len(copied)}")
    print(f"未找到的tif文件数: {len(not_found)}")
    print(f"图片保存位置: {photoes_folder}")

    if not_found:
        print("\n未找到的tif文件列表:")
        for i, filename in enumerate(not_found, 1):
            print(f"  {i}. {filename}")


if __name__ == "__main__":
    copy_matching_tif_files()
    print("\n操作完成！")
