import os
import re

def parse_folder_name(folder_name):
    """解析文件夹名称，返回(基础名称, 原始编号)，不符合格式返回None"""
    # 匹配格式：基础名称(数字)，数字可以是1位及以上（如001、12、490、1028等）
    match = re.match(r'^(.*)\((\d+)\)$', folder_name)
    if match:
        base_name = match.group(1)
        try:
            original_num = int(match.group(2))  # 提取原始编号（整数）
            return base_name, original_num
        except ValueError:
            return None
    return None

def rename_folders(photoes_dir):
    # 检查目标文件夹是否存在
    if not os.path.exists(photoes_dir):
        print(f"错误：文件夹 {photoes_dir} 不存在")
        return

    # 1. 收集所有符合格式的文件夹，按基础名称分组
    folder_groups = {}  # 键：基础名称，值：列表[(文件夹路径, 原始编号)]
    for item in os.listdir(photoes_dir):
        item_path = os.path.join(photoes_dir, item)
        if os.path.isdir(item_path):  # 只处理文件夹
            parsed = parse_folder_name(item)
            if parsed:
                base_name, original_num = parsed
                # 加入对应分组
                if base_name not in folder_groups:
                    folder_groups[base_name] = []
                folder_groups[base_name].append((item_path, original_num))
            else:
                print(f"跳过不符合格式的文件夹：{item}")

    # 2. 对每个分组进行重新编号
    for base_name, folders in folder_groups.items():
        print(f"\n处理基础名称：{base_name}")
        # 按原始编号从小到大排序（确保原有顺序逻辑）
        folders_sorted = sorted(folders, key=lambda x: x[1])
        total = len(folders_sorted)
        print(f"找到 {total} 个文件夹，开始重新编号...")

        # 重新编号（从001开始）
        for index, (old_path, original_num) in enumerate(folders_sorted, start=1):
            # 新编号（3位数字，补零）
            new_num = f"{index:03d}"
            new_name = f"{base_name}({new_num})"
            new_path = os.path.join(photoes_dir, new_name)

            # 执行重命名
            if old_path != new_path:  # 避免不必要的操作
                os.rename(old_path, new_path)
                print(f"重命名：{os.path.basename(old_path)} → {new_name}")

    print("\n所有符合条件的文件夹已完成重新编号")

if __name__ == "__main__":
    # 请修改为你的Photoes文件夹实际路径
    photoes_directory = r"F:\Microalgae_Photoes\20251104\20251025-L100_300-400-500.COM.3"
    rename_folders(photoes_directory)