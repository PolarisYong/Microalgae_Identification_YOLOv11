#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

def rename_folders(target_dir):
    """
    批量重命名指定文件夹下的子文件夹
    替换规则：将名称中的 HEIDSTAR.COM.1.2 替换为 样本[1]-L120-C-C-B.COM.2
    其余部分保持不变
    """
    # 1. 检查目标文件夹是否存在
    if not os.path.isdir(target_dir):
        print(f"错误：目标文件夹不存在 → {target_dir}")
        return

    # 2. 定义替换规则（旧文本 → 新文本）
    OLD_TEXT = "HEIDSTAR.COM.3.3"
    NEW_TEXT = "样本[3]-L120-B-D-E.COM.3"

    # 3. 遍历目标文件夹下的所有项目
    for item_name in os.listdir(target_dir):
        item_path = os.path.join(target_dir, item_name)

        # 只处理【子文件夹】，跳过文件
        if not os.path.isdir(item_path):
            continue

        # 判断文件夹名称是否包含需要替换的文本
        if OLD_TEXT in item_name:
            # 生成新名称：仅替换指定文本，其余部分保留
            new_name = item_name.replace(OLD_TEXT, NEW_TEXT)
            new_path = os.path.join(target_dir, new_name)

            try:
                # 执行重命名
                os.rename(item_path, new_path)
                print(f"✅ 重命名成功：{item_name} → {new_name}")
            except Exception as e:
                print(f"❌ 重命名失败：{item_name}，原因：{str(e)}")

    print("\n=== 批量替换完成 ===")

if __name__ == "__main__":
    # ========== 请在这里修改为你的目标文件夹路径 ==========
    TARGET_FOLDER = r"F:\Microalgae_Photoes\20260520\2026-Week-22"
    # ======================================================

    rename_folders(TARGET_FOLDER)