import os
import shutil

# ====================== 【请你修改这两个路径！】 ======================
# 1. CH6文件夹的绝对路径（Windows示例：D:/Data/CH6，Mac/Linux示例：/home/user/Data/CH6）
SOURCE_ROOT_DIR = r"F:\Microalgae_Photoes\20260504\CH5"
# 2. 你要存放复制后图片的目标路径（自动创建不存在的文件夹）
TARGET_SAVE_DIR = r"F:\Microalgae_Photoes\20260504\overlay_05"
# =====================================================================

# 固定的图片文件名（无需修改）
IMAGE_FILE_NAME = "shared_bgfit_template_overlay.png"
# 子文件夹名称（无需修改）
RESULT_FOLDER = "results"


def copy_and_rename_images():
    # 自动创建目标文件夹（不存在则创建，存在不报错）
    os.makedirs(TARGET_SAVE_DIR, exist_ok=True)
    print(f"目标文件夹已就绪：{TARGET_SAVE_DIR}\n")

    # 循环遍历 1~50，生成对应的文件夹名
    for num in range(1, 51):
        # 格式化文件夹名：IMG001x001、IMG001x002 ... IMG001x050
        folder_name = f"IMG001x{num:03d}"

        # 拼接【源图片完整路径】
        source_image_path = os.path.join(
            SOURCE_ROOT_DIR,
            folder_name,
            RESULT_FOLDER,
            IMAGE_FILE_NAME
        )

        # 拼接【目标图片完整路径】（重命名为 文件夹名.png）
        target_image_name = f"{folder_name}.png"
        target_image_path = os.path.join(TARGET_SAVE_DIR, target_image_name)

        # 执行复制逻辑
        try:
            # 检查源文件是否存在
            if not os.path.isfile(source_image_path):
                print(f"❌ 未找到文件：{source_image_path}")
                continue

            # 复制文件（copy2保留文件元数据，比copy更完整）
            shutil.copy2(source_image_path, target_image_path)
            print(f"✅ 成功复制：{target_image_name}")

        except Exception as e:
            print(f"❌ 复制失败 {folder_name}，原因：{str(e)}")

    print("\n========== 任务执行完成 ==========")


if __name__ == "__main__":
    copy_and_rename_images()