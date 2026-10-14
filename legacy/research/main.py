import re

s = r'F:\Microalgae_Photoes\20251002\Processed\001_L100_20\IMG001x001'
pattern = r'Processed\\0(\d+)_L100_.*?IMG001x0(\d+)'
# 执行匹配
match = re.search(pattern, s)

if match:
    # 提取第一个1
    first_one = match.group(1)
    # 提取第二个1
    second_one = match.group(2)
    print(f"从001_L100_20中提取的1: {first_one}")
    print(f"从x001中提取的1: {second_one}")
else:
    print("未找到匹配的模式")