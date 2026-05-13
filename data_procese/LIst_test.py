# import torch
# dim = 48
# order = 5
# dims = [dim // 2 ** (i) for i in range(order)]
# dims.reverse()
# print(dims)
# print(dims[0]+sum(dims))
# x = torch.randn(1,dim*2,4,4,4)
# y, x = torch.split(x, (dims[0], sum(dims)), dim=1)
# print(y.shape,x.shape)
import os
import shutil

# 源目录和目标基础目录
source_dir = r"E:\KiTS19"
target_base_dir = r"D:\Project\multi-organ seg\all_data\kits19\kits19"

# 获取源目录下所有master_*.nii.gz文件
master_files = [f for f in os.listdir(source_dir) if f.startswith("master_") and f.endswith(".nii.gz")]

# 遍历每个master文件
for master_file in master_files:
    # 提取编号（例如从"master_00000.nii.gz"中提取"00000"）
    try:
        case_id = master_file.split("_")[1].split(".")[0]
    except IndexError:
        print(f"文件名格式错误: {master_file}")
        continue

    # 构建目标文件夹路径
    target_case_dir = os.path.join(target_base_dir, f"case_{case_id}")

    # 检查目标文件夹是否存在
    if not os.path.exists(target_case_dir):
        print(f"目标文件夹不存在: {target_case_dir}")
        continue

    # 构建源文件和目标文件完整路径
    source_path = os.path.join(source_dir, master_file)
    target_path = os.path.join(target_case_dir, master_file)

    # 移动文件
    try:
        shutil.move(source_path, target_path)
        print(f"已移动: {master_file} -> {target_case_dir}")
    except Exception as e:
        print(f"移动文件失败 {master_file}: {str(e)}")

print("操作完成！")