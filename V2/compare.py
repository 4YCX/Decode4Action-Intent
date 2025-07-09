import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# 修改为你本地的 EEG 根目录路径
eeg_root = r"C:\Users\肖长屹Charlie\Desktop\EFDataset"

# # 获取所有手势文件夹名称（0, 1, ..., ok, left, ...）
# gesture_folders = [name for name in os.listdir(eeg_root) if os.path.isdir(os.path.join(eeg_root, name))]

# # 随机选取一个手势文件夹
# gesture = random.choice(gesture_folders)
# gesture_path = os.path.join(eeg_root, gesture)

# # 获取该手势下所有CSV文件
# all_csvs = [f for f in os.listdir(gesture_path) if f.endswith(".csv")]
# sample_files = random.sample(all_csvs, min(5, len(all_csvs)))

# # 读取数据并绘制通道波形
# plt.figure(figsize=(14, 6))

# # # 通道1绘图
# # plt.subplot(1, 2, 1)
# # for fname in sample_files:
# #     path = os.path.join(gesture_path, fname)
# #     df = pd.read_csv(path)
# #     ch1 = df.iloc[:, 0].values[:2000]  # 第1列为channel1
# #     plt.plot(ch1, label=fname)
# # plt.title(f"Channel 1 - Gesture: {gesture}")
# # plt.xlabel("Time")
# # plt.ylabel("Amplitude")
# # plt.grid(True)
# # plt.legend(fontsize=6)

# # # 通道2绘图
# # plt.subplot(1, 2, 2)
# # for fname in sample_files:
# #     path = os.path.join(gesture_path, fname)
# #     df = pd.read_csv(path)
# #     ch2 = df.iloc[:, 1].values[:2000]  # 第2列为channel2
# #     plt.plot(ch2, label=fname)
# # plt.title(f"Channel 2 - Gesture: {gesture}")
# # plt.xlabel("Time")
# # plt.ylabel("Amplitude")
# # plt.grid(True)
# # plt.legend(fontsize=6)

# # plt.tight_layout()
# # plt.show()


# # 通道1绘图
# plt.subplot(1, 2, 1)
# for fname in sample_files:
#     path = os.path.join(gesture_path, fname)
#     df = pd.read_csv(path)
#     if df.shape[1] < 2:
#         continue
#     ch1 = df.iloc[:, 0].values[:2000]
#     if len(ch1) == 0:
#         continue
#     plt.plot(ch1, label=os.path.splitext(fname)[0])
# plt.title(f"Channel 1 - Gesture: {gesture}")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend(fontsize=6)

# # 通道2绘图
# plt.subplot(1, 2, 2)
# for fname in sample_files:
#     path = os.path.join(gesture_path, fname)
#     df = pd.read_csv(path)
#     if df.shape[1] < 2:
#         continue
#     ch2 = df.iloc[:, 1].values[:2000]
#     if len(ch2) == 0:
#         continue
#     plt.plot(ch2, label=os.path.splitext(fname)[0])
# plt.title(f"Channel 2 - Gesture: {gesture}")
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.grid(True)
# plt.legend(fontsize=6)


import os
import random
import pandas as pd
import matplotlib.pyplot as plt

# 配置路径
## 文件夹路径说明: 主文件夹->EEG以及Freq分文件夹->手势文件夹->CSV文件

base_dir =  r"C:\Users\肖长屹Charlie\Desktop\EFDataset" 
eeg_dir = os.path.join(base_dir, "EEG")
gesture_list = os.listdir(eeg_dir)
gesture_list = [g for g in gesture_list if os.path.isdir(os.path.join(eeg_dir, g))]

# 随机选一个手势
gesture = random.choice(gesture_list)
gesture_path = os.path.join(eeg_dir, gesture)

# 随机选CSV文件
all_files = [f for f in os.listdir(gesture_path) if f.endswith(".csv")]
sample_files = random.sample(all_files, min(5, len(all_files)))

# 准备绘图
plt.figure(figsize=(12, 5))

# 通道1绘图
plt.subplot(1, 2, 1)
for fname in sample_files:
    path = os.path.join(gesture_path, fname)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        continue
    ch1 = df.iloc[:, 0].values[:2000]
    t = list(range(len(ch1)))
    plt.plot(t, ch1, label=fname)

plt.title(f"Channel 1 - Gesture: {gesture}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend(fontsize=6)

# 通道2绘图
plt.subplot(1, 2, 2)
for fname in sample_files:
    path = os.path.join(gesture_path, fname)
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        continue
    ch2 = df.iloc[:, 1].values[:2000]
    ### 进行曲线长点连接
    t = list(range(len(ch2)))
    plt.plot(t, ch2, label=fname)
    
plt.title(f"Channel 2 - Gesture: {gesture}")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.legend(fontsize=6)



plt.tight_layout()
plt.show()
