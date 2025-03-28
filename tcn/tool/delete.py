import os

# 视频文件名列表（需要删除对应的音频和标签文件）
video_files = [
    "0814.csv", "0841.csv", "0973.csv", "1041.csv", "1287.csv", "0711.csv"
]

# 音频和标签文件夹的路径
audio_folder = "/root/autodl-tmp/Audio_feature"  # 你音频文件夹的路径
label_folder = "/root/LMVD-main/label/label"  # 你标签文件夹的路径

# 遍历每个视频文件，删除相应的音频和标签文件
for video_file in video_files:
    # 获取对应的音频和标签文件名
    base_name = video_file.split(".")[0]  # 例如 "1041"
    audio_file = f"{base_name}.npy"  # 音频文件名
    label_file = f"{base_name}_Depression.csv"  # 标签文件名
    
    # 删除音频文件
    audio_file_path = os.path.join(audio_folder, audio_file)
    if os.path.exists(audio_file_path):
        os.remove(audio_file_path)
        print(f"Deleted audio file: {audio_file_path}")
    else:
        print(f"Audio file not found: {audio_file_path}")
    
    # 删除标签文件
    label_file_path = os.path.join(label_folder, label_file)
    if os.path.exists(label_file_path):
        os.remove(label_file_path)
        print(f"Deleted label file: {label_file_path}")
    else:
        print(f"Label file not found: {label_file_path}")

print("Deletion process completed.")
