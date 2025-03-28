import torch
import moviepy.editor as mp
import numpy as np

# 加载VGGish模型
model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()  # 设置为评估模式


# 从视频中提取音频并保存为.wav文件
def extract_audio_from_video(video_file, audio_file="audio.wav"):
    # 使用 moviepy 提取音频
    video = mp.VideoFileClip(video_file)
    video.audio.write_audiofile(audio_file)


# 使用VGGish提取音频特征
def extract_audio_features(audio_file):
    # 直接将音频文件路径传给模型的forward方法
    with torch.no_grad():
        features = model.forward(audio_file)

    return features.numpy()


# 保存音频特征到npy文件
def save_audio_features_to_npy(audio_file, output_npy_file="audio_features.npy"):
    # 提取音频特征
    audio_features = extract_audio_features(audio_file)
    # 将音频特征保存为npy文件
    np.save(output_npy_file, audio_features)
    print(f"音频特征已保存为 {output_npy_file}")


# 主程序
def process_audio_from_video(video_file, output_audio_file="extracted_audio.wav", output_npy_file="audio_features.npy"):
    # 从视频中提取音频
    extract_audio_from_video(video_file, output_audio_file)
    # 保存音频特征为npy文件
    save_audio_features_to_npy(output_audio_file, output_npy_file)


# 示例使用
video_file = r"video/1.mp4"  # 替换为你的视频路径
process_audio_from_video(video_file)
