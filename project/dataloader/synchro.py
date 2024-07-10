#hello
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
from pydub import AudioSegment
import tempfile
import os

def extract_audio(video_path):
    video = VideoFileClip(video_path)
    audio_path = tempfile.mktemp(suffix=".wav")
    video.audio.write_audiofile(audio_path)
    return audio_path

def get_audio_features(audio_path):
    sample_rate, audio_data = wavfile.read(audio_path)
    if audio_data.ndim > 1:
        audio_data = audio_data.mean(axis=1)  # ステレオをモノラルに変換
    return sample_rate, audio_data

def calculate_offset(audio_data1, audio_data2):
    correlation = correlate(audio_data1, audio_data2, mode='full')
    offset = correlation.argmax() - (len(audio_data2) - 1)
    return offset

def sync_videos(video_path1, video_path2, output_path):
    # 音声を抽出
    audio_path1 = extract_audio(video_path1)
    audio_path2 = extract_audio(video_path2)

    # 音声特徴量を取得
    sample_rate1, audio_data1 = get_audio_features(audio_path1)
    sample_rate2, audio_data2 = get_audio_features(audio_path2)

    # 音声のオフセットを計算
    offset = calculate_offset(audio_data1, audio_data2)

    # 動画クリップを読み込む
    video1 = VideoFileClip(video_path1)
    video2 = VideoFileClip(video_path2)

    # オフセットに基づいて動画をトリム
    if offset > 0:
        video1 = video1.subclip(offset / sample_rate1)
    else:
        video2 = video2.subclip(-offset / sample_rate2)

    # 同期した動画を結
    final_clip = concatenate_videoclips([video1, video2])
    final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

    # 一時ファイルを削除
    os.remove(audio_path1)
    os.remove(audio_path2)

#TODO 動画ファイルのパス
video_path1 = "/home/chenkaixu/Walk_Video_PyTorch/data/raw_data/ASD/20160523_1/full_lat.mp4"
video_path2 = "v/home/chenkaixu/Walk_Video_PyTorch/data/raw_data/ASD/20160523_1/full_lat.mp4ideo2.mp4"
output_path = "sync/synced_video.mp4"

# 動画を同期
sync_videos(video_path1, video_path2, output_path)