import moviepy.editor as mp
import librosa
import numpy as np

def extract_audio(file_path):
    video = mp.VideoFileClip(file_path)
    audio = video.audio
    audio_path = "temp_audio.wav"
    audio.write_audiofile(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    return y, sr

def compute_chroma(y, sr):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    return chroma

def compute_similarity(chroma1, chroma2, sr):
    cross_similarity = librosa.segment.cross_similarity(chroma1, chroma2)
    path, _ = librosa.sequence.dtw(C=1-cross_similarity, backtrack=True)
    start_time_video1 = path[0, 0] * 512 / sr
    start_time_video2 = path[0, 1] * 512 / sr
    print("start_time_video1:", start_time_video1)
    print("start_time_video2:", start_time_video2)
    return start_time_video1, start_time_video2, path

def sync_videos(video1_path, video2_path, output1_path, output2_path, output_mix_path):
    y1, sr1 = extract_audio(video1_path)
    y2, sr2 = extract_audio(video2_path)

    chroma1 = compute_chroma(y1, sr1)
    chroma2 = compute_chroma(y2, sr2)

    start_time_video1, start_time_video2, path = compute_similarity(chroma1, chroma2, sr1)

    video1 = mp.VideoFileClip(video1_path)
    video2 = mp.VideoFileClip(video2_path)

    # 動画の共通部分の期間を計算
    common_start_time = max(start_time_video1, start_time_video2)
    end_time_video1 = common_start_time + (video1.duration - (common_start_time - start_time_video1))
    end_time_video2 = common_start_time + (video2.duration - (common_start_time - start_time_video2))
    common_end_time = min(end_time_video1, end_time_video2)

    if common_start_time >= video1.duration or common_start_time >= video2.duration:
        raise ValueError("同期する部分が見つかりません")

    video1 = video1.subclip(common_start_time - start_time_video1, common_end_time - start_time_video1)
    video2 = video2.subclip(common_start_time - start_time_video2, common_end_time - start_time_video2)

    # 動画を上下に並べて新しい動画を作成
    final_clip = mp.clips_array([[video1], [video2]])

    video1.write_videofile(output1_path, codec="libx264")
    video2.write_videofile(output2_path, codec="libx264")
    final_clip.write_videofile(output_mix_path, codec="libx264")

# 動画のパスを指定して関数を呼び出します
sync_videos("/home/tsukumo/Walk_Video_PyTorch/data/ASD/20210518_1/full_ap.mp4", 
            "/home/tsukumo/Walk_Video_PyTorch/data/ASD/20210518_1/full_lat.mp4", 
            "/home/tsukumo/Walk_Video_PyTorch/data/sync/test_ap.mp4", 
            "/home/tsukumo/Walk_Video_PyTorch/data/sync/test_lat.mp4", 
            "/home/tsukumo/Walk_Video_PyTorch/data/sync/test_mix.mp4")

