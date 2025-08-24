
import ffmpeg
import os
import json
import librosa
import soundfile as sf
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def extract_audio(video_path, output_audio_path):
    """
    从视频文件中提取音频并保存为wav。
    参数:
    video_path (str): 视频文件的路径。
    output_audio_path (str): 输出音频文件的路径。
    """
    if not os.path.exists(video_path):
        raise "{} not find".format(video_path)
    if  os.path.exists(output_audio_path):
        os.remove(output_audio_path)
    try:
        (
            ffmpeg
            .input(video_path)
            .output(output_audio_path, acodec='mp3', audio_bitrate='320k')
            .run(overwrite_output=True)
        )
    except ffmpeg.Error as e:
        raise e

def merge_subtitles_to_video(video_path, subtitle_path, output_video_path):
    """
    将字幕文件合并到视频文件中。
    参数:
    video_path (str): 视频文件的路径。
    subtitle_path (str): 字幕文件的路径。
    output_video_path (str): 合并字幕后的输出视频文件的路径。
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"{video_path} not found")
    if not os.path.exists(subtitle_path):
        raise FileNotFoundError(f"{subtitle_path} not found")
    if os.path.exists(output_video_path):
        os.remove(output_video_path)
    
    subtitle_path = subtitle_path.replace("\\", "/")
    print("subtitle_path = {}".format(subtitle_path))
    
    # 构建支持中文字体的字幕滤镜
    # 转义冒号和特殊字符
    escaped_subtitle_path = subtitle_path.replace(":", "\\:")
    
    # 定义常见的中文字体列表（按优先级排序）
    chinese_fonts = [
        "Microsoft YaHei",  # 微软雅黑
        "SimHei",           # 黑体
        "SimSun",           # 宋体
        "NotoSansCJK",      # 思源黑体
        "WenQuanYi Micro Hei",  # 文泉驿微米黑
        "DejaVu Sans",      # DejaVu Sans (备用)
        "Arial Unicode MS", # Arial Unicode MS
        "Noto Sans CJK SC", # 思源黑体简体中文
    ]
    
    # 构建字体配置字符串
    font_config = ":".join(chinese_fonts)
    
    try:
        # 使用subtitles滤镜并指定字体
        vf_filter = f"subtitles={escaped_subtitle_path}:force_style='FontName={chinese_fonts[0]}'"
        
        print(f"[INFO] 使用字幕滤镜: {vf_filter}")
        
        (
            ffmpeg
            .input(video_path)
            .output(output_video_path, vf=vf_filter)
            .run(overwrite_output=True)
        )
        
        print(f"[INFO] 字幕合成成功: {output_video_path}")
        
    except ffmpeg.Error as e:
        print(f"[WARNING] 使用指定字体失败，尝试使用默认配置: {e}")
        
        # 如果指定字体失败，尝试使用默认配置
        try:
            (
                ffmpeg
                .input(video_path)
                .output(output_video_path, vf=f"subtitles={escaped_subtitle_path}")
                .run(overwrite_output=True)
            )
            print(f"[INFO] 使用默认配置字幕合成成功: {output_video_path}")
        except ffmpeg.Error as e2:
            print(f"[ERROR] 字幕合成失败: {e2}")
            raise RuntimeError(f"Failed to merge subtitles into video: {e2}")

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)
    print("清空文件夹：{}".format(folder_path))

def split_audio_by_vad(audio_path, max_segment_duration=30, min_segment_duration=5, output_dir="./temp"):
    """
    使用VAD将音频分割成适合字幕的片段
    参数:
    audio_path (str): 输入音频文件路径
    max_segment_duration (int): 最大片段时长（秒）
    min_segment_duration (int): 最小片段时长（秒）
    output_dir (str): 输出目录
    
    返回:
    list: 分割后的音频文件路径列表
    """
    try:
        # 加载音频
        audio, sr = librosa.load(audio_path, sr=16000)
        audio_duration = len(audio) / sr
        
        print(f"[INFO] 音频时长: {audio_duration:.2f}秒")
        
        # 如果音频时长小于最大片段时长，直接返回原文件
        if audio_duration <= max_segment_duration:
            print(f"[INFO] 音频时长小于{max_segment_duration}秒，无需分割")
            return [audio_path]
        
        # 使用librosa进行简单的静音检测
        # 计算音频的RMS能量
        frame_length = 2048
        hop_length = 512
        rms = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 设置静音阈值（可以根据需要调整）
        silence_threshold = np.percentile(rms, 20)  # 使用20%分位数作为静音阈值
        
        # 找到静音段
        silence_frames = rms < silence_threshold
        
        # 将帧索引转换为时间
        times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # 找到静音段的开始和结束时间
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_silence:
                silence_start = times[i]
                in_silence = True
            elif not is_silent and in_silence:
                silence_end = times[i]
                if silence_end - silence_start > 0.5:  # 静音段至少0.5秒
                    silence_segments.append((silence_start, silence_end))
                in_silence = False
        
        # 生成分割点
        split_points = [0]
        current_segment_start = 0
        
        for silence_start, silence_end in silence_segments:
            segment_duration = silence_start - current_segment_start
            
            # 如果当前片段超过最大时长，在此处分割
            if segment_duration >= max_segment_duration:
                split_points.append(silence_start)
                current_segment_start = silence_start
            # 如果当前片段超过最小时长且接近最大时长，也可以分割
            elif segment_duration >= min_segment_duration and segment_duration >= max_segment_duration * 0.8:
                split_points.append(silence_start)
                current_segment_start = silence_start
        
        # 添加音频结束点
        split_points.append(audio_duration)
        
        # 确保分割点合理
        filtered_split_points = [split_points[0]]
        for i in range(1, len(split_points)):
            if split_points[i] - filtered_split_points[-1] >= min_segment_duration:
                filtered_split_points.append(split_points[i])
            else:
                # 如果片段太短，合并到前一个片段
                filtered_split_points[-1] = split_points[i]
        
        split_points = filtered_split_points
        
        print(f"[INFO] 将音频分割为{len(split_points)-1}个片段")
        
        # 分割音频并保存
        segment_files = []
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        
        for i in range(len(split_points) - 1):
            start_time = split_points[i]
            end_time = split_points[i + 1]
            
            # 计算样本索引
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 提取音频片段
            segment_audio = audio[start_sample:end_sample]
            
            # 保存片段
            segment_filename = f"{base_name}_segment_{i+1:03d}.wav"
            segment_path = os.path.join(output_dir, segment_filename)
            
            sf.write(segment_path, segment_audio, sr)
            segment_files.append(segment_path)
            
            print(f"[INFO] 保存片段 {i+1}: {start_time:.2f}s - {end_time:.2f}s -> {segment_filename}")
        
        return segment_files
        
    except Exception as e:
        print(f"[ERROR] 音频分割失败: {str(e)}")
        # 如果分割失败，返回原文件
        return [audio_path]

def process_audio_segment_concurrent(transcribe_instance, segment_path, segment_index, language, task, beam_size, initial_prompt, vad_parameters):
    """
    并发处理单个音频片段
    参数:
    transcribe_instance: Transcribe实例
    segment_path (str): 音频片段路径
    segment_index (int): 片段索引
    language (str): 语言代码
    task (str): 任务类型
    beam_size (int): beam size
    initial_prompt (str): 初始提示词
    vad_parameters (dict): VAD参数
    
    返回:
    tuple: (segment_index, segments_list, segment_info)
    """
    try:
        print(f"[INFO] 开始处理片段 {segment_index + 1}: {os.path.basename(segment_path)}")
        
        # 为每个线程创建独立的模型实例（避免并发冲突）
        segments, info = transcribe_instance.model.transcribe(
            audio=segment_path,
            task=task,
            beam_size=beam_size,
            language=language,
            vad_filter=vad_parameters is not None,
            vad_parameters=vad_parameters,
            initial_prompt=initial_prompt,
            word_timestamps=True
        )
        
        # 将segments转换为列表（因为它是生成器）
        segments_list = list(segments)
        
        print(f"[INFO] 片段 {segment_index + 1} 处理完成，识别到 {len(segments_list)} 个片段")
        
        return (segment_index, segments_list, info)
        
    except Exception as e:
        print(f"[ERROR] 处理片段 {segment_index + 1} 失败: {str(e)}")
        return (segment_index, [], None)

def merge_segment_results(segment_results, segment_files, output_dir="./temp", audio_name="merged"):
    """
    合并分割处理的结果
    参数:
    segment_results (list): 各片段的处理结果
    segment_files (list): 音频片段文件列表
    output_dir (str): 输出目录
    audio_name (str): 输出文件名前缀
    
    返回:
    tuple: (srt_filename, ass_filename)
    """
    try:
        import pysubs2
        from srt2ass import srt2ass
        
        # 按片段索引排序
        segment_results.sort(key=lambda x: x[0])
        
        # 计算每个片段的时间偏移
        time_offsets = [0]
        
        # 使用librosa获取每个片段的实际时长
        for i, segment_file in enumerate(segment_files[:-1]):
            try:
                import librosa
                audio, sr = librosa.load(segment_file, sr=None)
                duration = len(audio) / sr
                time_offsets.append(time_offsets[-1] + duration)
            except Exception as e:
                print(f"[WARNING] 无法获取片段 {i+1} 的时长，使用估算值: {e}")
                # 如果无法获取时长，使用前一个片段的最后时间戳
                if i > 0 and segment_results[i-1][1]:
                    last_segment = segment_results[i-1][1][-1]
                    time_offsets.append(time_offsets[-1] + (last_segment.end - last_segment.start))
                else:
                    time_offsets.append(time_offsets[-1] + 30)  # 默认30秒
        
        # 合并所有片段的结果
        all_segments = []
        
        for segment_index, segments_list, info in segment_results:
            if segments_list:
                time_offset = time_offsets[segment_index]
                
                for segment in segments_list:
                    # 调整时间戳
                    adjusted_segment = {
                        'start': segment.start + time_offset,
                        'end': segment.end + time_offset,
                        'text': segment.text
                    }
                    all_segments.append(adjusted_segment)
        
        print(f"[INFO] 合并完成，总共 {len(all_segments)} 个字幕片段")
        
        # 创建字幕对象
        subs = pysubs2.load_from_whisper(all_segments)
        
        # 保存srt文件
        srt_filename = os.path.join(output_dir, f"{audio_name}.srt")
        subs.save(srt_filename)
        print(f"[INFO] 生成srt：{srt_filename}")
        
        # 保存ass文件
        ass_filename = srt2ass(srt_filename, "default", False, "Modest")
        print(f"[INFO] 生成ass：{ass_filename}")
        
        return srt_filename, ass_filename
        
    except Exception as e:
        print(f"[ERROR] 合并结果失败: {str(e)}")
        raise e

def import_config_file(file):
    if file is not None:
        content = file.read()
        try:
            json_data = json.loads(content)
            return json_data  
        except Exception as e:
            raise e
            
if __name__ == "__main__":
    pass
