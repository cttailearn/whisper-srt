# 字幕提取
import torch
# pip install faster-whisper
from faster_whisper import WhisperModel

import os
from tqdm import tqdm
import time
import pandas as pd
# pip install pysubs2
import pysubs2
from srt2ass import srt2ass

class Transcribe:
    def __init__(self,model_name="small",device='cuda',compute_type="auto") -> None:
        # 如果compute_type为auto，则智能选择计算类型
        if compute_type == "auto":
            if device == 'cuda' and torch.cuda.is_available():
                try:
                    # 测试GPU是否支持float16
                    test_tensor = torch.tensor([1.0], dtype=torch.float16, device='cuda')
                    _ = test_tensor * 2
                    compute_type = "float16"
                    print("[INFO] 自动选择float16计算类型")
                except Exception as e:
                    print(f"[WARNING] GPU不支持float16，使用float32: {e}")
                    compute_type = "float32"
            else:
                compute_type = "float32"
                print(f"[INFO] 自动选择float32计算类型")
        else:
            print(f"[INFO] 使用指定的计算类型: {compute_type}")
            
        print(f"[INFO] 初始化Whisper模型 - 模型: {model_name}, 设备: {device}, 计算类型: {compute_type}")
        self.model = WhisperModel(model_name,device=device,compute_type=compute_type)
        torch.cuda.empty_cache()

    def run(self,file_name,audio_binary_io = None,language='ja',
            task="transcribe",
            beam_size = 5,
            is_vad_filter=False,
            min_silence_duration_ms=500,
            is_split = False,
            split_method = "Modest",
            sub_style = "default",
            initial_prompt= None):
        '''
        beam_size：数值越高，在识别时探索的路径越多，这在一定范围内可以帮助提高识别准确性，但是相对的VRAM使用也会更高. 同时，Beam Size在超过5-10后有可能降低精确性，详情请见https://arxiv.org/pdf/2204.05424.pdf                                          
        is_vad_filter：使用VAD过滤。
            使用[Silero VAD model](https://github.com/snakers4/silero-vad)以检测并过滤音频中的无声段落（推荐小语种使用）
            【注意】使用VAD filter有优点亦有缺点，请用户自行根据音频内容决定是否启用. [关于VAD filter](https://github.com/Ayanaminn/N46Whisper/blob/main/FAQ.md)
        is_split：是否使用空格将文本分割成多行
            [True,False]
        split_method：分割方法
            普通分割（Modest)：当空格后的文本长度超过5个字符，则另起一行
            全部分割（Aggressive): 只要遇到空格即另起一行
        sub_style：字幕样式
            default
        initial_prompt: 使用提示词能够提高输出质量,详情见： https://platform.openai.com/docs/guides/speech-to-text/prompting
        '''
        audio_name = os.path.splitext(os.path.basename(file_name))[0]   

        # 如果没有传入音频的二进制，则认为是本地文件
        if audio_binary_io == None:
            if not os.path.exists(file_name):
                raise Exception("File not found")
            audio = file_name
        else:
            audio = audio_binary_io

        tic = time.time()

        print("transcribe param")
        print(f"audio: {audio}")
        print(f"language: {language}")
        print(f"is_vad_filter: {is_vad_filter}")
        print(f"beam_size: {beam_size}")
        print(f"initial_prompt: {initial_prompt}")

        if is_vad_filter == False:
            vad_parameters = None
        else:
            vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)
        
        try:
            print(f"[INFO] 开始转录音频，模型: {self.model}")
            segments, info = self.model.transcribe(audio = audio,
                                            task=task,
                                            beam_size=beam_size,
                                            language=language,
                                            vad_filter=is_vad_filter,
                                            vad_parameters=vad_parameters,
                                            initial_prompt = initial_prompt,
                                            word_timestamps=True,
                                            #condition_on_previous_text=False,
                                            #no_speech_threshold=0.6,
                                            )
            print(f"[INFO] 转录完成，音频时长: {info.duration:.2f}秒")
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] 字幕生成失败: {error_msg}")
            
            # 针对特定错误提供解决方案
            if "Invalid input features shape" in error_msg:
                print(f"[ERROR] 模型输入特征形状不匹配错误")
                print(f"[ERROR] 错误详情: {error_msg}")
                print(f"[SOLUTION] 可能的解决方案:")
                print(f"[SOLUTION] 1. 模型版本不兼容，请尝试使用标准模型名称如 'large-v3'")
                print(f"[SOLUTION] 2. 如果使用自定义模型，请确保模型是正确的CTranslate2格式")
                print(f"[SOLUTION] 3. 尝试重新转换模型: ct2-transformers-converter --model [原模型路径] --output_dir [输出路径]")
                print(f"[SOLUTION] 4. 检查faster-whisper和ctranslate2版本兼容性")
                print(f"[SOLUTION] 5. 尝试使用CPU设备而非GPU")
            elif "CUDA" in error_msg or "cuda" in error_msg:
                print(f"[ERROR] CUDA相关错误: {error_msg}")
                print(f"[SOLUTION] 1. 检查CUDA和cuDNN版本兼容性")
                print(f"[SOLUTION] 2. 尝试使用CPU设备")
                print(f"[SOLUTION] 3. 检查GPU内存是否足够")
            elif "Unable to open file" in error_msg or "model.bin" in error_msg:
                print(f"[ERROR] 模型文件错误: {error_msg}")
                print(f"[SOLUTION] 1. 检查模型文件是否完整")
                print(f"[SOLUTION] 2. 确保模型是CTranslate2格式")
                print(f"[SOLUTION] 3. 重新下载或转换模型")
            else:
                print(f"[ERROR] 未知错误: {error_msg}")
                print(f"[SOLUTION] 请检查音频文件格式和模型配置")
            
            # 重新抛出异常，让上层处理
            raise Exception(f"字幕生成失败: {error_msg}")

        results= []
        with tqdm(total=round(info.duration, 2), unit=" seconds") as pbar:
            for s in segments:
                segment_dict = {'start':s.start,'end':s.end,'text':s.text}
                results.append(segment_dict)
                segment_duration = round(s.end - s.start, 2)  
                pbar.update(segment_duration)
        toc = time.time()
        subs = pysubs2.load_from_whisper(results)
    
        # 保存srt文件
        srt_filename = os.path.join("./temp",audio_name + ".srt") 
        subs.save(srt_filename)
        print('生成srt：{} 识别耗时：{}'.format(srt_filename,toc-tic) )
        
        # 保存ass文件
        ass_filename  = srt2ass(srt_filename, sub_style, is_split,split_method)
        print('生成ass：{}'.format(ass_filename))
        return srt_filename,ass_filename

    def run_with_vad_splitting(self, file_name, audio_binary_io=None, language='ja',
                              task="transcribe", beam_size=5, is_vad_filter=False,
                              min_silence_duration_ms=500, is_split=False,
                              split_method="Modest", sub_style="default",
                              initial_prompt=None, max_workers=2,
                              max_segment_duration=30, min_segment_duration=5):
        """
        使用VAD分割和并发处理的音频转录方法
        参数:
        max_workers (int): 并发线程数
        max_segment_duration (int): 最大片段时长（秒）
        min_segment_duration (int): 最小片段时长（秒）
        其他参数与run方法相同
        """
        from utils import split_audio_by_vad, process_audio_segment_concurrent, merge_segment_results
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        audio_name = os.path.splitext(os.path.basename(file_name))[0]
        
        # 如果没有传入音频的二进制，则认为是本地文件
        if audio_binary_io == None:
            if not os.path.exists(file_name):
                raise Exception("File not found")
            audio = file_name
        else:
            audio = audio_binary_io
        
        tic = time.time()
        
        print("transcribe param (VAD splitting mode)")
        print(f"audio: {audio}")
        print(f"language: {language}")
        print(f"is_vad_filter: {is_vad_filter}")
        print(f"beam_size: {beam_size}")
        print(f"initial_prompt: {initial_prompt}")
        print(f"max_workers: {max_workers}")
        print(f"max_segment_duration: {max_segment_duration}s")
        print(f"min_segment_duration: {min_segment_duration}s")
        
        # 如果不启用VAD过滤，使用原始方法
        if not is_vad_filter:
            print("[INFO] VAD过滤未启用，使用标准处理方法")
            return self.run(file_name, audio_binary_io, language, task, beam_size,
                          is_vad_filter, min_silence_duration_ms, is_split,
                          split_method, sub_style, initial_prompt)
        
        try:
            # 第一步：分割音频
            print("[INFO] 开始分割音频...")
            segment_files = split_audio_by_vad(
                audio_path=audio,
                max_segment_duration=max_segment_duration,
                min_segment_duration=min_segment_duration,
                output_dir="./temp"
            )
            
            # 如果只有一个片段（即未分割），使用原始方法
            if len(segment_files) == 1 and segment_files[0] == audio:
                print("[INFO] 音频无需分割，使用标准处理方法")
                return self.run(file_name, audio_binary_io, language, task, beam_size,
                              is_vad_filter, min_silence_duration_ms, is_split,
                              split_method, sub_style, initial_prompt)
            
            # 第二步：并发处理各个片段
            print(f"[INFO] 开始并发处理 {len(segment_files)} 个音频片段...")
            
            # 设置VAD参数
            if is_vad_filter:
                vad_parameters = dict(min_silence_duration_ms=min_silence_duration_ms)
            else:
                vad_parameters = None
            
            # 使用线程池并发处理
            segment_results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # 提交所有任务
                future_to_segment = {
                    executor.submit(
                        process_audio_segment_concurrent,
                        self, segment_path, i, language, task,
                        beam_size, initial_prompt, vad_parameters
                    ): i for i, segment_path in enumerate(segment_files)
                }
                
                # 收集结果
                for future in as_completed(future_to_segment):
                    segment_index = future_to_segment[future]
                    try:
                        result = future.result()
                        segment_results.append(result)
                    except Exception as exc:
                        print(f"[ERROR] 片段 {segment_index + 1} 处理异常: {exc}")
                        segment_results.append((segment_index, [], None))
            
            # 第三步：合并结果
            print("[INFO] 开始合并处理结果...")
            srt_filename, ass_filename = merge_segment_results(
                segment_results=segment_results,
                segment_files=segment_files,
                output_dir="./temp",
                audio_name=audio_name
            )
            
            # 清理临时片段文件
            print("[INFO] 清理临时片段文件...")
            for segment_file in segment_files:
                if segment_file != audio and os.path.exists(segment_file):
                    try:
                        os.remove(segment_file)
                    except Exception as e:
                        print(f"[WARNING] 无法删除临时文件 {segment_file}: {e}")
            
            toc = time.time()
            print(f"[INFO] VAD分割并发处理完成，总耗时：{toc-tic:.2f}秒")
            
            return srt_filename, ass_filename
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] VAD分割并发处理失败: {error_msg}")
            
            # 如果分割处理失败，回退到原始方法
            print("[INFO] 回退到标准处理方法")
            return self.run(file_name, audio_binary_io, language, task, beam_size,
                          is_vad_filter, min_silence_duration_ms, is_split,
                          split_method, sub_style, initial_prompt)


if __name__ == "__main__":
    # 使用标准模型名称而不是本地路径
    test = Transcribe(model_name="large-v3", device="cuda")
    # 测试直接传入文件地址
    #test.run(file_name="./test.mp3")

    # 测试传入二进制
    with open('./file/2.wav', 'rb') as f:
        test.run(file_name="test",
                 audio_binary_io=f,
                 language="zh",
                 #initial_prompt="简体中文",
                 #is_vad_filter=True,
                 #is_split=False
        )





