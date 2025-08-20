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
    def __init__(self,model_name="small",device='cuda') -> None:
        # 智能选择计算类型，避免float16兼容性问题
        if device == 'cuda' and torch.cuda.is_available():
            try:
                # 测试GPU是否支持float16
                test_tensor = torch.tensor([1.0], dtype=torch.float16, device='cuda')
                _ = test_tensor * 2
                compute_type = "float16"
                print("使用float16计算类型")
            except Exception as e:
                print(f"GPU不支持float16，使用float32: {e}")
                compute_type = "float32"
        else:
            compute_type = "float32"
            
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

