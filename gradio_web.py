import gradio as gr
import json
import os
import traceback
from datetime import datetime
from transcribe import Transcribe
from zipfile import ZipFile
import base64
import io
import ffmpeg
from translation import GPT, Baidu, Tencent, translation
from utils import extract_audio, merge_subtitles_to_video, clear_folder, import_config_file
from uvr import UVR_Client
import torch

# 临时文件存放地址
TEMP = "./temp"

# 全局变量存储状态
class AppState:
    def __init__(self):
        self.transcribe = None
        self.audio_temp = None
        self.video_temp = None
        self.video_temp_name = None
        self.audio_separator_temp = None
        self.uvr_client = None
        self.engine = None
        
        # 模型配置
        self.model_list = ["tiny", "base", "small", "medium", "large-v2", "large-v3",
                          "tiny.en", "base.en", "medium.en", "small.en"]
        self.model_name = "large-v3"# 设备配置
        self.device_name = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if torch.cuda.is_available() else "float32"
        
        # 翻译配置
        self.chat_url = "https://api.openai.com/v1"
        self.chat_key = ""
        self.chat_model_list = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        self.chat_model_name = "gpt-4-turbo"
        self.baidu_appid = ""
        self.baidu_appkey = ""
        self.tencent_appid = ""
        self.tencent_secretKey = ""

app_state = AppState()



def load_model(model_path, device_name, compute_type="auto"):
    """加载转录模型"""
    try:
        if app_state.transcribe is not None:
            del app_state.transcribe
        
        # 优化设备和计算类型选择
        if device_name == "auto":
            if torch.cuda.is_available():
                device_name = "cuda"
                print(f"[INFO] 检测到CUDA可用，自动选择GPU设备")
                print(f"[INFO] GPU信息: {torch.cuda.get_device_name(0)}")
                print(f"[INFO] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            else:
                device_name = "cpu"
                print(f"[INFO] CUDA不可用，使用CPU设备")
        
        # 优化计算类型选择
        if compute_type == "auto":
            if device_name == "cuda":
                # 检查GPU是否支持float16
                try:
                    if torch.cuda.is_available():
                        # 检查GPU计算能力
                        capability = torch.cuda.get_device_capability(0)
                        if capability[0] >= 7:  # Volta架构及以上支持更好的float16
                            compute_type = "float16"
                            print(f"[INFO] GPU支持float16，使用float16计算类型")
                        else:
                            compute_type = "float32"
                            print(f"[INFO] GPU计算能力较低，使用float32计算类型")
                    else:
                        compute_type = "float32"
                except Exception as e:
                    print(f"[WARNING] 无法检测GPU能力: {e}，使用float32")
                    compute_type = "float32"
            else:
                compute_type = "float32"
                print(f"[INFO] CPU设备使用float32计算类型")
        
        print(f"[INFO] 最终设备配置: {device_name}, 计算类型: {compute_type}")
        
        # 检查模型路径
        if model_path and model_path.strip():
            model_path = model_path.strip()
            print(f"尝试加载模型：{model_path}")
            
            # 如果是本地路径，检查是否存在
            if not model_path.startswith(('tiny', 'base', 'small', 'medium', 'large')):
                if not os.path.exists(model_path):
                    return f"模型路径不存在：{model_path}"
                
                if not os.path.isdir(model_path):
                    return f"模型路径必须是目录：{model_path}"
                
                # 检查目录中是否包含必要的模型文件
                required_files = ['config.json']
                model_files = ['model.bin', 'pytorch_model.bin', 'model.safetensors']
                
                # 检查配置文件
                if not any(os.path.exists(os.path.join(model_path, f)) for f in required_files):
                    return f"模型目录缺少配置文件 (config.json)：{model_path}"
                
                # 检查模型文件
                existing_model_files = [f for f in model_files if os.path.exists(os.path.join(model_path, f))]
                if not existing_model_files:
                    available_files = [f for f in os.listdir(model_path) if f.endswith(('.bin', '.safetensors', '.pt', '.pth'))]
                    if available_files:
                        return f"模型目录包含模型文件但格式不匹配：{available_files}\n支持的格式：{model_files}\n路径：{model_path}"
                    else:
                        return f"模型目录缺少模型文件 (model.bin/pytorch_model.bin/model.safetensors)：{model_path}"
                
                print(f"找到模型文件：{existing_model_files}")
        
        # 检查标准模型的本地路径
        if model_path in app_state.model_list:
            models_path = f"./models/faster-whisper-{model_path}"
            direct_model_path = f"./models/{model_path}"
            
            if os.path.exists(models_path):
                print(f"加载本地模型：{models_path}")
                model_path = models_path
            elif os.path.exists(direct_model_path):
                print(f"加载本地模型：{direct_model_path}")
                model_path = direct_model_path
            else:
                print(f"加载HuggingFace模型：{model_path}")
        
        # 创建Transcribe实例，传递计算类型参数
        app_state.transcribe = Transcribe(model_name=model_path, device=device_name, compute_type=compute_type)
        
        # 保存配置
        app_state.model_name = model_path
        app_state.device_name = device_name
        app_state.compute_type = compute_type
        
        model_display_name = os.path.basename(model_path) if os.path.exists(model_path) else model_path
        return f"模型加载成功：{model_display_name} (设备: {device_name})"
        
    except Exception as e:
        error_msg = f"模型加载失败：{str(e)}"
        print(f"[ERROR] {error_msg}")
        
        # 提供针对性的错误建议
        if 'model.bin' in str(e):
            error_msg += "\n\n建议解决方案：\n1. 检查模型目录是否包含正确的文件\n2. 确保模型文件完整下载\n3. 尝试重新下载模型\n4. 检查文件权限"
        elif 'CUDA' in str(e) or 'cuda' in str(e):
            error_msg += "\n\n建议解决方案：\n1. 检查CUDA是否正确安装\n2. 尝试使用CPU设备\n3. 检查GPU内存是否足够"
        elif 'Invalid input features shape' in str(e):
            error_msg += "\n\n建议解决方案：\n1. 模型特征维度不匹配，可能是版本兼容性问题\n2. 尝试使用标准模型名称如 'large-v3'\n3. 更新faster-whisper和ctranslate2到最新版本\n4. 检查模型是否为正确的Whisper架构"
        
        return error_msg

def clear_cache():
    """清空缓存"""
    try:
        clear_folder("./temp")
        app_state.audio_temp = None
        app_state.video_temp = None
        app_state.audio_separator_temp = None
        return "缓存清空成功！"
    except Exception as e:
        return f"缓存清空失败：{str(e)}"

def upload_media(media_file, media_type):
    """上传媒体文件"""
    if media_file is None:
        return "请选择媒体文件", None, None
    
    try:
        if media_type == "视频":
            # 保存视频文件
            temp_input_video = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_file.name))[0] + "_temp.mp4"
            )
            
            if not os.path.exists(temp_input_video):
                # 复制文件
                import shutil
                shutil.copy2(media_file.name, temp_input_video)
            
            app_state.video_temp_name = os.path.basename(media_file.name)
            app_state.video_temp = temp_input_video
            
            # 提取音频
            temp_audio_path = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_file.name))[0] + ".wav"
            )
            
            if not os.path.exists(temp_audio_path):
                extract_audio(temp_input_video, temp_audio_path)
            
            app_state.audio_temp = temp_audio_path
            return f"视频上传成功：{media_file.name}\n音频提取完成", temp_audio_path, None
            
        else:  # 音频
            temp_audio_path = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_file.name))[0] + ".wav"
            )
            
            if not os.path.exists(temp_audio_path):
                import shutil
                shutil.copy2(media_file.name, temp_audio_path)
            
            app_state.audio_temp = temp_audio_path
            return f"音频上传成功：{media_file.name}", temp_audio_path, None
            
    except Exception as e:
        return f"媒体上传失败：{str(e)}", None, None

def load_media_from_path(media_path, media_type):
    """从路径加载媒体文件"""
    try:
        if not media_path or not media_path.strip():
            return "请输入媒体文件路径", None, None
        
        media_path = media_path.strip()
        
        # 检查文件是否存在
        if not os.path.exists(media_path):
            return f"❌ 文件不存在：{media_path}", None, None
        
        # 检查是否为文件
        if not os.path.isfile(media_path):
            return f"❌ 路径不是文件：{media_path}", None, None
        
        # 获取文件扩展名
        file_ext = os.path.splitext(media_path)[1].lower()
        
        # 检查文件格式
        video_formats = [".mp4", ".avi", ".mov", ".mkv"]
        audio_formats = [".mp3", ".wav", ".m4a"]
        
        if media_type == "视频" and file_ext not in video_formats:
            return f"不支持的视频格式：{file_ext}。支持的格式：{', '.join(video_formats)}", None, None
        elif media_type == "音频" and file_ext not in audio_formats:
            return f"不支持的音频格式：{file_ext}。支持的格式：{', '.join(audio_formats)}", None, None
        
        if media_type == "视频":
            # 保存视频文件
            temp_input_video = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_path))[0] + "_temp.mp4"
            )
            
            if not os.path.exists(temp_input_video):
                # 复制文件
                import shutil
                shutil.copy2(media_path, temp_input_video)
            
            app_state.video_temp_name = os.path.basename(media_path)
            app_state.video_temp = temp_input_video
            
            # 提取音频
            temp_audio_path = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_path))[0] + ".wav"
            )
            
            if not os.path.exists(temp_audio_path):
                extract_audio(temp_input_video, temp_audio_path)
            
            app_state.audio_temp = temp_audio_path
            return f"✅ 视频加载成功：{os.path.basename(media_path)}\n🎵 音频提取完成", temp_audio_path, None
            
        else:  # 音频
            temp_audio_path = os.path.join(
                TEMP,
                os.path.splitext(os.path.basename(media_path))[0] + ".wav"
            )
            
            if not os.path.exists(temp_audio_path):
                import shutil
                shutil.copy2(media_path, temp_audio_path)
            
            app_state.audio_temp = temp_audio_path
            return f"✅ 音频加载成功：{os.path.basename(media_path)}", temp_audio_path, None
            
    except Exception as e:
        return f"加载失败：{str(e)}", None, None

def toggle_upload_method(upload_method):
    """切换上传方式"""
    if upload_method == "文件上传":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def clean_audio():
    """音频清洁（去除背景音）"""
    if app_state.audio_temp is None:
        error_msg = "请先上传媒体文件"
        print(f"[ERROR] {error_msg}")
        return error_msg, None
    
    try:
        if app_state.uvr_client is None:
            print("[INFO] 开始加载UVR模型...")
            try:
                app_state.uvr_client = UVR_Client()
                
                # 检查UVR是否可用
                if not hasattr(app_state.uvr_client, 'uvr_available') or not app_state.uvr_client.uvr_available:
                    print("[WARNING] UVR模型不可用，跳过音频清洁")
                    return f"UVR模型不可用，已跳过音频清洁步骤\n\n原始音频文件: {app_state.audio_temp}\n您可以直接使用此音频进行字幕生成", app_state.audio_temp
                
                print("[INFO] UVR模型加载完成")
            except Exception as uvr_init_error:
                # UVR初始化失败时，返回原始音频
                error_str = str(uvr_init_error)
                print(f"[WARNING] UVR模型初始化失败: {error_str}")
                return f"UVR模型初始化失败，已跳过音频清洁步骤\n\n错误信息: {error_str}\n\n原始音频文件: {app_state.audio_temp}\n您可以直接使用此音频进行字幕生成", app_state.audio_temp
        
        print(f"[INFO] 开始处理音频文件: {app_state.audio_temp}")
        
        # 检查音频文件是否存在
        if not os.path.exists(app_state.audio_temp):
            return f"音频文件不存在：{app_state.audio_temp}", None
        
        # 检查音频文件大小
        file_size = os.path.getsize(app_state.audio_temp)
        if file_size == 0:
            return "音频文件为空，请重新上传", None
        elif file_size < 1024:  # 小于1KB
            return "音频文件过小，可能损坏，请重新上传", None
        
        try:
            # UVR客户端的infer方法只返回一个文件路径
            output_file = app_state.uvr_client.infer(app_state.audio_temp)
            
            # 如果返回的是原始音频文件，说明处理失败
            if output_file == app_state.audio_temp:
                print("[WARNING] UVR处理失败，返回原始音频")
                return f"音频清洁失败，使用原始音频\n\n原始音频: {app_state.audio_temp}\n您可以直接进行字幕生成", app_state.audio_temp
            
            # 设置处理后的音频文件路径
            if os.path.isabs(output_file):
                app_state.audio_separator_temp = output_file
            else:
                app_state.audio_separator_temp = os.path.join('./temp', os.path.basename(output_file))
            
            # 检查输出文件是否生成成功
            if not os.path.exists(app_state.audio_separator_temp):
                print("[WARNING] 音频清洁失败，使用原始音频")
                return f"音频清洁失败，使用原始音频\n\n原始音频: {app_state.audio_temp}\n您可以直接进行字幕生成", app_state.audio_temp
            
            output_size = os.path.getsize(app_state.audio_separator_temp)
            if output_size == 0:
                print("[WARNING] 输出文件为空，使用原始音频")
                return f"音频清洁失败，使用原始音频\n\n原始音频: {app_state.audio_temp}\n您可以直接进行字幕生成", app_state.audio_temp
            
            print(f"[INFO] 音频清洁完成，输出文件: {app_state.audio_separator_temp}")
            return "音频清洁完成", app_state.audio_separator_temp
            
        except Exception as infer_error:
            error_str = str(infer_error)
            print(f"[WARNING] 音频处理过程失败: {error_str}")
            return f"音频处理失败，使用原始音频\n\n错误信息: {error_str}\n\n原始音频: {app_state.audio_temp}\n您可以直接进行字幕生成", app_state.audio_temp
        
    except Exception as e:
        # 任何其他错误，都返回原始音频
        error_str = str(e)
        print(f"[WARNING] 音频清洁过程中发生错误: {error_str}")
        return f"音频清洁过程中发生错误，使用原始音频\n\n错误信息: {error_str}\n\n原始音频: {app_state.audio_temp}\n您可以直接进行字幕生成", app_state.audio_temp

def toggle_model_source(model_source):
    """切换模型来源显示"""
    if model_source == "预设模型":
        return gr.update(visible=True), gr.update(visible=False)
    else:
        return gr.update(visible=False), gr.update(visible=True)

def setup_translation(translation_type, chat_url, chat_key, chat_model, baidu_appid, baidu_appkey, tencent_appid, tencent_secretkey):
    """设置翻译引擎"""
    try:
        if translation_type == "否":
            app_state.engine = None
            return "已关闭翻译功能"
        elif translation_type == "GPT翻译":
            if not chat_key:
                return "请输入GPT API Key"
            app_state.engine = GPT(key=chat_key, base_url=chat_url, model=chat_model)
            return f"GPT翻译引擎设置成功 (模型: {chat_model})"
        elif translation_type == "百度翻译":
            if not baidu_appid or not baidu_appkey:
                return "请输入百度翻译的AppID和AppKey"
            app_state.engine = Baidu(appid=baidu_appid, secretKey=baidu_appkey)
            return "百度翻译引擎设置成功"
        elif translation_type == "腾讯翻译":
            if not tencent_appid or not tencent_secretkey:
                return "请输入腾讯翻译的AppID和SecretKey"
            app_state.engine = Tencent(appid=tencent_appid, secretKey=tencent_secretkey)
            return "腾讯翻译引擎设置成功"
    except Exception as e:
        return f"翻译引擎设置失败：{str(e)}"

def simple_transcribe_audio(audio_file, language, mode="transcribe", enable_translation="启用"):
    """简单音频转录功能"""
    try:
        if not audio_file:
            return "❌ 请先上传音频文件", "", None
        
        if app_state.transcribe is None:
            return "❌ 请先在模型管理页面加载模型", "", None
        
        # 语言映射
        language_map = {
            "中文": "zh",
            "日文": "ja", 
            "英文": "en",
            "自动检测": None
        }
        
        # 执行转录
        lang_code = language_map.get(language)
        
        try:
            # 根据模式选择任务类型
            task = "translate" if mode == "translate" else "transcribe"
            
            srt, ass = app_state.transcribe.run(
                file_name=audio_file,
                audio_binary_io=audio_file,
                language=lang_code,
                task=task,
                is_vad_filter=False,
                is_split=False
            )
            
            # 从SRT文件中提取纯文本
            text_content = ""
            if os.path.exists(srt):
                with open(srt, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        # 跳过序号行、时间戳行和空行
                        if line and not line.isdigit() and '-->' not in line:
                            text_content += line + "\n"
            
            # 处理翻译模式的结果
            if mode == "translate":
                # 检查是否为标准Whisper模型
                model_name = getattr(app_state.transcribe.model, 'model_path', str(app_state.transcribe.model))
                is_standard_model = any(std_model in str(model_name).lower() for std_model in 
                                      ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'])
                
                if is_standard_model:
                    # 标准模型：Whisper已翻译为英文
                    # 如果启用外部翻译引擎，优先使用外部翻译结果
                    if enable_translation == "启用" and app_state.engine is not None:
                        try:
                            from translation import translation
                            t = translation(app_state.engine)
                            translate_ass, translate_srt = t.translate_save(ass)
                            
                            # 从翻译后的SRT文件中提取文本
                            if os.path.exists(translate_srt):
                                with open(translate_srt, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    translated_content = ""
                                    for line in lines:
                                        line = line.strip()
                                        if line and not line.isdigit() and '-->' not in line:
                                            translated_content += line + "\n"
                                text_content = translated_content.strip()
                        except Exception as e:
                            # 外部翻译失败时使用Whisper英文结果
                            pass
                    # 如果没有启用外部翻译或外部翻译失败，使用Whisper英文结果
                    # text_content 保持原始Whisper的英文翻译结果
                else:
                    # 微调模型：可能已直接翻译为目标语言
                    # 如果启用外部翻译引擎，优先使用外部翻译结果
                    if enable_translation == "启用" and app_state.engine is not None:
                        try:
                            from translation import translation
                            t = translation(app_state.engine)
                            translate_ass, translate_srt = t.translate_save(ass)
                            
                            # 从翻译后的SRT文件中提取文本
                            if os.path.exists(translate_srt):
                                with open(translate_srt, 'r', encoding='utf-8') as f:
                                    lines = f.readlines()
                                    translated_content = ""
                                    for line in lines:
                                        line = line.strip()
                                        if line and not line.isdigit() and '-->' not in line:
                                            translated_content += line + "\n"
                                text_content = translated_content.strip()
                        except Exception as e:
                            # 外部翻译失败时使用微调模型结果
                            pass
                    # 如果没有启用外部翻译或外部翻译失败，使用微调模型结果
                    # text_content 保持原始微调模型的翻译结果
            
            # 保存文本文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = "_translated" if mode == "translate" else ""
            txt_file = os.path.join(TEMP, f"transcription_{timestamp}{mode_suffix}.txt")
            with open(txt_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            return f"✅ {mode.capitalize()}完成！", text_content.strip(), txt_file
            
        except Exception as e:
            return f"❌ {mode.capitalize()}失败: {str(e)}", "", None
        
    except Exception as e:
        error_msg = f"❌ {mode.capitalize()}失败: {str(e)}"
        print(f"音频转录时出错: {e}")
        return error_msg, "", None

def process_subtitle(language, subtitle_mode, vad_filter, min_silence_duration, text_split, split_method, prompt, enable_translation, translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model, subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey, show_video, target_language="中文"):
    """处理字幕生成"""
    if app_state.transcribe is None:
        return "请先加载模型", None, None, None
    
    # 选择音频源
    if app_state.audio_separator_temp is not None:
        input_audio = app_state.audio_separator_temp
    elif app_state.audio_temp is not None:
        input_audio = app_state.audio_temp
    else:
        return "请先上传媒体文件", None, None, None
    
    try:
        # 语言映射
        language_mapping = {"中文": "zh", "日文": "ja", "英文": "en"}
        lang_code = language_mapping[language]
        
        # VAD设置
        is_vad_filter = vad_filter == "是"
        min_silence_ms = min_silence_duration if is_vad_filter else None
        
        # 文本分割设置
        is_split = text_split == "是"
        
        # 提示词处理
        initial_prompt = prompt if prompt.strip() else None
        
        print(f"[INFO] 开始处理音频：{input_audio}")
        print(f"[INFO] 转录参数 - 语言: {lang_code}, VAD: {is_vad_filter}, 分割: {is_split}")
        
        # 设置翻译引擎（如果启用翻译）
        if enable_translation == "是":
            try:
                setup_translation(translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model, subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey)
            except Exception as e:
                print(f"[WARNING] 翻译引擎设置失败: {e}")
        
        # 生成字幕
        try:
            # 根据模式选择任务类型
            task = "translate" if subtitle_mode == "translate" else "transcribe"
            
            # 对于translate模式，尝试直接翻译到目标语言
            # 如果是微调模型，可能支持直接翻译到目标语言
            # 如果是标准模型，则只能翻译到英文，需要后续外部翻译
            whisper_target_lang = None
            if subtitle_mode == "translate":
                # 检查是否为标准Whisper模型（这些模型只能翻译为英文）
                model_name = getattr(app_state.transcribe.model, 'model_path', str(app_state.transcribe.model))
                is_standard_model = any(std_model in str(model_name).lower() for std_model in 
                                      ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'])
                
                if is_standard_model:
                    # 标准模型只能翻译为英文
                    whisper_target_lang = "en"
                    print(f"[INFO] 使用标准Whisper模型，翻译为英文")
                else:
                    # 微调模型可能支持直接翻译到目标语言
                    target_lang_map = {"中文": "zh", "日文": "ja", "英文": "en"}
                    whisper_target_lang = target_lang_map.get(target_language, "en")
                    print(f"[INFO] 使用微调模型，尝试直接翻译到{target_language} ({whisper_target_lang})")
            
            # 根据VAD设置选择处理方法
            if is_vad_filter:
                # 使用VAD分割和并发处理
                print(f"[INFO] 启用VAD分割并发处理模式")
                srt, ass = app_state.transcribe.run_with_vad_splitting(
                    file_name=input_audio,
                    audio_binary_io=input_audio,
                    language=lang_code,
                    task=task,
                    is_vad_filter=is_vad_filter,
                    min_silence_duration_ms=min_silence_ms,
                    is_split=is_split,
                    split_method=split_method,
                    initial_prompt=initial_prompt,
                    max_workers=2,  # 并发线程数，可以根据需要调整
                    max_segment_duration=30,  # 最大片段时长30秒
                    min_segment_duration=5   # 最小片段时长5秒
                )
            else:
                # 使用标准处理方法
                srt, ass = app_state.transcribe.run(
                    file_name=input_audio,
                    audio_binary_io=input_audio,
                    language=lang_code,
                    task=task,
                    is_vad_filter=is_vad_filter,
                    min_silence_duration_ms=min_silence_ms,
                    is_split=is_split,
                    split_method=split_method,
                    initial_prompt=initial_prompt
                )
            print(f"[INFO] 字幕生成成功 - 任务类型: {task}")
            
            # 判断是否需要外部翻译
            need_external_translation = False
            if subtitle_mode == "translate":
                # 检查是否为标准模型且目标语言不是英文
                model_name = getattr(app_state.transcribe.model, 'model_path', str(app_state.transcribe.model))
                is_standard_model = any(std_model in str(model_name).lower() for std_model in 
                                      ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'])
                
                if is_standard_model and target_language != "英文":
                    need_external_translation = True
                    print(f"[INFO] 标准模型已翻译为英文，需要外部翻译引擎翻译为{target_language}")
                elif not is_standard_model:
                    print(f"[INFO] 微调模型已尝试直接翻译为{target_language}")
            
            # 处理外部翻译需求
            if need_external_translation:
                if enable_translation != "是" or app_state.engine is None:
                    print(f"[WARNING] 需要外部翻译引擎将英文翻译为{target_language}，但翻译引擎未启用")
                    result_message = f"⚠️ 注意：Whisper已将音频翻译为英文字幕\n要翻译为{target_language}，请启用外部翻译引擎"
                else:
                    # 使用外部翻译引擎进一步翻译
                    enable_translation = "是"  # 确保后续翻译逻辑执行
        except Exception as transcribe_error:
            error_msg = str(transcribe_error)
            print(f"[ERROR] 字幕转录过程失败: {error_msg}")
            
            # 提供用户友好的错误信息
            if "Invalid input features shape" in error_msg:
                user_error_msg = f"模型输入特征不匹配错误\n\n" + \
                               f"错误详情: {error_msg}\n\n" + \
                               f"解决方案:\n" + \
                               f"1. 当前模型可能不兼容，请尝试使用标准模型名称如 'large-v3'\n" + \
                               f"2. 如果使用自定义模型，请确保是正确的CTranslate2格式\n" + \
                               f"3. 尝试切换到CPU设备\n" + \
                               f"4. 检查模型文件是否完整下载"
            elif "CUDA" in error_msg or "cuda" in error_msg:
                user_error_msg = f"GPU/CUDA错误\n\n" + \
                               f"错误详情: {error_msg}\n\n" + \
                               f"解决方案:\n" + \
                               f"1. 尝试切换到CPU设备\n" + \
                               f"2. 检查CUDA和cuDNN版本\n" + \
                               f"3. 重启程序释放GPU内存"
            else:
                user_error_msg = f"字幕生成失败\n\n错误详情: {error_msg}\n\n请检查终端输出获取详细信息"
            
            return user_error_msg, None, None, None
        
        # 创建下载包
        zip_name = os.path.splitext(os.path.basename(app_state.audio_temp))[0] + ".zip"
        zip_name_path = os.path.join("./temp", zip_name)
        
        with ZipFile(zip_name_path, "w") as zipObj:
            zipObj.write(srt, os.path.basename(srt))
            zipObj.write(ass, os.path.basename(ass))
            
            # 如果启用翻译且有翻译引擎
            if enable_translation == "是" and app_state.engine is not None:
                print(f"开始翻译到{target_language}...")
                t = translation(app_state.engine)
                translate_ass, translate_srt = t.translate_save(ass, language=target_language)
                zipObj.write(translate_ass, os.path.basename(translate_ass))
                zipObj.write(translate_srt, os.path.basename(translate_srt))
        
        # 生成结果消息
        if subtitle_mode == "translate":
            # 检查模型类型
            model_name = getattr(app_state.transcribe.model, 'model_path', str(app_state.transcribe.model))
            is_standard_model = any(std_model in str(model_name).lower() for std_model in 
                                  ['tiny', 'base', 'small', 'medium', 'large-v1', 'large-v2', 'large-v3'])
            
            if target_language == "英文":
                result_message = f"✅ 字幕翻译完成！\n🔄 Whisper已将音频翻译为英文字幕\n📄 字幕文件：{os.path.basename(srt)}\n"
            elif is_standard_model:
                # 标准模型的处理
                if enable_translation == "是" and app_state.engine is not None:
                    result_message = f"✅ 字幕翻译完成！\n🔄 标准Whisper模型翻译为英文 → 外部引擎翻译为{target_language}\n📄 原始字幕：{os.path.basename(srt)}\n📄 翻译字幕已生成\n"
                else:
                    result_message = f"⚠️ 部分完成！\n🔄 标准Whisper模型已将音频翻译为英文字幕\n📄 字幕文件：{os.path.basename(srt)}\n💡 要翻译为{target_language}，请启用外部翻译引擎\n"
            else:
                # 微调模型的处理
                if enable_translation == "是" and app_state.engine is not None:
                    result_message = f"✅ 字幕翻译完成！\n🔄 微调模型直接翻译为{target_language} + 外部引擎优化\n📄 原始字幕：{os.path.basename(srt)}\n📄 翻译字幕已生成\n"
                else:
                    result_message = f"✅ 字幕翻译完成！\n🔄 微调模型已尝试直接翻译为{target_language}\n📄 字幕文件：{os.path.basename(srt)}\n💡 如需更好的翻译质量，可启用外部翻译引擎\n"
        else:
            result_message = f"✅ 字幕转录完成！\n📄 字幕文件：{os.path.basename(srt)}\n"
            if enable_translation == "是" and app_state.engine is not None:
                result_message += f"📄 翻译字幕已生成（翻译为{target_language}）\n"
        
        result_message += "\n🎬 可以使用 Aegisub 进行后期编辑优化"
        
        # 生成带字幕的视频（如果需要）
        output_video = None
        if app_state.video_temp and show_video == "是":
            try:
                output_video_path = os.path.join(
                    TEMP,
                    os.path.splitext(os.path.basename(app_state.video_temp_name))[0] + "_output.mp4"
                )
                merge_subtitles_to_video(app_state.video_temp, ass, output_video_path)
                if os.path.exists(output_video_path):
                    output_video = output_video_path
            except Exception as e:
                print(f"视频生成失败：{e}")
        
        return result_message, zip_name_path, srt, output_video
        
    except Exception as e:
        return f"字幕生成失败：{str(e)}", None, None, None

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="AI字幕生成器", theme=gr.themes.Ocean()) as demo:
        gr.Markdown("# 🎬 AI字幕生成器")
        gr.Markdown("基于Whisper的智能字幕生成工具，支持多种语言识别和翻译")
        
        with gr.Tabs():
            # 第一个标签页：模型管理
            with gr.TabItem("⚙️ 模型管理"):
                gr.Markdown("### 模型配置")
                gr.Markdown(
                    "如果本地models目录中没有模型，将自动从HuggingFace下载。\n"
                    "也可以手动下载模型到models目录，或选择自定义微调模型。\n"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ 清空缓存", variant="secondary")
                
                # 模型选择方式
                model_source = gr.Radio(
                    choices=["预设模型", "自定义模型"],
                    value="预设模型",
                    label="模型来源"
                )
                
                # 预设模型选择
                with gr.Group(visible=True) as preset_model_group:
                    with gr.Row():
                        model_name = gr.Dropdown(
                            choices=app_state.model_list,
                            value="large-v3",
                            label="选择预设模型（推荐使用large-v3获得最佳效果）"
                        )
                        device_name = gr.Dropdown(
                            choices=["cpu", "cuda"],
                            value=app_state.device_name,
                            label="计算设备"
                        )
                        compute_type = gr.Dropdown(
                            choices=["int8", "int8_float16", "int16", "float16", "float32"],
                            value=app_state.compute_type,
                            label="计算类型"
                        )
                
                # 自定义模型选择
                with gr.Group(visible=False) as custom_model_group:
                    custom_model_path = gr.Textbox(
                        label="自定义模型路径",
                        placeholder="请输入本地微调模型的文件夹路径",
                        info="支持faster-whisper格式和HuggingFace transformers格式的本地模型"
                    )
                    with gr.Row():
                        custom_device_name = gr.Dropdown(
                            choices=["cpu", "cuda"],
                            value=app_state.device_name,
                            label="计算设备"
                        )
                        custom_compute_type = gr.Dropdown(
                            choices=["int8", "int8_float16", "int16", "float16", "float32"],
                            value=app_state.compute_type,
                            label="计算类型"
                        )
                
                load_model_btn = gr.Button("🚀 加载模型", variant="primary")
                model_status = gr.Textbox(label="模型状态", interactive=False)
            
            # 第二个标签页：音频转录
            with gr.TabItem("🎤 音频转录"):
                gr.Markdown("### 音频转录")
                gr.Markdown("上传音频文件，使用已加载的模型进行转录或翻译")
                
                with gr.Row():
                    with gr.Column():
                        audio_input = gr.Audio(
                            label="上传音频文件",
                            type="filepath",
                            sources=["upload"]
                        )
                        
                        with gr.Row():
                            audio_language = gr.Dropdown(
                                choices=["中文", "日文", "英文", "自动检测"],
                                value="中文",
                                label="音频语言"
                            )
                            
                            audio_mode = gr.Radio(
                                choices=["transcribe", "translate"],
                                value="transcribe",
                                label="处理模式",
                                info="transcribe: 转录原语言 | translate: 标准模型翻译为英文，微调模型可直接翻译"
                            )
                        
                        # 翻译引擎配置（仅在translate模式下显示）
                        with gr.Group(visible=False) as audio_translation_config:
                            gr.Markdown("#### 外部翻译引擎配置")
                            gr.Markdown("💡 **说明**: 标准模型需要外部翻译将英文翻译为其他语言；微调模型可选择启用以获得更好的翻译质量。")
                            
                            audio_enable_translation = gr.Radio(
                                choices=["启用", "禁用"],
                                value="启用",
                                label="外部翻译引擎",
                                info="是否启用外部翻译引擎进行二次翻译"
                            )
                            
                            with gr.Group() as audio_translation_engine_config:
                                audio_translation_type = gr.Radio(
                                    choices=["GPT翻译", "百度翻译", "腾讯翻译"],
                                    value="GPT翻译",
                                    label="翻译引擎"
                                )
                            
                            # GPT翻译配置
                            with gr.Group(visible=True) as audio_gpt_config:
                                with gr.Row():
                                    audio_chat_url = gr.Textbox(
                                        label="Base URL",
                                        value="https://api.openai.com/v1",
                                        type="password"
                                    )
                                    audio_chat_key = gr.Textbox(
                                        label="API Key",
                                        type="password"
                                    )
                                audio_chat_model = gr.Dropdown(
                                    choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                                    value="gpt-4-turbo",
                                    label="模型选择"
                                )
                            
                            # 百度翻译配置
                            with gr.Group(visible=False) as audio_baidu_config:
                                with gr.Row():
                                    audio_baidu_appid = gr.Textbox(label="AppID", type="password")
                                    audio_baidu_appkey = gr.Textbox(label="AppKey", type="password")
                            
                            # 腾讯翻译配置
                            with gr.Group(visible=False) as audio_tencent_config:
                                with gr.Row():
                                    audio_tencent_appid = gr.Textbox(label="AppID", type="password")
                                    audio_tencent_secretkey = gr.Textbox(label="SecretKey", type="password")
                            
                            audio_setup_translation_btn = gr.Button("🔧 设置翻译引擎", variant="secondary")
                        
                        transcribe_btn = gr.Button("🚀 开始处理", variant="primary")
                    
                    with gr.Column():
                        transcribe_result = gr.Textbox(
                            label="处理结果",
                            lines=15,
                            max_lines=20,
                            interactive=False
                        )
                        
                        download_txt = gr.File(label="下载文本文件")
                
                transcribe_status = gr.Textbox(label="处理状态", interactive=False)
            
            # 第三个标签页：字幕生成
            with gr.TabItem("🎬 字幕生成"):
                gr.Markdown("### 媒体上传")
                
                with gr.Row():
                    media_type = gr.Radio(
                        choices=["视频", "音频"],
                        value="视频",
                        label="媒体类型（支持视频格式：mp4, avi, mov, mkv；音频格式：mp3, wav, m4a）"
                    )
                
                # 上传方式选择
                upload_method = gr.Radio(
                    choices=["文件上传", "路径输入"],
                    value="文件上传",
                    label="上传方式"
                )
                
                # 文件上传组件
                with gr.Group(visible=True) as file_upload_group:
                    media_file = gr.File(
                        label="上传媒体文件",
                        file_types=[".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".m4a"]
                    )
                
                # 路径输入组件
                with gr.Group(visible=False) as path_input_group:
                    media_path = gr.Textbox(
                        label="媒体文件路径",
                        placeholder="请输入媒体文件的完整路径，例如：D:\\videos\\example.mp4",
                        info="支持本地文件路径，确保文件存在且格式正确"
                    )
                    load_from_path_btn = gr.Button("📁 从路径加载", variant="secondary")
                
                upload_status = gr.Textbox(label="上传状态", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 原始音频")
                        original_audio = gr.Audio(label="提取的音频", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### 清洁音频（可选）")
                        gr.Markdown(
                            "💡 **音频清洁功能说明**\n"
                            "- 使用UVR技术分离人声和背景音乐\n"
                            "- 首次使用会自动下载模型文件\n"
                            "- 如遇到模型加载失败，请检查网络连接\n"
                            "- 建议使用audio-separator==0.16.5版本"
                        )
                        clean_audio_btn = gr.Button("🧹 音频清洁（去除背景音乐，提高识别准确度）")
                        cleaned_audio = gr.Audio(label="清洁后的音频", interactive=False)
                
                clean_status = gr.Textbox(label="清洁状态", interactive=False)
                
                gr.Markdown("### 转录配置")
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=["中文", "日文", "英文"],
                        value="日文",
                        label="媒体语言（选择音频/视频的主要语言）"
                    )
                    
                    subtitle_mode = gr.Radio(
                        choices=["transcribe", "translate"],
                        value="transcribe",
                        label="字幕模式",
                        info="transcribe: 转录原语言 | translate: 标准模型翻译为英文，微调模型可直接翻译为目标语言"
                    )
                    
                    # 目标翻译语言选择（仅在translate模式下显示）
                    target_language = gr.Dropdown(
                        choices=["中文", "英文", "日文", "韩文", "法文", "德文", "西班牙文", "俄文"],
                        value="中文",
                        label="最终目标语言",
                        info="标准模型: Whisper翻译为英文→外部引擎翻译为此语言 | 微调模型: 可能直接翻译为此语言",
                        visible=False
                    )
                
                with gr.Row():
                    vad_filter = gr.Radio(
                        choices=["是", "否"],
                        value="是",
                        label="启用VAD过滤（过滤无声段落，避免识别出无意义内容）"
                    )
                    
                    text_split = gr.Radio(
                        choices=["是", "否"],
                        value="是",
                        label="文本分割（当单行文本过长时启用）"
                    )
                
                min_silence_duration = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=500,
                    step=100,
                    label="最小静默时长 (毫秒)（仅在启用VAD时生效）",
                    visible=False
                )
                
                split_method = gr.Dropdown(
                    choices=["Modest", "Aggressive"],
                    value="Modest",
                    label="分割方式（Modest: 智能分割; Aggressive: 遇空格就分割）",
                    visible=False
                )
                
                prompt = gr.Textbox(
                    label="提示词（帮助模型更好地识别特定内容）",
                    placeholder="例如：简体中文"
                )
                
                show_video = gr.Radio(
                    choices=["是", "否"],
                    value="是",
                    label="生成带字幕视频（仅对视频文件有效）"
                )
                
                gr.Markdown("### 外部翻译设置")
                gr.Markdown("💡 **说明**: 外部翻译用于优化翻译质量。标准模型需要此功能将英文翻译为其他语言；微调模型可选择启用以获得更好的翻译质量。")
                
                enable_translation = gr.Radio(
                    choices=["否", "是"],
                    value="否",
                    label="启用外部翻译引擎"
                )
                
                with gr.Group(visible=False) as subtitle_translation_config:
                    translation_engine = gr.Radio(
                        choices=["GPT翻译", "百度翻译", "腾讯翻译"],
                        value="GPT翻译",
                        label="翻译引擎"
                    )
                    
                    # GPT翻译配置
                    with gr.Group(visible=True) as subtitle_gpt_config:
                        gr.Markdown("#### GPT翻译配置")
                        with gr.Row():
                            subtitle_chat_url = gr.Textbox(
                                label="Base URL",
                                value="https://api.openai.com/v1",
                                type="password"
                            )
                            subtitle_chat_key = gr.Textbox(
                                label="API Key",
                                type="password"
                            )
                        subtitle_chat_model = gr.Dropdown(
                            choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                            value="gpt-4-turbo",
                            label="模型选择"
                        )
                    
                    # 百度翻译配置
                    with gr.Group(visible=False) as subtitle_baidu_config:
                        gr.Markdown("#### 百度翻译配置")
                        gr.Markdown("[申请地址](https://fanyi-api.baidu.com/manage/developer)")
                        with gr.Row():
                            subtitle_baidu_appid = gr.Textbox(label="AppID", type="password")
                            subtitle_baidu_appkey = gr.Textbox(label="AppKey", type="password")
                    
                    # 腾讯翻译配置
                    with gr.Group(visible=False) as subtitle_tencent_config:
                        gr.Markdown("#### 腾讯翻译配置")
                        gr.Markdown("[申请地址](https://console.cloud.tencent.com/tmt)")
                        with gr.Row():
                            subtitle_tencent_appid = gr.Textbox(label="AppID", type="password")
                            subtitle_tencent_secretkey = gr.Textbox(label="SecretKey", type="password")
                    
                    subtitle_setup_translation_btn = gr.Button("🔧 设置翻译引擎", variant="secondary")
                    subtitle_translation_status = gr.Textbox(label="翻译引擎状态", interactive=False)
                
                gr.Markdown("### 开始处理")
                
                process_btn = gr.Button("🚀 开始生成字幕", variant="primary", size="lg")
                
                process_status = gr.Textbox(label="处理状态", interactive=False)
                
                gr.Markdown("### 下载结果")
                with gr.Row():
                    with gr.Column():
                        download_file = gr.File(label="下载字幕包 (ZIP)")
                        subtitle_preview = gr.File(label="字幕预览 (SRT)")
                    
                    with gr.Column():
                        result_video = gr.Video(label="带字幕视频")
        
        # 事件绑定
        
        # 清空缓存
        clear_btn.click(fn=clear_cache, outputs=[model_status])
        
        # 模型来源切换
        model_source.change(
            fn=toggle_model_source,
            inputs=[model_source],
            outputs=[preset_model_group, custom_model_group]
        )
        
        # 加载模型 - 动态处理预设模型和自定义模型
        def handle_load_model(model_source, model_name, device_name, compute_type, custom_model_path, custom_device_name, custom_compute_type):
            if model_source == "预设模型":
                return load_model(model_name, device_name, compute_type)
            else:
                return load_model(custom_model_path, custom_device_name, custom_compute_type)
        
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[model_source, model_name, device_name, compute_type, custom_model_path, custom_device_name, custom_compute_type],
            outputs=[model_status]
        )
        
        # 音频转录页面事件绑定
        
        # 音频模式切换
        def toggle_audio_translation_config(mode):
            return gr.update(visible=(mode == "translate"))
        
        audio_mode.change(
            fn=toggle_audio_translation_config,
            inputs=[audio_mode],
            outputs=[audio_translation_config]
        )
        
        # 音频外部翻译引擎配置显示/隐藏
        def toggle_audio_translation_enable(enable_status):
            return gr.update(visible=(enable_status == "启用"))
        
        audio_enable_translation.change(
            fn=toggle_audio_translation_enable,
            inputs=[audio_enable_translation],
            outputs=[audio_translation_engine_config]
        )
        
        # 音频翻译引擎配置显示/隐藏
        def toggle_audio_translation_engine(engine_type):
            return (
                gr.update(visible=(engine_type == "GPT翻译")),
                gr.update(visible=(engine_type == "百度翻译")),
                gr.update(visible=(engine_type == "腾讯翻译"))
            )
        
        audio_translation_type.change(
            fn=toggle_audio_translation_engine,
            inputs=[audio_translation_type],
            outputs=[audio_gpt_config, audio_baidu_config, audio_tencent_config]
        )
        
        # 音频翻译引擎设置
        def setup_audio_translation(trans_type, url, key, model, baidu_id, baidu_key, tencent_id, tencent_key):
            return setup_translation(trans_type, url, key, model, baidu_id, baidu_key, tencent_id, tencent_key)
        
        audio_setup_translation_btn.click(
            fn=setup_audio_translation,
            inputs=[audio_translation_type, audio_chat_url, audio_chat_key, audio_chat_model,
                   audio_baidu_appid, audio_baidu_appkey, audio_tencent_appid, audio_tencent_secretkey],
            outputs=[transcribe_status]
        )
        
        # 音频转录
        transcribe_btn.click(
            fn=simple_transcribe_audio,
            inputs=[audio_input, audio_language, audio_mode, audio_enable_translation],
            outputs=[transcribe_status, transcribe_result, download_txt]
        )
        
        # 字幕生成页面事件绑定
        
        # 上传方式切换
        upload_method.change(
            fn=toggle_upload_method,
            inputs=[upload_method],
            outputs=[file_upload_group, path_input_group]
        )
        
        # 媒体上传
        media_file.change(
            fn=upload_media,
            inputs=[media_file, media_type],
            outputs=[upload_status, original_audio, cleaned_audio]
        )
        
        # 从路径加载媒体文件
        load_from_path_btn.click(
            fn=load_media_from_path,
            inputs=[media_path, media_type],
            outputs=[upload_status, original_audio, cleaned_audio]
        )
        
        # 音频清洁
        clean_audio_btn.click(
            fn=clean_audio,
            outputs=[clean_status, cleaned_audio]
        )
        
        # VAD设置显示/隐藏
        def toggle_vad_settings(vad_choice):
            return gr.update(visible=(vad_choice == "是"))
        
        vad_filter.change(
            fn=toggle_vad_settings,
            inputs=[vad_filter],
            outputs=[min_silence_duration]
        )
        
        # 文本分割设置显示/隐藏
        def toggle_split_settings(split_choice):
            return gr.update(visible=(split_choice == "是"))
        
        text_split.change(
            fn=toggle_split_settings,
            inputs=[text_split],
            outputs=[split_method]
        )
        
        # 翻译配置显示/隐藏
        def toggle_subtitle_translation_config(enable):
            return gr.update(visible=(enable == "是"))
        
        enable_translation.change(
            fn=toggle_subtitle_translation_config,
            inputs=[enable_translation],
            outputs=[subtitle_translation_config]
        )
        
        # 目标语言选择显示/隐藏（仅在translate模式下显示）
        def toggle_target_language(mode):
            return gr.update(visible=(mode == "translate"))
        
        subtitle_mode.change(
            fn=toggle_target_language,
            inputs=[subtitle_mode],
            outputs=[target_language]
        )
        
        # 字幕翻译引擎配置显示/隐藏
        def toggle_subtitle_translation_engine(engine_type):
            return (
                gr.update(visible=(engine_type == "GPT翻译")),
                gr.update(visible=(engine_type == "百度翻译")),
                gr.update(visible=(engine_type == "腾讯翻译"))
            )
        
        translation_engine.change(
            fn=toggle_subtitle_translation_engine,
            inputs=[translation_engine],
            outputs=[subtitle_gpt_config, subtitle_baidu_config, subtitle_tencent_config]
        )
        
        # 字幕翻译引擎设置
        def setup_subtitle_translation(trans_type, url, key, model, baidu_id, baidu_key, tencent_id, tencent_key):
            return setup_translation(trans_type, url, key, model, baidu_id, baidu_key, tencent_id, tencent_key)
        
        subtitle_setup_translation_btn.click(
            fn=setup_subtitle_translation,
            inputs=[translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model,
                   subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey],
            outputs=[subtitle_translation_status]
        )
        
        # 处理字幕
        def handle_process_subtitle(language, subtitle_mode, vad_filter, min_silence_duration, text_split, split_method, prompt, enable_translation, translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model, subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey, show_video, target_language):
            return process_subtitle(language, subtitle_mode, vad_filter, min_silence_duration, text_split, split_method, prompt, enable_translation, translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model, subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey, show_video, target_language)
        
        process_btn.click(
            fn=handle_process_subtitle,
            inputs=[language, subtitle_mode, vad_filter, min_silence_duration, text_split, 
                   split_method, prompt, enable_translation, translation_engine, subtitle_chat_url, subtitle_chat_key, subtitle_chat_model, subtitle_baidu_appid, subtitle_baidu_appkey, subtitle_tencent_appid, subtitle_tencent_secretkey, show_video, target_language],
            outputs=[process_status, download_file, subtitle_preview, result_video]
        )
    
    return demo

if __name__ == "__main__":
    # 确保temp目录存在
    if not os.path.exists('temp'):
        os.makedirs('temp')
    
    # 创建并启动界面
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
