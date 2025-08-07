import gradio as gr
import json
import os
import traceback
from transcribe import Transcribe
from zipfile import ZipFile
import base64
import io
import ffmpeg
from translation import GPT, Baidu, Tencent, translation
from utils import extract_audio, merge_subtitles_to_video, clear_folder, import_config_file
from uvr import UVR_Client

# 临时文件存放地址
TEMP = "./temp"

# 全局变量存储状态
class AppState:
    def __init__(self):
        self.transcribe = None
        self.config = None
        self.audio_temp = None
        self.video_temp = None
        self.video_temp_name = None
        self.audio_separator_temp = None
        self.uvr_client = None
        self.engine = None
        
        # 默认配置
        self.model_list = ["tiny", "base", "small", "medium", "large-v2", "large-v3",
                          "tiny.en", "base.en", "medium.en", "small.en"]
        self.model_name = "large-v2"
        self.chat_url = "https://api.openai.com/v1"
        self.chat_key = ""
        self.chat_model_list = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
        self.chat_model_name = "gpt-4-turbo"
        self.baidu_appid = ""
        self.baidu_appkey = ""
        self.tencent_appid = ""
        self.tencent_secretKey = ""

app_state = AppState()

def load_config(config_file):
    """加载配置文件"""
    if config_file is None:
        return "请选择配置文件", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    try:
        with open(config_file.name, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        app_state.config = config_data
        
        # 更新各种配置
        model_name = config_data.get("model_name", app_state.model_name)
        chat_url = config_data.get("chat_url", app_state.chat_url)
        chat_key = config_data.get("chat_key", app_state.chat_key)
        chat_model = config_data.get("chat_model_name", app_state.chat_model_name)
        baidu_appid = config_data.get("baidu_appid", app_state.baidu_appid)
        baidu_appkey = config_data.get("baidu_appkey", app_state.baidu_appkey)
        tencent_appid = config_data.get("tencent_appid", app_state.tencent_appid)
        tencent_secretkey = config_data.get("tencent_secretKey", app_state.tencent_secretKey)
        
        return ("配置文件加载成功！", 
                gr.update(value=model_name),
                gr.update(value=chat_url),
                gr.update(value=chat_key),
                gr.update(value=chat_model),
                gr.update(value=baidu_appid),
                gr.update(value=baidu_appkey),
                gr.update(value=tencent_appid),
                gr.update(value=tencent_secretkey))
    except Exception as e:
        return f"配置文件加载失败: {str(e)}", gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()

def load_model(model_name, device_name, custom_model_path=None):
    """加载转录模型"""
    try:
        if app_state.transcribe is not None:
            del app_state.transcribe
        
        # 优先使用自定义模型路径
        if custom_model_path and os.path.exists(custom_model_path):
            print(f"加载自定义本地模型：{custom_model_path}")
            app_state.transcribe = Transcribe(model_name=custom_model_path, device=device_name)
            model_display_name = os.path.basename(custom_model_path)
            return f"自定义模型加载成功：{model_display_name} (设备: {device_name})"
        
        # 检查标准模型路径
        models_path = "./models" + "/faster-whisper-" + model_name
        
        if os.path.exists(models_path):
            print(f"加载本地模型：{models_path}")
            app_state.transcribe = Transcribe(model_name=models_path, device=device_name)
        else:
            print(f"加载HuggingFace模型：{model_name}")
            app_state.transcribe = Transcribe(model_name=model_name, device=device_name)
        
        app_state.model_name = model_name
        return f"模型加载成功：{model_name} (设备: {device_name})"
    except Exception as e:
        return f"模型加载失败：{str(e)}"

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

def clean_audio():
    """音频清洁（去除背景音）"""
    if app_state.audio_temp is None:
        error_msg = "请先上传媒体文件"
        print(f"[ERROR] {error_msg}")
        return error_msg, None
    
    try:
        if app_state.uvr_client is None:
            print("[INFO] 开始加载UVR模型...")
            app_state.uvr_client = UVR_Client()
            print("[INFO] UVR模型加载完成")
        
        print(f"[INFO] 开始处理音频文件: {app_state.audio_temp}")
        primary_stem_output_path, secondary_stem_output_path = app_state.uvr_client.infer(app_state.audio_temp)
        app_state.audio_separator_temp = os.path.join('./temp', secondary_stem_output_path)
        
        print(f"[INFO] 音频清洁完成，输出文件: {app_state.audio_separator_temp}")
        return "音频清洁完成", app_state.audio_separator_temp
    except Exception as e:
        # 获取完整的错误堆栈信息
        error_traceback = traceback.format_exc()
        error_msg = f"音频清洁失败：{str(e)}"
        
        # 在终端输出详细错误信息
        print(f"[ERROR] {error_msg}")
        print(f"[ERROR] 详细错误信息:")
        print(error_traceback)
        
        # 针对特定错误提供解决建议
        if 'roformer_download_list' in str(e).lower() or 'audio-separator库版本不兼容' in str(e):
            suggestion = "\n建议解决方案：\n1. 这是audio-separator库版本兼容性问题\n2. 请执行以下命令修复：\n   pip uninstall audio-separator\n   pip install audio-separator==0.16.5\n3. 重启应用程序\n4. 如果问题持续，请检查Python环境"
            print(f"[SUGGESTION] {suggestion}")
            error_msg += suggestion
        elif 'model' in str(e).lower():
            suggestion = "\n建议解决方案：\n1. 检查 models/uvr5_weights 目录下的模型文件\n2. 重新下载模型文件\n3. 检查模型文件权限"
            print(f"[SUGGESTION] {suggestion}")
            error_msg += suggestion
        
        return error_msg, None

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

def process_subtitle(language, vad_filter, min_silence_duration, text_split, split_method, prompt, show_video):
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
        
        print(f"开始处理音频：{input_audio}")
        
        # 生成字幕
        srt, ass = app_state.transcribe.run(
            file_name=input_audio,
            audio_binary_io=input_audio,
            language=lang_code,
            is_vad_filter=is_vad_filter,
            min_silence_duration_ms=min_silence_ms,
            is_split=is_split,
            split_method=split_method,
            initial_prompt=initial_prompt
        )
        
        # 创建下载包
        zip_name = os.path.splitext(os.path.basename(app_state.audio_temp))[0] + ".zip"
        zip_name_path = os.path.join("./temp", zip_name)
        
        with ZipFile(zip_name_path, "w") as zipObj:
            zipObj.write(srt, os.path.basename(srt))
            zipObj.write(ass, os.path.basename(ass))
            
            # 如果需要翻译
            if app_state.engine is not None:
                print("开始翻译...")
                t = translation(app_state.engine)
                translate_ass, translate_srt = t.translate_save(ass)
                zipObj.write(translate_ass, os.path.basename(translate_ass))
                zipObj.write(translate_srt, os.path.basename(translate_srt))
        
        result_message = f"字幕生成完成！\n原始字幕：{os.path.basename(srt)}\n"
        if app_state.engine is not None:
            result_message += "翻译字幕已生成\n"
        result_message += "\n可以使用 Aegisub 进行后期编辑优化"
        
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
            # 配置标签页
            with gr.TabItem("⚙️ 配置管理"):
                gr.Markdown("### 配置文件管理")
                with gr.Row():
                    config_file = gr.File(label="上传配置文件 (JSON)", file_types=[".json"])
                    clear_btn = gr.Button("🗑️ 清空缓存", variant="secondary")
                
                config_status = gr.Textbox(label="配置状态", interactive=False)
                
                gr.Markdown("### 模型配置")
                gr.Markdown(
                    "如果本地models目录中没有模型，将自动从HuggingFace下载。\n"
                    "也可以手动下载模型到models目录，或选择自定义微调模型。\n"
                )
                
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
                            value="cuda",
                            label="计算设备（强烈建议使用CUDA加速）"
                        )
                
                # 自定义模型选择
                with gr.Group(visible=False) as custom_model_group:
                    gr.Markdown(
                        "#### 📝 自定义模型使用说明\n"
                        "- **faster-whisper格式**：包含config.json、model.bin、tokenizer.json等文件的文件夹\n"
                        "- **HuggingFace格式**：包含pytorch_model.bin、config.json、tokenizer.json等文件的文件夹\n"
                        "- **支持微调模型**：使用OpenAI Whisper、faster-whisper或transformers库训练的模型\n"
                        "- **路径示例**：`./models/my-whisper-model` 或 `D:/models/fine-tuned-whisper`"
                    )
                    custom_model_path = gr.Textbox(
                        label="自定义模型路径",
                        placeholder="请输入本地微调模型的文件夹路径，例如：./models/my-fine-tuned-whisper",
                        info="支持faster-whisper格式和HuggingFace transformers格式的本地模型"
                    )
                    custom_device_name = gr.Dropdown(
                        choices=["cpu", "cuda"],
                        value="cuda",
                        label="计算设备（强烈建议使用CUDA加速）"
                    )
                
                load_model_btn = gr.Button("🚀 加载模型", variant="primary")
                model_status = gr.Textbox(label="模型状态", interactive=False)
            
            # 媒体上传标签页
            with gr.TabItem("📁 媒体上传"):
                gr.Markdown("### 选择媒体类型")
                media_type = gr.Radio(
                    choices=["视频", "音频"],
                    value="视频",
                    label="媒体类型（支持视频格式：mp4, avi, mov, mkv；音频格式：mp3, wav, m4a）"
                )
                
                media_file = gr.File(
                    label="上传媒体文件",
                    file_types=[".mp4", ".avi", ".mov", ".mkv", ".mp3", ".wav", ".m4a"]
                )
                
                upload_status = gr.Textbox(label="上传状态", interactive=False)
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### 原始音频")
                        original_audio = gr.Audio(label="提取的音频", interactive=False)
                    
                    with gr.Column():
                        gr.Markdown("#### 清洁音频（可选）")
                        clean_audio_btn = gr.Button("🧹 音频清洁（去除背景音乐，提高识别准确度）")
                        cleaned_audio = gr.Audio(label="清洁后的音频", interactive=False)
                
                clean_status = gr.Textbox(label="清洁状态", interactive=False)
            
            # 转录配置标签页
            with gr.TabItem("🎯 转录配置"):
                gr.Markdown("### 语言和处理设置")
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=["中文", "日文", "英文"],
                        value="日文",
                        label="媒体语言（选择音频/视频的主要语言）"
                    )
                    
                    vad_filter = gr.Radio(
                        choices=["是", "否"],
                        value="否",
                        label="启用VAD过滤（过滤无声段落，避免识别出无意义内容）"
                    )
                
                min_silence_duration = gr.Slider(
                    minimum=0,
                    maximum=10000,
                    value=500,
                    step=100,
                    label="最小静默时长 (毫秒)（仅在启用VAD时生效）",
                    visible=False
                )
                
                with gr.Row():
                    text_split = gr.Radio(
                        choices=["是", "否"],
                        value="否",
                        label="文本分割（当单行文本过长时启用）"
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
            
            # 翻译设置标签页
            with gr.TabItem("🌐 翻译设置"):
                gr.Markdown("### 翻译引擎配置")
                
                translation_type = gr.Radio(
                    choices=["否", "GPT翻译", "百度翻译", "腾讯翻译"],
                    value="否",
                    label="翻译选项（选择翻译服务，翻译为中文）"
                )
                
                # GPT翻译配置
                with gr.Group(visible=False) as gpt_config:
                    gr.Markdown("#### GPT翻译配置")
                    with gr.Row():
                        chat_url = gr.Textbox(
                            label="Base URL",
                            value="https://api.openai.com/v1",
                            type="password"
                        )
                        chat_key = gr.Textbox(
                            label="API Key",
                            type="password"
                        )
                    chat_model = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                        value="gpt-4-turbo",
                        label="模型选择"
                    )
                
                # 百度翻译配置
                with gr.Group(visible=False) as baidu_config:
                    gr.Markdown("#### 百度翻译配置")
                    gr.Markdown("[申请地址](https://fanyi-api.baidu.com/manage/developer)")
                    with gr.Row():
                        baidu_appid = gr.Textbox(label="AppID", type="password")
                        baidu_appkey = gr.Textbox(label="AppKey", type="password")
                
                # 腾讯翻译配置
                with gr.Group(visible=False) as tencent_config:
                    gr.Markdown("#### 腾讯翻译配置")
                    gr.Markdown("[申请地址](https://console.cloud.tencent.com/tmt)")
                    with gr.Row():
                        tencent_appid = gr.Textbox(label="AppID", type="password")
                        tencent_secretkey = gr.Textbox(label="SecretKey", type="password")
                
                setup_translation_btn = gr.Button("🔧 设置翻译引擎", variant="secondary")
                translation_status = gr.Textbox(label="翻译引擎状态", interactive=False)
            
            # 处理结果标签页
            with gr.TabItem("🎬 处理与结果"):
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
        
        # 配置文件加载
        config_file.change(
            fn=load_config,
            inputs=[config_file],
            outputs=[config_status, model_name, chat_url, chat_key, chat_model, 
                    baidu_appid, baidu_appkey, tencent_appid, tencent_secretkey]
        )
        
        # 清空缓存
        clear_btn.click(fn=clear_cache, outputs=[config_status])
        
        # 模型来源切换
        model_source.change(
            fn=toggle_model_source,
            inputs=[model_source],
            outputs=[preset_model_group, custom_model_group]
        )
        
        # 加载模型 - 动态处理预设模型和自定义模型
        def handle_load_model(model_source, model_name, device_name, custom_model_path, custom_device_name):
            if model_source == "预设模型":
                return load_model(model_name, device_name)
            else:
                return load_model(None, custom_device_name, custom_model_path)
        
        load_model_btn.click(
            fn=handle_load_model,
            inputs=[model_source, model_name, device_name, custom_model_path, custom_device_name],
            outputs=[model_status]
        )
        
        # 媒体上传
        media_file.change(
            fn=upload_media,
            inputs=[media_file, media_type],
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
        def toggle_translation_config(trans_type):
            return (
                gr.update(visible=(trans_type == "GPT翻译")),
                gr.update(visible=(trans_type == "百度翻译")),
                gr.update(visible=(trans_type == "腾讯翻译"))
            )
        
        translation_type.change(
            fn=toggle_translation_config,
            inputs=[translation_type],
            outputs=[gpt_config, baidu_config, tencent_config]
        )
        
        # 设置翻译引擎
        setup_translation_btn.click(
            fn=setup_translation,
            inputs=[translation_type, chat_url, chat_key, chat_model,
                   baidu_appid, baidu_appkey, tencent_appid, tencent_secretkey],
            outputs=[translation_status]
        )
        
        # 处理字幕
        process_btn.click(
            fn=process_subtitle,
            inputs=[language, vad_filter, min_silence_duration, text_split, 
                   split_method, prompt, show_video],
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