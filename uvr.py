# https://github.com/karaokenerds/python-audio-separator
# pip install audio-separator[gpu]
# pip install audio-separator[cpu]
#
# 注意：本代码已修改MD5校验处理逻辑
# - MD5校验失败时不会删除模型文件，而是跳过该模型
# - 如需重新下载模型，请手动删除对应文件后重启程序
# - 这样可以避免因网络问题或临时校验失败导致的文件误删

from audio_separator.separator import Separator  
import logging
import traceback
import os
LOG_LE = logging.WARN

class UVR_Client:
    def __init__(self,model_file_dir="./models/uvr5_weights",output_dir='./temp',sample_rate=44000) -> None:
        try:
            print(f"[INFO] 初始化UVR客户端，模型目录: {model_file_dir}")
            
            # 检查模型目录是否存在
            if not os.path.exists(model_file_dir):
                print(f"[WARNING] 模型目录不存在: {model_file_dir}")
                os.makedirs(model_file_dir, exist_ok=True)
                print(f"[INFO] 已创建模型目录: {model_file_dir}")
            
            # 检查输出目录是否存在
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                print(f"[INFO] 已创建输出目录: {output_dir}")
            
            # 尝试初始化Separator，如果失败则提供降级方案
            try:
                self.model = Separator(log_level=LOG_LE,
                                       model_file_dir=model_file_dir,
                                       output_dir=output_dir,
                                       sample_rate=sample_rate)
                print(f"[INFO] Separator初始化成功")
            except Exception as sep_e:
                if 'roformer_download_list' in str(sep_e):
                    print(f"[ERROR] Separator初始化失败，这是audio-separator库版本问题")
                    print(f"[ERROR] 错误详情: {str(sep_e)}")
                    print(f"[SOLUTION] 请执行以下命令修复:")
                    print(f"[SOLUTION] pip uninstall audio-separator")
                    print(f"[SOLUTION] pip install audio-separator==0.16.5")
                    raise RuntimeError("audio-separator库版本不兼容，请降级到0.16.5版本")
                else:
                    raise sep_e
            
            # 检查本地模型文件
            local_models = []
            for file in os.listdir(model_file_dir):
                if file.endswith(('.pth', '.onnx')):
                    local_models.append(file)
            
            if not local_models:
                print(f"[WARNING] 模型目录中没有找到模型文件")
                print(f"[INFO] 将尝试下载默认模型")
            else:
                print(f"[INFO] 找到本地模型文件: {local_models}")
            
            # 尝试加载模型，优先使用本地文件
            model_loaded = False
            
            # 首先尝试加载本地模型文件
            if local_models:
                # 按优先级排序模型文件
                priority_models = []
                for model in local_models:
                    if 'HP2_all_vocals' in model or 'Karaoke' in model:
                        priority_models.insert(0, model)  # 高优先级
                    else:
                        priority_models.append(model)  # 低优先级
                
                for local_model in priority_models:
                    try:
                        print(f"[INFO] 尝试加载本地模型: {local_model}")
                        
                        # 检查文件大小，过小的文件可能损坏
                        model_path = os.path.join(model_file_dir, local_model)
                        file_size = os.path.getsize(model_path)
                        if file_size < 1024:  # 小于1KB的文件可能有问题
                            print(f"[WARNING] 模型文件 {local_model} 大小异常 ({file_size} bytes)，跳过")
                            continue
                        
                        self.model.load_model(local_model)
                        print(f"[INFO] 本地模型加载成功: {local_model}")
                        model_loaded = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        print(f"[WARNING] 本地模型 {local_model} 加载失败: {error_str}")
                        
                        # 如果是MD5哈希错误，跳过该模型但不删除文件
                        if 'MD5 hash' in error_str:
                            print(f"[WARNING] 模型 {local_model} MD5校验失败，跳过该模型（文件保留）")
                            print(f"[INFO] 如需重新下载该模型，请手动删除文件后重启程序")
                        continue
            
            # 如果本地模型都失败，尝试默认模型（这些会自动下载）
            if not model_loaded:
                # 扩展的模型列表，包含更多兼容的模型
                default_models = [
                    # VR架构模型（通常兼容性更好）
                    'UVR-MDX-NET-Voc_FT.onnx',
                    'UVR-MDX-NET-Inst_HQ_3.onnx', 
                    'Kim_Vocal_2.onnx',
                    'kuielab_a_vocals.onnx',
                    
                    # 传统VR模型
                    '1_HP-DeEcho-De-reverb_By_FoxJoy.pth',
                    '2_HP-UVR.pth',
                    '3_HP-Vocal-UVR.pth',
                    '4_HP-Vocal-UVR.pth',
                    
                    # 备用模型
                    'HP2_all_vocals.pth',
                    '5_HP-Karaoke-UVR.pth',
                    'VR-DeEchoNormal.pth',
                    'UVR_MDXNET_Main.onnx'
                ]
                
                print(f"[INFO] 开始尝试加载兼容的默认模型...")
                
                for default_model in default_models:
                    try:
                        print(f"[INFO] 尝试下载并加载默认模型: {default_model}")
                        self.model.load_model(default_model)
                        print(f"[INFO] 默认模型加载成功: {default_model}")
                        model_loaded = True
                        break
                    except Exception as e:
                        error_str = str(e)
                        print(f"[WARNING] 默认模型 {default_model} 加载失败: {error_str}")
                        
                        # 提供具体的错误分析
                        if 'not found in supported model files' in error_str:
                            print(f"[INFO] 模型 {default_model} 不在支持列表中，尝试下一个")
                        elif 'MD5 hash' in error_str:
                            print(f"[INFO] 模型 {default_model} MD5校验失败，跳过该模型（如有本地文件将保留）")
                            print(f"[INFO] 如需重新下载该模型，请手动删除对应文件后重启程序")
                        elif 'network' in error_str.lower() or 'download' in error_str.lower():
                            print(f"[INFO] 网络问题导致模型下载失败，尝试下一个")
                        elif 'timeout' in error_str.lower():
                            print(f"[INFO] 下载超时，尝试下一个模型")
                        continue
                
                # 如果所有预定义模型都失败，尝试使用默认配置初始化
                if not model_loaded:
                    try:
                        print(f"[INFO] 尝试使用默认配置初始化分离器...")
                        # 不指定模型，让audio-separator使用默认模型
                        test_audio = os.path.join(self.model.output_dir, 'test_init.wav')
                        # 创建一个很短的测试音频来验证分离器是否工作
                        import numpy as np
                        import soundfile as sf
                        
                        # 生成1秒的静音测试音频
                        test_data = np.zeros((44100, 2), dtype=np.float32)
                        sf.write(test_audio, test_data, 44100)
                        
                        # 尝试处理测试音频
                        result = self.model.separate(test_audio)
                        if result:
                            print(f"[INFO] 默认配置初始化成功")
                            model_loaded = True
                            # 清理测试文件
                            if os.path.exists(test_audio):
                                os.remove(test_audio)
                        
                    except Exception as default_e:
                        print(f"[WARNING] 默认配置初始化也失败: {str(default_e)}")
            
            if not model_loaded:
                print(f"[WARNING] 所有UVR模型加载失败，音频清洁功能将不可用")
                print(f"[INFO] 您仍然可以使用原始音频进行字幕生成")
                
                # 不抛出异常，而是设置一个标记表示UVR不可用
                self.model = None
                self.uvr_available = False
                
                print(f"[SUGGESTION] 如需使用音频清洁功能，请尝试以下解决方案：")
                print(f"[SUGGESTION] 1. 重新安装audio-separator: pip install audio-separator==0.16.5")
                print(f"[SUGGESTION] 2. 检查网络连接并重试")
                print(f"[SUGGESTION] 3. 清空 models/uvr5_weights 目录后重启程序")
                print(f"[SUGGESTION] 4. 或直接使用原始音频进行字幕生成")
            else:
                self.uvr_available = True
            
        except Exception as e:
            error_msg = f"UVR客户端初始化失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 详细错误信息:")
            print(traceback.format_exc())
            
            # 针对特定错误提供解决建议
            if 'roformer_download_list' in str(e).lower():
                print(f"[SUGGESTION] 检测到roformer相关错误，这是audio-separator库版本兼容性问题")
                print(f"[SUGGESTION] 1. 尝试降级到稳定版本: pip install audio-separator==0.16.5")
                print(f"[SUGGESTION] 2. 或者尝试最新版本: pip install --upgrade audio-separator[gpu]")
                print(f"[SUGGESTION] 3. 系统已自动尝试使用备用模型")
                print(f"[SUGGESTION] 4. 手动下载模型文件到 {model_file_dir}")
        
            raise e

    def change_model(self,model_name):
        try:
            print(f"[INFO] 切换模型: {model_name}")
            self.model.load_model(model_name)
            print(f"[INFO] 模型切换成功: {model_name}")
        except Exception as e:
            error_msg = f"模型切换失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 详细错误信息:")
            print(traceback.format_exc())
            
            if 'roformer_download_list' in str(e).lower():
                print(f"[SUGGESTION] 检测到roformer相关错误，可能的解决方案:")
                print(f"[SUGGESTION] 1. 检查网络连接")
                print(f"[SUGGESTION] 2. 重新安装audio-separator")
                print(f"[SUGGESTION] 3. 检查模型文件 {model_name} 是否存在")
            
            raise e

    def infer(self, input_audio):
        """音频分离推理"""
        try:
            print(f"[INFO] 开始分离音频: {input_audio}")
            
            # 检查UVR是否可用
            if not hasattr(self, 'uvr_available') or not self.uvr_available or self.model is None:
                print(f"[INFO] UVR模型不可用，返回原始音频文件: {input_audio}")
                return input_audio
            
            # 检查输入音频文件是否存在
            if not os.path.exists(input_audio):
                raise FileNotFoundError(f"音频文件不存在: {input_audio}")
            
            # 执行音频分离
            output_files = self.model.separate(input_audio)
            
            # 返回人声文件路径
            if output_files:
                # 通常第一个文件是人声
                result_file = output_files[0] if isinstance(output_files, list) else output_files
                print(f"[INFO] 音频清洁完成: {result_file}")
                return result_file
            else:
                print(f"[WARNING] 音频分离未生成输出文件，返回原始音频")
                return input_audio
                
        except Exception as e:
            print(f"[WARNING] UVR音频分离失败: {str(e)}，返回原始音频")
            return input_audio


if __name__ == "__main__":
    uvr = UVR_Client()
    test_audio = "E:\\audio_AI\\audio\\test\\感受孤独.flac"
    print(uvr.infer(test_audio))
    uvr.change_model("VR-DeEchoAggressive.pth")
    print(uvr.infer(test_audio))
