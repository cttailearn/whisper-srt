# https://github.com/karaokenerds/python-audio-separator
# pip install audio-separator[gpu]
# pip install audio-separator[cpu]

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
                for local_model in local_models:
                    try:
                        print(f"[INFO] 尝试加载本地模型: {local_model}")
                        self.model.load_model(local_model)
                        print(f"[INFO] 本地模型加载成功: {local_model}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"[WARNING] 本地模型 {local_model} 加载失败: {str(e)}")
                        continue
            
            # 如果本地模型都失败，尝试默认模型
            if not model_loaded:
                default_models = ['UVR_MDXNET_Main.onnx', 'HP2_all_vocals.pth', '5_HP-Karaoke-UVR.pth', 'VR-DeEchoNormal.pth']
                for default_model in default_models:
                    try:
                        print(f"[INFO] 尝试加载默认模型: {default_model}")
                        self.model.load_model(default_model)
                        print(f"[INFO] 默认模型加载成功: {default_model}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"[WARNING] 默认模型 {default_model} 加载失败: {str(e)}")
                        continue
            
            if not model_loaded:
                raise RuntimeError("所有模型加载失败，请检查audio-separator版本或手动下载模型文件")
            
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

    def infer(self,audio="E:\\audio_AI\\audio\\test\\感受孤独.flac"):
        try:
            print(f"[INFO] 开始分离音频: {audio}")
            
            # 检查输入音频文件是否存在
            if not os.path.exists(audio):
                raise FileNotFoundError(f"音频文件不存在: {audio}")
            
            primary_stem_output_path, secondary_stem_output_path = self.model.separate(audio)
            print(f"[INFO] 音频分离完成")
            print(f"[INFO] 主音轨输出: {primary_stem_output_path}")
            print(f"[INFO] 副音轨输出: {secondary_stem_output_path}")
            
            return primary_stem_output_path, secondary_stem_output_path
            
        except Exception as e:
            error_msg = f"音频分离失败: {str(e)}"
            print(f"[ERROR] {error_msg}")
            print(f"[ERROR] 详细错误信息:")
            print(traceback.format_exc())
            
            # 针对特定错误提供解决建议
            if 'roformer_download_list' in str(e).lower():
                print(f"[SUGGESTION] 检测到roformer相关错误，可能的解决方案:")
                print(f"[SUGGESTION] 1. 检查网络连接，确保能够下载模型")
                print(f"[SUGGESTION] 2. 重新安装audio-separator: pip install --upgrade audio-separator[gpu]")
                print(f"[SUGGESTION] 3. 尝试使用其他模型文件")
            elif 'cuda' in str(e).lower() or 'gpu' in str(e).lower():
                print(f"[SUGGESTION] 检测到GPU相关错误，可能的解决方案:")
                print(f"[SUGGESTION] 1. 检查CUDA是否正确安装")
                print(f"[SUGGESTION] 2. 尝试使用CPU版本: pip install audio-separator[cpu]")
            
            raise e


if __name__ == "__main__":
    uvr = UVR_Client()
    print(uvr.infer())
    uvr.change_model("VR-DeEchoAggressive.pth")
    print(uvr.infer())