# whisper-srt
使用whisper 模型进行语音识别，生成字幕srt文件，项目根据[auto-subtitle](https://github.com/lissettecarlr/auto-subtitle)修改

## 🚀 主要功能

* **媒体输入**：支持视频和音频文件上传
* **智能识别**：基于faster-whisper的高精度语音识别
* **多语言支持**：中文、日文、英文识别
* **字幕输出**：生成SRT和ASS格式字幕
* **翻译服务**：
  - GPT翻译（支持OpenAI API）
  - 百度翻译
  - 腾讯翻译
* **音频清洁**：去除背景音乐，提高识别准确度
* **配置管理**：支持配置文件导入导出
* **Web界面**：基于Gradio的现代化用户界面

## 环境

* conda
    ```bash
    conda create -n subtitle python=3.10
    conda activate subtitle
    ```

* torch（CUDA 11.8，其他版本去[官网](https://pytorch.org/get-started/locally/)找）
    ```bash
    # GPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

    # CPU
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```

* 安装ffmpeg（windows）。
    去[官网](https://ffmpeg.org/download.html#build-windows)下载，解压后将bin目录添加到环境变量

* 安装ffmpeg（ubuntu）
    ```bash
    apt install ffmpeg
    ```

* 安装依赖
    ```bash
    # 安装所有依赖
    pip install -r requirements.txt
    
    # 如果遇到audio-separator版本兼容问题，请使用指定版本
    pip uninstall audio-separator
    pip install audio-separator==0.16.5
    
    # GPU加速版本（推荐，需要CUDA支持）
    pip install audio-separator[gpu]==0.16.5
    
    # CPU版本
    pip install audio-separator[cpu]==0.16.5
    ```

* **重要提示**：
  - 如果遇到 `'roformer_download_list'` 错误，请确保使用 `audio-separator==0.16.5` 版本
  - 建议使用GPU版本以获得更好的性能


## 📦 模型下载

### Whisper语音识别模型

* **自动下载**：首次使用时会自动从HuggingFace下载模型
* **手动下载**：将模型文件夹放入 `./models/` 目录
* **推荐模型**：`large-v3`（最佳效果）或 `large-v2`（平衡性能）
* **模型来源**：
  - [HuggingFace faster-whisper](https://huggingface.co/collections/guillaumekln/faster-whisper-64f9c349b3115b4f51434976)
  - [百度云备份](https://pan.baidu.com/s/1NbutR2cHvHbboUy-QTg5zw?pwd=kuon)

### 音频清洁模型（UVR）

* **必需文件**：`UVR_MDXNET_Main.onnx`
* **存放位置**：`./models/uvr5_weights/`
* **下载链接**：[百度云](https://pan.baidu.com/s/1wDQ_I1NIL942o1Dm2XU8zg?pwd=kuon)
* **备用模型**：系统会自动尝试加载备用模型（如HP2_all_vocals.pth等）

### 📁 目录结构示例
```text
│models/
├───faster-whisper-large-v3/
│       config.json
│       model.bin
│       preprocessor_config.json
│       tokenizer.json
│       vocabulary.json
│
└───uvr5_weights/
        UVR_MDXNET_Main.onnx
        HP2_all_vocals.pth
        5_HP-Karaoke-UVR.pth
        VR-DeEchoNormal.pth
        download_checks.json
        mdx_model_data.json
        vr_model_data.json
```

## 🚀 运行应用

```bash
# 启动Gradio Web界面
python gradio_web.py

# 或者使用传统的Streamlit界面
python web.py
```

访问 `http://localhost:7860` 即可使用Web界面

## 📖 使用说明

### 1. 配置管理
- 上传JSON配置文件或手动配置各项参数
- 选择Whisper模型（推荐large-v3）
- 设置计算设备（推荐CUDA）

### 2. 媒体处理
- 上传视频或音频文件
- 可选择进行音频清洁（去除背景音乐）
- 支持多种格式：mp4, avi, mov, mkv, mp3, wav, m4a

### 3. 转录配置
- 选择媒体语言（中文/日文/英文）
- 配置VAD过滤和文本分割
- 设置提示词以提高识别准确度

### 4. 翻译设置
- 支持GPT、百度、腾讯三种翻译服务
- 可配置API密钥和相关参数

### 5. 生成字幕
- 一键生成SRT和ASS格式字幕
- 自动打包下载所有文件
- 可选生成带字幕的视频文件

演示视频：
<video src="https://github.com/lissettecarlr/auto-subtitle/assets/16299917/bd83db31-a830-441a-82ad-caccaa9c3833" controls="controls" width="100%" height="100%"></video>




## 效果


### 葬送的芙莉蓮 OP 主題曲 -「勇者」/ YOASOBI

|识别出的歌词|本软件输出|
|---|---|
|まるでおとぎの話 終わり迎えた証|就像童话故事迎来了结局的证明|
|長すぎる旅路から 切り出した一節|从过长的旅程中切出的一节|
|それはかつてこの地に 影を落とした悪を|那是曾经在这片土地上投下阴影的恶|
|打ち取る自由者との 短い旅の記憶 | 是与击败自由者的短暂旅行的记忆|
|物語は終わり 勇者は眠りにつく | 故事结束了 勇者已经入睡|
|穏やかな日常を この地に残して | 留下了平静的日常在这片土地上|
|時の眺めは無情に 人を忘れさせる | 时间的眺望无情地让人忘记|
|そこに生きた奇跡も 錆びついてく | 在那里生活的奇迹也开始生锈了|
|それでも君は 生きてる | 但是你依然活着|
|君の言葉も 願いも 勇気も | 你的话语 你的愿望 你的勇气|
|今は確かに私の中で 生きてる | 现在它们确实在我心中活着|
|同じ道を選んだ それだけだった | 只是选择了相同的道路|


## 参考

* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* [N46Whisper](https://github.com/Ayanaminn/N46Whisper/blob/main/README_CN.md)
