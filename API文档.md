# 音频分离 API 文档

## 概述
本 API 提供音乐文件分离功能，可以将音频文件中的人声和伴奏分离。

---

## 服务器信息
- **地址**: `http://localhost:5000`
- **启动命令**: `python -m src.server`

---

## 接口列表

### 1. 分离音频（POST）
**端点**: `/separate`

**功能**: 上传音频文件并分离人声和伴奏

**请求格式**:
- **Content-Type**: `multipart/form-data`
- **字段**:
  - `file`: 音频文件（支持 MP3、WAV、FLAC 等格式）

**请求示例（curl）**:
```bash
curl -X POST \
  -F 'file=@/path/to/your/song.mp3' \
  http://localhost:5000/separate
```

**请求示例（Python）**:
```python
import requests

url = 'http://localhost:5000/separate'
files = {'file': open('your_song.mp3', 'rb')}
response = requests.post(url, files=files)

print(response.json())
```

**响应示例（成功）**:
```json
{
  "success": true,
  "vocals_path": "g:/jz_work/jz3/fork3/AICoverGen/temp_output/song_vocals.wav",
  "accompaniment_path": "g:/jz_work/jz3/fork3/AICoverGen/temp_output/song_accompaniment.wav"
}
```

**响应示例（失败）**:
```json
{
  "success": false,
  "error": "错误描述信息"
}
```

**状态码**:
- `200`: 分离成功
- `400`: 请求参数错误（如未上传文件）
- `500`: 服务器内部错误

---

### 2. 下载文件（GET）
**端点**: `/download/<filename>`

**功能**: 下载分离后的音频文件

**请求示例（浏览器）**:
```
http://localhost:5000/download/song_vocals.wav
http://localhost:5000/download/song_accompaniment.wav
```

**请求示例（curl）**:
```bash
curl -X GET http://localhost:5000/download/song_vocals.wav -o vocals.wav
```

**请求示例（Python）**:
```python
import requests

url = 'http://localhost:5000/download/song_vocals.wav'
response = requests.get(url)

with open('vocals.wav', 'wb') as f:
    f.write(response.content)
```

**响应**:
- `200`: 返回文件内容
- `404`: 文件不存在

---

## 使用流程

1. **启动服务器**:
```bash
cd g:/jz_work/jz3/fork3/AICoverGen
python -m src.server
```

2. **上传音频文件**（使用 POST 请求到 `/separate`）

3. **获取分离结果**（响应中包含人声和伴奏的文件路径）

4. **下载分离后的文件**（使用 GET 请求到 `/download/<filename>`）

---

## 错误处理

### 常见错误及解决方法:

1. **"No file part"**
   - 原因: 请求中没有包含文件字段
   - 解决: 确保请求中包含 `file` 字段

2. **"No selected file"**
   - 原因: 上传了空文件
   - 解决: 选择有效的音频文件上传

3. **"无法连接到服务器"**
   - 原因: 服务器未启动或地址错误
   - 解决: 确保服务器已启动，地址为 `http://localhost:5000`

4. **"File not found"**（下载时）
   - 原因: 要下载的文件不存在
   - 解决: 确保文件已成功分离，并且文件名正确

---

## 支持的音频格式

- MP3
- WAV
- FLAC
- AAC
- OGG
- 其他常见音频格式

---

## 注意事项

1. 处理时间取决于音频文件的长度和复杂度
2. 分离后的文件格式为 WAV
3. 临时文件会保存在 `temp_output` 目录中
4. 建议定期清理 `temp_output` 目录以释放存储空间

---

## 测试脚本

提供了测试脚本 `test_separate.py`，可以用来测试 API 功能:

```bash
python test_separate.py
```

注意需要先修改脚本中的音频文件路径。

---

## 技术实现

- **框架**: Flask
- **分离模型**: MDXNet
- **音频处理**: Librosa, SoundFile
- **并发处理**: 多线程

---

## 联系方式

如有问题，请联系开发人员。