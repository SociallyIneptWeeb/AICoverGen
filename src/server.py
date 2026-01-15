import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
from src.separate import separate_vocals_and_accompaniment

app = Flask(__name__)

# 临时目录用于存储处理后的文件
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'temp_output')
os.makedirs(TEMP_DIR, exist_ok=True)

@app.route('/separate', methods=['POST'])
def separate_audio():
    """
    接收音频文件并分离人声和伴奏
    
    Request:
        - file: 音频文件 (multipart/form-data)
    
    Response:
        {
            "success": bool,
            "vocals_path": str,
            "accompaniment_path": str
        }
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file:
        # 保存上传的文件到临时目录
        input_file_path = os.path.join(TEMP_DIR, file.filename)
        file.save(input_file_path)
        
        try:
            # 分离人声和伴奏
            vocals_path, accompaniment_path = separate_vocals_and_accompaniment(
                input_file=input_file_path,
                output_dir=TEMP_DIR
            )
            
            # 返回结果
            return jsonify({
                "success": True,
                "vocals_path": vocals_path,
                "accompaniment_path": accompaniment_path
            })
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
        finally:
            # 清理上传的文件
            if os.path.exists(input_file_path):
                os.remove(input_file_path)

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """
    下载处理后的文件
    """
    try:
        return send_from_directory(TEMP_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"success": False, "error": "File not found"}), 404

if __name__ == '__main__':
    print("=" * 80)
    print("服务器即将启动...")
    print("=" * 80)
    print("\n使用方法:")
    print("1. 使用 curl 发送音频文件:")
    print("   curl -X POST -F 'file=@/path/to/your/audio.mp3' http://localhost:5000/separate")
    print("\n2. 使用 Python requests 发送音频文件:")
    print("   import requests")
    print("   url = 'http://localhost:5000/separate'")
    print("   files = {'file': open('your_audio_file.mp3', 'rb')}")
    print("   response = requests.post(url, files=files)")
    print("   print(response.json())")
    print("\n3. 下载分离后的文件:")
    print("   - 人声文件: http://localhost:5000/download/filename_vocals.wav")
    print("   - 伴奏文件: http://localhost:5000/download/filename_accompaniment.wav")
    print("\n" + "=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)