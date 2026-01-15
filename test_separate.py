"""
测试音频分离功能的脚本
"""
import requests
import json

# 服务器地址
BASE_URL = 'http://localhost:5000'

def test_separate_audio(audio_file_path):
    """
    测试音频分离功能
    
    Args:
        audio_file_path (str): 音频文件的路径
    """
    print(f"正在测试音频分离功能...")
    print(f"输入文件: {audio_file_path}")
    print()
    
    # 发送请求
    url = f'{BASE_URL}/separate'
    
    try:
        with open(audio_file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(url, files=files)
        
        # 处理响应
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("✓ 分离成功！")
                print(f"人声文件: {result['vocals_path']}")
                print(f"伴奏文件: {result['accompaniment_path']}")
                print()
                print("下载链接:")
                vocals_filename = result['vocals_path'].split('/')[-1]
                accompaniment_filename = result['accompaniment_path'].split('/')[-1]
                print(f"- 人声: {BASE_URL}/download/{vocals_filename}")
                print(f"- 伴奏: {BASE_URL}/download/{accompaniment_filename}")
            else:
                print(f"✗ 分离失败: {result['error']}")
        else:
            print(f"✗ 请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
    
    except FileNotFoundError:
        print(f"✗ 错误: 文件 '{audio_file_path}' 不存在")
    except requests.exceptions.ConnectionError:
        print(f"✗ 错误: 无法连接到服务器，请确保服务器已在 {BASE_URL} 运行")
    except Exception as e:
        print(f"✗ 发生未知错误: {e}")

if __name__ == '__main__':
    # 示例用法
    audio_file = 'path/to/your/audio_file.mp3'  # 替换为你的音频文件路径
    test_separate_audio(audio_file)