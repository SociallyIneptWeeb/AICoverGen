import os
import json
from src.mdx import run_mdx

def load_model_params():
    with open('mdxnet_models/model_data.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def separate_vocals_and_accompaniment(input_file, output_dir):
    """
    分离音乐文件中的人声和伴奏
    
    Args:
        input_file (str): 输入音乐文件的路径
        output_dir (str): 输出目录路径
        
    Returns:
        tuple: (人声文件路径, 伴奏文件路径)
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型参数
    model_params = load_model_params()
    
    # 使用高质量人声模型进行分离
    model_path = "mdxnet_models/UVR_MDXNET_HQ_4.onnx"
    
    # 运行分离
    vocals_path, accompaniment_path = run_mdx(
        model_params=model_params,
        output_dir=output_dir,
        model_path=model_path,
        filename=input_file,
        exclude_main=False,
        exclude_inversion=False,
        suffix='vocals',
        invert_suffix='accompaniment',
        denoise=False,
        keep_orig=True,
        m_threads=2
    )
    
    return vocals_path, accompaniment_path