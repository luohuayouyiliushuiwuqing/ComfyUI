"""
检测模型 Architecture 是否被 ComfyUI 支持

Usage:
    python demos/check_model_arch.py models/unet/FHDR_ComfyUI-Q4_K_M.gguf
    python demos/check_model_arch.py models/checkpoints/*.safetensors
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_gguf_arch(path):
    from gguf import GGUFReader
    reader = GGUFReader(path)
    arch_bytes = reader.fields['general.architecture'].parts[-1]
    arch = bytes(arch_bytes).decode('utf-8', errors='ignore')
    return {
        'architecture': arch,
        'tensors': len(reader.tensors),
        'format': 'GGUF',
    }


def check_safetensors_arch(path):
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())

    # 通过 key 名称推断架构
    arch = 'unknown'
    if any('model.diffusion_model.double_blocks' in k for k in keys):
        arch = 'flux'
    elif any('model.diffusion_model.input_blocks' in k for k in keys):
        if any('conditioner.embedders.1' in k for k in keys):
            arch = 'sdxl'
        else:
            arch = 'sd1'
    elif any('model.diffusion_model.joint_blocks' in k for k in keys):
        arch = 'sd3'
    elif any('model.diffusion_model.blocks' in k for k in keys):
        if any('text_encoder' in k for k in keys):
            arch = 'sdxl'
        else:
            arch = 'sd1'

    return {
        'architecture': arch,
        'tensors': len(keys),
        'format': 'safetensors',
    }


SUPPORTED_ARCHS = {
    'flux': 'Flux (图像/视频生成)',
    'sd1': 'Stable Diffusion 1.x',
    'sd2': 'Stable Diffusion 2.x',
    'sdxl': 'Stable Diffusion XL',
    'sd3': 'Stable Diffusion 3',
    'ascade': 'Stable Cascade',
    'pixart': 'PixArt',
    'cosmos': 'Cosmos',
    'lumina2': 'Lumina 2',
    'wan': 'Wan (视频)',
    'hidream': 'HiDream',
    'chroma': 'Chroma',
    'ace': 'Ace',
    'omnigen2': 'OmniGen2',
    'hunyuan_image': 'Hunyuan Image',
    'qwen_image': 'Qwen Image',
    'ltxv': 'LTX Video',
    'mochi': 'Mochi',
    'stable_audio': 'Stable Audio',
    'cogvideox': 'CogVideoX',
    'ideogram4': 'Ideogram 4',
}


def check_model(path):
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext == '.gguf':
            info = check_gguf_arch(path)
        elif ext in ('.safetensors', '.ckpt', '.pt', '.bin'):
            info = check_safetensors_arch(path)
        else:
            return {'error': f'不支持的格式: {ext}'}

        info['path'] = path
        info['name'] = os.path.basename(path)
        info['size_gb'] = os.path.getsize(path) / (1024**3)
        info['supported'] = info['architecture'] in SUPPORTED_ARCHS
        info['model_type'] = SUPPORTED_ARCHS.get(info['architecture'], '未知')
        return info

    except Exception as e:
        return {'error': str(e), 'path': path}


def main():
    parser = argparse.ArgumentParser(description="检查模型是否被 ComfyUI 支持")
    parser.add_argument("paths", nargs="+", help="模型文件路径")
    args = parser.parse_args()

    for path in args.paths:
        if os.path.isdir(path):
            for f in sorted(os.listdir(path)):
                fp = os.path.join(path, f)
                if os.path.isfile(fp):
                    info = check_model(fp)
                    if 'error' in info:
                        continue
                    print_model_info(info)
        elif os.path.isfile(path):
            info = check_model(path)
            if 'error' in info:
                print(f"错误: {info['error']}")
            else:
                print_model_info(info)


def print_model_info(info):
    status = '✅' if info['supported'] else '❌'
    print(f"{status} {info['name']}")
    print(f"   Architecture: {info['architecture']}")
    print(f"   模型类型: {info['model_type']}")
    print(f"   格式: {info['format']}, Tensors: {info['tensors']}, 大小: {info['size_gb']:.1f} GB")
    print()


if __name__ == "__main__":
    main()
