"""
Pure Python Flux Generation - using ComfyUI internal modules
No UI, no server required.

Usage:
    conda activate comfyui
    python demos/pure_flux_generate.py --prompt "a beautiful sunset"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import folder_paths
import comfy.sd
import comfy.model_management
import comfy.utils
import comfy.model_sampling
import comfy.samplers
import comfy.sample
import nodes

from comfy.cli_args import args as comfy_args
comfy_args.cuda_device = None

import importlib
gguf_ops = importlib.import_module("custom_nodes.ComfyUI-GGUF.ops")
gguf_loader = importlib.import_module("custom_nodes.ComfyUI-GGUF.loader")
GGMLOps = gguf_ops.GGMLOps
gguf_sd_loader = gguf_loader.gguf_sd_loader


def encode_prompt(clip, text):
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]


def run_flux_workflow(
    prompt_text,
    negative_text="",
    width=1024,
    height=1024,
    steps=20,
    cfg=1.0,
    seed=None,
    output_path="output/pure_flux.png",
):
    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    print(f"Prompt: {prompt_text}")
    print(f"Size: {width}x{height}, Steps: {steps}, CFG: {cfg}, Seed: {seed}")
    print()

    # [1] Load Flux GGUF model
    print("[1/6] Loading Flux model (Q8, 12GB)...")
    t0 = time.time()
    unet_path = folder_paths.get_full_path("unet", "FHDR_ComfyUI-Q8_0.gguf")
    sd, extra = gguf_sd_loader(unet_path)
    ops = GGMLOps()
    model = comfy.sd.load_diffusion_model_state_dict(
        sd, model_options={"custom_operations": ops}
    )
    print(f"      Loaded in {time.time()-t0:.1f}s")

    # [2] Load T5 text encoder
    print("[2/6] Loading T5-XXL FP16 text encoder...")
    t0 = time.time()
    clip_path = folder_paths.get_full_path("text_encoders", "t5xxl_fp16.safetensors")
    clip_sd = comfy.utils.load_torch_file(clip_path, safe_load=True)
    clip = comfy.sd.load_text_encoder_state_dicts(
        clip_type=comfy.sd.CLIPType.FLUX,
        state_dicts=[clip_sd],
        model_options={"custom_operations": GGMLOps},
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
    )
    print(f"      Loaded in {time.time()-t0:.1f}s")

    # [3] Load VAE
    print("[3/6] Loading VAE...")
    t0 = time.time()
    vae_loader = nodes.VAELoader()
    vae, = vae_loader.load_vae("ae.safetensors")
    print(f"      Loaded in {time.time()-t0:.1f}s")

    # [4] Encode prompts
    print("[4/6] Encoding prompts...")
    positive = encode_prompt(clip, prompt_text)
    negative = encode_prompt(clip, negative_text)

    # [5] Create latent and sample
    print("[5/6] Sampling...")
    t0 = time.time()
    latent_image = torch.zeros([1, 16, height // 8, width // 8], device="cpu")
    noise = comfy.sample.prepare_noise(latent_image, seed)

    samples = comfy.sample.sample(
        model=model,
        noise=noise,
        steps=steps,
        cfg=cfg,
        sampler_name="euler",
        scheduler="simple",
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        denoise=1.0,
        seed=seed,
    )
    print(f"      Sampled in {time.time()-t0:.1f}s")

    out = {"samples": samples}

    # [6] Decode and save
    print("[6/6] Decoding & saving...")
    images = vae.decode(out["samples"])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    from PIL import Image
    import numpy as np

    img = images[0]
    if img.dim() == 3 and img.shape[-1] == 3:
        pass
    elif img.dim() == 3:
        img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    img = (img * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(output_path)
    print(f"      Saved: {output_path}")
    print()
    print("Done!")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Pure Python Flux Image Generation")
    parser.add_argument("--prompt", "-p", default="masterpiece best quality, a beautiful sunset over the ocean, golden hour, photorealistic, 8k")
    parser.add_argument("--negative", "-n", default="")
    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)
    parser.add_argument("--steps", "-s", type=int, default=20)
    parser.add_argument("--cfg", "-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", "-o", default="output/pure_flux.png")
    args = parser.parse_args()

    run_flux_workflow(
        prompt_text=args.prompt,
        negative_text=args.negative,
        width=args.width,
        height=args.height,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
