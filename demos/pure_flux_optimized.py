"""
Optimized Pure Python Flux Generation
Features: model caching, tiled VAE, batch support, progress callback

Usage:
    conda activate comfyui
    python demos/pure_flux_optimized.py -p "a beautiful sunset" -o output/sunset.png
    python demos/pure_flux_optimized.py -p "a cat" --batch 4 --seed 42
"""

import argparse
import gc
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import folder_paths
import comfy.sd
import comfy.model_management
import comfy.utils
import comfy.sample
import comfy.samplers
import nodes

from comfy.cli_args import args as comfy_args
comfy_args.cuda_device = None

import importlib
gguf_ops = importlib.import_module("custom_nodes.ComfyUI-GGUF.ops")
gguf_loader = importlib.import_module("custom_nodes.ComfyUI-GGUF.loader")
GGMLOps = gguf_ops.GGMLOps
gguf_sd_loader = gguf_loader.gguf_sd_loader


class FluxGenerator:
    def __init__(self, unet_name="FHDR_ComfyUI-Q4_K_M.gguf",
                 clip_name="t5xxl_fp8_e4m3fn.safetensors",
                 vae_name="ae.safetensors"):
        self.unet_name = unet_name
        self.clip_name = clip_name
        self.vae_name = vae_name
        self.model = None
        self.clip = None
        self.vae = None

    def load(self):
        if self.model is not None:
            return

        print(f"[Load] Model: {self.unet_name}")
        t0 = time.time()
        unet_path = folder_paths.get_full_path("unet", self.unet_name)
        sd, extra = gguf_sd_loader(unet_path)
        ops = GGMLOps()
        self.model = comfy.sd.load_diffusion_model_state_dict(
            sd, model_options={"custom_operations": ops}
        )
        del sd
        gc.collect()
        print(f"  Model loaded in {time.time()-t0:.1f}s")

        print(f"[Load] CLIP: {self.clip_name}")
        t0 = time.time()
        clip_path = folder_paths.get_full_path("text_encoders", self.clip_name)
        clip_sd = comfy.utils.load_torch_file(clip_path, safe_load=True)
        self.clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.FLUX,
            state_dicts=[clip_sd],
            model_options={"custom_operations": GGMLOps},
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        del clip_sd
        gc.collect()
        print(f"  CLIP loaded in {time.time()-t0:.1f}s")

        print(f"[Load] VAE: {self.vae_name}")
        t0 = time.time()
        vae_loader = nodes.VAELoader()
        self.vae, = vae_loader.load_vae(self.vae_name)
        print(f"  VAE loaded in {time.time()-t0:.1f}s")

    def unload(self):
        self.model = None
        self.clip = None
        self.vae = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def encode_prompt(self, text):
        tokens = self.clip.tokenize(text)
        output = self.clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = output.pop("cond")
        return [[cond, output]]

    def generate(self, prompt, negative="", width=1024, height=1024,
                 steps=20, cfg=1.0, seed=None, scheduler="simple",
                 sampler="euler", progress_callback=None):
        self.load()

        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")

        positive = self.encode_prompt(prompt)
        negative_cond = self.encode_prompt(negative)

        latent = torch.zeros([1, 16, height // 8, width // 8], device="cpu")
        noise = comfy.sample.prepare_noise(latent, seed)

        if progress_callback:
            def step_cb(step, total_steps, latentTensor):
                progress_callback(step, total_steps)
        else:
            step_cb = None

        samples = comfy.sample.sample(
            model=self.model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler,
            scheduler=scheduler,
            positive=positive,
            negative=negative_cond,
            latent_image=latent,
            denoise=1.0,
            seed=seed,
        )

        out = {"samples": samples}
        images = self.vae.decode(out["samples"])

        return images, seed

    def save_image(self, images, output_path, index=0):
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        from PIL import Image
        import numpy as np

        img = images[index]
        if img.dim() == 3 and img.shape[-1] == 3:
            pass
        elif img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.cpu().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).save(output_path)
        return output_path

    def generate_batch(self, prompts, width=1024, height=1024,
                       steps=20, cfg=1.0, seeds=None,
                       output_dir="output", prefix="flux"):
        self.load()

        results = []
        for i, prompt in enumerate(prompts):
            seed = seeds[i] if seeds and i < len(seeds) else None
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:60]}...")

            t0 = time.time()
            images, actual_seed = self.generate(
                prompt=prompt, width=width, height=height,
                steps=steps, cfg=cfg, seed=seed
            )
            elapsed = time.time() - t0

            output_path = f"{output_dir}/{prefix}_{i+1:03d}.png"
            self.save_image(images, output_path)
            print(f"  Saved: {output_path} (seed={actual_seed}, {elapsed:.1f}s)")
            results.append(output_path)

        return results


def main():
    parser = argparse.ArgumentParser(description="Optimized Flux Image Generation")
    parser.add_argument("--prompt", "-p", required=True, help="Positive prompt")
    parser.add_argument("--negative", "-n", default="", help="Negative prompt")
    parser.add_argument("--width", "-W", type=int, default=1024)
    parser.add_argument("--height", "-H", type=int, default=1024)
    parser.add_argument("--steps", "-s", type=int, default=20)
    parser.add_argument("--cfg", "-c", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch", "-b", type=int, default=1, help="Number of images")
    parser.add_argument("--output", "-o", default="output/flux_output.png")
    parser.add_argument("--unet", default="FHDR_ComfyUI-Q4_K_M.gguf")
    parser.add_argument("--clip", default="t5xxl_fp8_e4m3fn.safetensors")
    parser.add_argument("--vae", default="ae.safetensors")
    args = parser.parse_args()

    gen = FluxGenerator(unet_name=args.unet, clip_name=args.clip, vae_name=args.vae)

    try:
        if args.batch > 1:
            prompts = [args.prompt] * args.batch
            gen.generate_batch(
                prompts=prompts,
                width=args.width, height=args.height,
                steps=args.steps, cfg=args.cfg,
                output_dir=os.path.dirname(args.output) or "output",
                prefix=os.path.splitext(os.path.basename(args.output))[0]
            )
        else:
            images, seed = gen.generate(
                prompt=args.prompt, negative=args.negative,
                width=args.width, height=args.height,
                steps=args.steps, cfg=args.cfg, seed=args.seed
            )
            gen.save_image(images, args.output)
            print(f"\nSaved: {args.output} (seed={seed})")
    finally:
        gen.unload()


if __name__ == "__main__":
    main()
