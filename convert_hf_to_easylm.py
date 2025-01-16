"""
Usage:
python convert_hf_to_easylm.py  \
       --checkpoint_dir     /path/hf_format_dir/    \
       --output_file /path/easylm_format.stream   \
       --model_size 7b \
       --streaming
"""
import time
from pathlib import Path
import argparse

import mlxu
import torch
import flax

from EasyLM.checkpoint import StreamingCheckpointer

LLAMA_STANDARD_CONFIGS = {
    '1b': {
        'dim': 2048,
        'intermediate_size': 8192,
        'n_layers': 16,
        'n_heads': 32,
        'norm_eps': 1e-5,
    },
    '3b32-rm': {
        'dim': 3072,  
        'intermediate_size': 8192,  
        'n_layers': 28,  
        'n_heads': 24,  
        'n_kv_heads': 8,  
        'norm_eps': 1e-5,  
        'vocab_size': 128256,  
        'rope_theta': 500000,  
        'max_position_embeddings': 131072,  
        'rope_scaling': {  
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
    },
    '1b32': {
        'dim': 2048,
        'intermediate_size': 8192,
        'n_layers': 16,
        'n_heads': 32,
        'n_kv_heads': 8,
        'norm_eps': 1e-5,
        'vocab_size': 128256,
        'rope_theta': 500000,
        'max_position_embeddings': 131072,
        'rope_scaling': {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
    },
    '3b': {
        'dim': 3200,
        'intermediate_size': 8640,
        'n_layers': 26,
        'n_heads': 32,
        'norm_eps': 1e-6,
    },
    "7b": {
        "dim": 4096,
        "intermediate_size": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "norm_eps": 1e-6,
    },
    "13b": {
        "dim": 5120,
        "intermediate_size": 13824,
        "n_layers": 40,
        "n_heads": 40,
        "norm_eps": 1e-6,
    },
    "30b": {
        "dim": 6656,
        "intermediate_size": 17920,
        "n_layers": 60,
        "n_heads": 52,
        "norm_eps": 1e-6,
    },
    "65b": {
        "dim": 8192,
        "intermediate_size": 22016,
        "n_layers": 80,
        "n_heads": 64,
        "norm_eps": 1e-5,
    },
    "70b": {
        "dim": 8192,
        "intermediate_size": 28672,
        "n_layers": 80,
        "n_heads": 64,
        "n_kv_heads": 8,
        "norm_eps": 1e-5,
    },
    '8b31': {
        'dim': 4096,
        'intermediate_size': 14336,
        'n_layers': 32,
        'n_heads': 32,
        'n_kv_heads': 8,
        'norm_eps': 1e-6,
        'vocab_size': 128256,
        'rope_theta': 500000,
        'max_position_embeddings': 131072,
        'rope_scaling': {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3"
        },
    },
}


def inverse_permute(w, n_heads, input_dim, output_dim):
    reshaped_w = w.reshape(n_heads, 2, output_dim // n_heads // 2, input_dim)
    transposed_w = reshaped_w.transpose(0, 2, 1, 3)
    inverted_w = transposed_w.reshape(output_dim, input_dim)
    return inverted_w


def main(args):
    start = time.time()
    params = LLAMA_STANDARD_CONFIGS[args.model_size]

    ckpt = {}
    if args.use_safetensors:
        from safetensors import safe_open
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.safetensors"))
        for i, ckpt_path in enumerate(ckpt_paths):
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model."):
                        k = key[6:]
                        ckpt[k] = f.get_tensor(key)
                    else:
                        ckpt[key] = f.get_tensor(key)
                    
    else:
        ckpt_paths = sorted(Path(args.checkpoint_dir).glob("*.bin"))
        for i, ckpt_path in enumerate(ckpt_paths):
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            for k, v in checkpoint.items():
                if k.startswith("model."):
                    k = k[6:]
                ckpt[k] = v
    print(f"Start convert weight to easylm format...")
    jax_weights = {
        "transformer": {
            "wte": {"embedding": ckpt["embed_tokens.weight"].to(torch.float16).numpy()[:-8, :]},
            "ln_f": {"kernel": ckpt["norm.weight"].to(torch.float16).numpy()},
            "h": {
                "%d"
                % (layer): {
                    "attention": {
                        "wq": {
                            "kernel": inverse_permute(
                                ckpt[f"layers.{layer}.self_attn.q_proj.weight"].to(torch.float16).numpy(),
                                n_heads=params["n_heads"],
                                input_dim=params["dim"],
                                output_dim=params["dim"],
                            ).transpose()
                        },
                        "wk": {
                            "kernel": inverse_permute(
                                ckpt[f"layers.{layer}.self_attn.k_proj.weight"].to(torch.float16).numpy(),
                                n_heads=params.get("n_kv_heads", params["n_heads"]),
                                input_dim=params["dim"],
                                output_dim=(params["dim"] // (params["n_heads"] // params.get("n_kv_heads", params["n_heads"]))),
                            ).transpose()
                        },
                        "wv": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.v_proj.weight"]
                            .to(torch.float16)
                            .numpy()
                            .transpose()
                        },
                        "wo": {
                            "kernel": ckpt[f"layers.{layer}.self_attn.o_proj.weight"]
                            .to(torch.float16)
                            .numpy()
                            .transpose()
                        },
                    },
                    "feed_forward": {
                        "w1": {
                            "kernel": ckpt[f"layers.{layer}.mlp.gate_proj.weight"]
                            .to(torch.float16)
                            .numpy()
                            .transpose()
                        },
                        "w2": {
                            "kernel": ckpt[f"layers.{layer}.mlp.down_proj.weight"]
                            .to(torch.float16)
                            .numpy()
                            .transpose()
                        },
                        "w3": {
                            "kernel": ckpt[f"layers.{layer}.mlp.up_proj.weight"]
                            .to(torch.float16)
                            .numpy()
                            .transpose()
                        },
                    },
                    "attention_norm": {
                        "kernel": ckpt[f"layers.{layer}.input_layernorm.weight"].to(torch.float16).numpy()
                    },
                    "ffn_norm": {
                        "kernel": ckpt[
                            f"layers.{layer}.post_attention_layernorm.weight"
                        ].to(torch.float16).numpy()
                    },
                }
                for layer in range(params["n_layers"])
            },
        },
        # "lm_head": {"kernel": ckpt["lm_head.weight"].to(torch.float16).numpy().transpose()[:, :-8]},
        "lm_head": {"kernel": ckpt.get("lm_head.weight", ckpt["embed_tokens.weight"]).to(torch.float16).numpy().transpose()[:, :-8]},
    }
    if args.model_size.endswith('rm'):
        print('Adding rm score head')
        jax_weights['score'] = ckpt[f"score.weight"].to(torch.float16).numpy().transpose()
    print(f"Convert weight to easylm format finished...")
    print(f"Start to save...")

    if args.streaming:
        StreamingCheckpointer.save_train_state_to_file(jax_weights, args.output_file)
    else:
        with mlxu.open_file(args.output_file, "wb") as fout:
            fout.write(flax.serialization.msgpack_serialize(jax_weights, in_place=True))

    print(
        f"Save finished!!! take time: {time.time() - start} save path: {args.output_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="hf to easylm format script")

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Need to be converted model weight dir. it is a dir",
    )
    parser.add_argument(
        "--output_file", type=str, help="Save model weight file path, it is a file."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="7b",
        choices=list(LLAMA_STANDARD_CONFIGS.keys()),
        help="model size",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        default=True,
        help="whether is model weight saved stream format",
    )
    parser.add_argument(
        '--use_safetensors',
        action='store_true',
        help='Load SafeTensors for model weights',
    )

    args = parser.parse_args()

    print(f"checkpoint_dir: {args.checkpoint_dir}")
    print(f"output_file: {args.output_file}")
    print(f"model_size: {args.model_size}")
    print(f"streaming: {args.streaming}")

    main(args)
