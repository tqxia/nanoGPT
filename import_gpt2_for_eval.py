"""Import local HF GPT-2 weights as a nanoGPT checkpoint for adapter validation."""
import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel

from model import GPT, GPTConfig


def convert(source, destination):
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f'Refusing to overwrite {destination}')
    hf = GPT2LMHeadModel.from_pretrained(source, local_files_only=True,
                                       dtype=torch.float32, attn_implementation='eager')
    c = hf.config
    if c.activation_function != 'gelu_new' or c.layer_norm_epsilon != 1e-5:
        raise ValueError('This converter is for standard public GPT-2 checkpoints')
    if (c.n_inner not in (None, 4*c.n_embd) or not c.scale_attn_weights
            or c.scale_attn_by_inverse_layer_idx or c.reorder_and_upcast_attn):
        raise ValueError('Unsupported nonstandard GPT-2 attention or MLP configuration')
    config = GPTConfig(block_size=c.n_positions, vocab_size=c.vocab_size,
                       n_layer=c.n_layer, n_head=c.n_head, n_embd=c.n_embd,
                       dropout=0.0, bias=True, gelu_approximate='tanh')
    nano = GPT(config)
    expected = nano.state_dict()
    converted = {}
    transposed = ('attn.c_attn.weight', 'attn.c_proj.weight',
                  'mlp.c_fc.weight', 'mlp.c_proj.weight')
    for key, value in hf.state_dict().items():
        if key.endswith(('.attn.bias', '.attn.masked_bias')):
            continue
        if key.endswith(transposed):
            value = value.t().contiguous()
        if key not in expected or expected[key].shape != value.shape:
            raise ValueError(f'Unexpected parameter: {key}, {value.shape}')
        converted[key] = value
    nano.load_state_dict(converted, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model_args': asdict(config), 'model': nano.state_dict(),
                'source': str(source), 'purpose': 'public GPT-2 adapter validation'}, destination)
    print(f'Saved public GPT-2 weights in nanoGPT format: {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Local Hugging Face GPT-2 directory')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    convert(args.source, args.output)
