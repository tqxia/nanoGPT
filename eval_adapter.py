"""Likelihood-only nanoGPT backend for lm-eval 0.4.13.

Run ``python eval_adapter.py run --model nanogpt ...`` to register the backend
without modifying lm-eval. Training has no dependency on this module.
"""
import hashlib
import os
from pathlib import Path

import torch
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers import AutoTokenizer, GPT2Config
from transformers.modeling_outputs import CausalLMOutput

from model import GPT, GPTConfig


class _NanoGPTModel(torch.nn.Module):
    """Expose only the model attributes used by HFLM's likelihood evaluator."""

    def __init__(self, gpt, tokenizer):
        super().__init__()
        self.gpt = gpt
        c = gpt.config
        self.config = GPT2Config(
            vocab_size=c.vocab_size, n_positions=c.block_size,
            n_embd=c.n_embd, n_layer=c.n_layer, n_head=c.n_head,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            activation_function='gelu_new' if c.gelu_approximate == 'tanh' else 'gelu',
        )

    @property
    def device(self):
        return next(self.gpt.parameters()).device

    def tie_weights(self):
        # nanoGPT ties these in its constructor, before loading the checkpoint.
        assert self.gpt.lm_head.weight is self.gpt.transformer.wte.weight

    def forward(self, input_ids):
        # HFLM right-pads likelihood batches and excludes padded positions from
        # scoring. Causal attention prevents later padding from affecting text.
        logits, _ = self.gpt(input_ids, return_all_logits=True)
        return CausalLMOutput(logits=logits)


@register_model('nanogpt')
class NanoGPTLM(HFLM):
    """Reuse HFLM's likelihood logic with native nanoGPT weights and forward pass.

    Tokenizer is required: checkpoints do not identify their token-to-ID mapping.
    A matching vocabulary size alone does not prove tokenizer compatibility.
    """

    def __init__(self, checkpoint, tokenizer, device='cuda:0', dtype='float32',
                 batch_size=4, max_length=None, attention='sdpa', **kwargs):
        if kwargs:
            raise TypeError(f'Unsupported nanoGPT model arguments: {sorted(kwargs)}')
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise ValueError('This adapter supports single-device evaluation only.')
        dtypes = {'float32': torch.float32, 'bfloat16': torch.bfloat16,
                  'float16': torch.float16}
        if dtype not in dtypes:
            raise ValueError(f'dtype must be one of {list(dtypes)}')
        if attention not in ('sdpa', 'eager'):
            raise ValueError('attention must be sdpa or eager')
        path = Path(checkpoint).expanduser().resolve()
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config = GPTConfig(**payload['model_args'])
        if config.gelu_approximate not in ('none', 'tanh'):
            raise ValueError('Unknown GELU approximation in checkpoint')
        context_length = config.block_size if max_length is None else int(max_length)
        if not 1 <= context_length <= config.block_size:
            raise ValueError(f'max_length must be between 1 and {config.block_size}')
        tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
        if tok.eos_token_id is None:
            raise ValueError('Tokenizer must define an EOS token for document scoring')
        if max(tok.get_vocab().values()) >= config.vocab_size:
            raise ValueError('Tokenizer contains IDs outside the checkpoint vocabulary')
        # Reuse an existing token: adding a pad token would change vocabulary.
        tok.pad_token = tok.eos_token
        gpt = GPT(config)
        state = {}
        for key, value in payload['model'].items():
            name = key.removeprefix('_orig_mod.')
            if name in state:
                raise ValueError(f'Duplicate checkpoint key after prefix removal: {name}')
            state[name] = value
        gpt.load_state_dict(state, strict=True)
        del payload, state
        if attention == 'eager':
            for block in gpt.transformer.h:
                block.attn.flash = False
                if not hasattr(block.attn, 'bias'):
                    block.attn.register_buffer(
                        'bias', torch.tril(torch.ones(config.block_size, config.block_size))
                        .view(1, 1, config.block_size, config.block_size), persistent=False,
                    )
        gpt.to(device=device, dtype=dtypes[dtype]).eval()
        wrapper = _NanoGPTModel(gpt, tok)
        super().__init__(pretrained=wrapper, tokenizer=tok, backend='causal',
                         device=device, dtype=dtype, batch_size=batch_size,
                         max_length=context_length, add_bos_token=False)
        self.pretrained = str(path)
        self.checkpoint_path = str(path)
        self.attention = attention
        with path.open('rb') as stream:
            self.checkpoint_sha256 = hashlib.file_digest(stream, 'sha256').hexdigest()

    @torch.inference_mode()
    def _model_call(self, inps, attn_mask=None, labels=None):
        if attn_mask is not None or labels is not None:
            raise ValueError('Only causal, right-padded likelihood batches are supported')
        return self.model(inps).logits

    def generate_until(self, requests, disable_tqdm=False):
        raise NotImplementedError(
            'This adapter supports likelihood tasks (WikiText, PIQA, HellaSwag), '
            'not generation tasks yet.'
        )

    def get_model_info(self):
        return {
            'model_num_parameters': sum(p.numel() for p in self.model.parameters()),
            'model_dtype': str(next(self.model.parameters()).dtype),
            'checkpoint': self.checkpoint_path,
            'checkpoint_sha256': self.checkpoint_sha256,
            'tokenizer': self.tokenizer.name_or_path,
            'attention': self.attention,
            'gelu_approximate': self.model.gpt.config.gelu_approximate,
        }


if __name__ == '__main__':
    from lm_eval.__main__ import cli_evaluate
    cli_evaluate()
