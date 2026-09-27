"""GPU integration checks against the same public GPT-2 weights in HF and nanoGPT."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import tempfile

import torch
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM

from eval_adapter import NanoGPTLM
from model import GPT, GPTConfig


def requests(kind, arguments):
    return [Instance(request_type=kind, doc={}, arguments=args, idx=i)
            for i, args in enumerate(arguments)]


def run(args):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Exercise old checkpoint defaults, compile prefixes, and unpadded training API.
    with tempfile.TemporaryDirectory() as directory:
        c = GPTConfig(n_layer=2, n_embd=32, n_head=2, block_size=64,
                      vocab_size=50304, bias=False)
        tiny = GPT(c).eval()
        x = torch.randint(0, c.vocab_size, (2, 8))
        y = torch.randint(0, c.vocab_size, (2, 8))
        with torch.no_grad():
            last, no_loss = tiny(x)
            full, full_no_loss = tiny(x, return_all_logits=True)
            training, loss = tiny(x, y)
        assert no_loss is None and full_no_loss is None
        torch.testing.assert_close(last, full[:, [-1], :])
        torch.testing.assert_close(full, training)
        torch.testing.assert_close(loss, torch.nn.functional.cross_entropy(
            full.reshape(-1, c.vocab_size), y.reshape(-1)))
        old_args = asdict(c)
        old_args.pop('gelu_approximate')
        path = Path(directory) / 'legacy.pt'
        torch.save({'model_args': old_args,
                    'model': {'_orig_mod.' + k: v for k, v in tiny.state_dict().items()}}, path)
        legacy = NanoGPTLM(str(path), args.tokenizer, device='cpu')
        torch.testing.assert_close(legacy._model_call(x), full)
        try:
            legacy.generate_until([])
        except NotImplementedError:
            pass
        else:
            raise AssertionError('Generation must fail explicitly')
        for overrides in ({'max_length': 65},):
            try:
                NanoGPTLM(str(path), args.tokenizer, device='cpu', **overrides)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid context length accepted')
        bad_args = dict(old_args, vocab_size=2)
        torch.save({'model_args': bad_args, 'model': {}}, Path(directory) / 'bad.pt')
        try:
            NanoGPTLM(str(Path(directory) / 'bad.pt'), args.tokenizer, device='cpu')
        except ValueError as error:
            assert 'vocabulary' in str(error)
        else:
            raise AssertionError('Out-of-range tokenizer accepted')
        del legacy, tiny

    nano = NanoGPTLM(args.checkpoint, args.tokenizer, device='cuda:0',
                    dtype='float32', batch_size=4, max_length=64, attention='eager')
    hf = HFLM(pretrained=args.tokenizer, tokenizer=args.tokenizer, device='cuda:0',
              dtype='float32', batch_size=4, max_length=64, attn_implementation='eager')
    texts = ['Hello, world!', 'The capital of France is Paris.',
             'Whitespace:\n\n  one\ttwo. Unicode: café 中文 🐈.']
    max_logit_error = 0.0
    for text in texts:
        assert nano.tok_encode(text) == hf.tok_encode(text)
        ids = torch.tensor([nano.tok_encode(text)], device='cuda:0')
        a, b = nano._model_call(ids), hf._model_call(ids)
        max_logit_error = max(max_logit_error, (a-b).abs().max().item())
        torch.testing.assert_close(a, b, atol=5e-4, rtol=5e-5)

    pairs = [('The capital of France is', ' Paris.'), ('', 'Hello, world!'),
             ('I like ', 'cats.'), ('A much longer context about a dog.', ' It ran home.')]
    batch_requests = requests('loglikelihood', pairs)
    actual = nano.loglikelihood(batch_requests, disable_tqdm=True)
    expected = hf.loglikelihood(batch_requests, disable_tqdm=True)
    torch.testing.assert_close(torch.tensor([x[0] for x in actual]),
                               torch.tensor([x[0] for x in expected]), atol=2e-3, rtol=1e-5)
    assert [x[1] for x in actual] == [x[1] for x in expected]
    nano.batch_size_per_gpu = 1
    individual = nano.loglikelihood(batch_requests, disable_tqdm=True)
    torch.testing.assert_close(torch.tensor([x[0] for x in actual]),
                               torch.tensor([x[0] for x in individual]), atol=2e-3, rtol=1e-5)
    nano.batch_size_per_gpu = 4
    # Boundaries on both sides of the 64-token evaluation window, plus empty text.
    docs = ['', ' hello' * 63, ' hello' * 64, ' hello' * 65, ' hello' * 129]
    rolling = requests('loglikelihood_rolling', [(text,) for text in docs])
    actual_rolling = nano.loglikelihood_rolling(rolling)
    expected_rolling = hf.loglikelihood_rolling(rolling)
    torch.testing.assert_close(torch.tensor(actual_rolling), torch.tensor(expected_rolling),
                               atol=3e-3, rtol=1e-5)
    assert actual_rolling[0] == 0.0
    result = {'passed': True, 'max_logit_absolute_error': max_logit_error,
              'continuation_max_logprob_error': max(abs(a[0]-b[0]) for a,b in zip(actual,expected)),
              'rolling_max_logprob_error': max(abs(a-b) for a,b in zip(actual_rolling,expected_rolling)),
              'checks': ['full logits and legacy forward behavior', 'legacy bias=False checkpoint',
                         'compiled checkpoint prefix', 'vocabulary and context guards',
                         'generation explicitly unsupported', 'HF GPT-2 logits/tokenization',
                         'continuation-only scores', 'batched versus individual scores',
                         'rolling scores at context boundaries and empty document']}
    Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    run(parser.parse_args())
