import math
from tqdm import tqdm

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    max_x = max(x)
    exp_x = [math.exp(val - max_x) for val in x]
    sum_exp_x = sum(exp_x)
    return [val / sum_exp_x for val in exp_x]

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = sum(x) / len(x)
    variance = sum([(val - mean) ** 2 for val in x]) / len(x)
    return [g * (val - mean) / math.sqrt(variance + eps) + b for val in x]

def linear(x, w, b):
    return [sum(xi * wi for xi, wi in zip(x, row)) + b for row in w]

def ffn(x, c_fc, c_proj):
    return linear([gelu(val) for val in linear(x, **c_fc)], **c_proj)

def attention(q, k, v, mask):
    qk = [sum(qi * ki for qi, ki in zip(q_row, k_row)) for q_row, k_row in zip(q, zip(*k))]
    scaled_qk = [val / math.sqrt(len(q[0])) for val in qk]
    masked_scaled_qk = [sum(val, mask_val) for val, mask_val in zip(scaled_qk, mask)]
    return [sum(qki * vi for qki, vi in zip(masked_scaled_qk, v_row)) for v_row in zip(*v)]

def mha(x, c_attn, c_proj, n_head):
    x = linear(x, **c_attn)
    qkv_heads = [zip(*[iter(x)] * (len(x[0]) // n_head)) for _ in range(n_head)]
    causal_mask = [[-1e10 if j < i else 0 for j in range(len(x[0]))] for i in range(len(x[0]))]
    out_heads = [attention(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]
    return linear([val for sublist in out_heads for val in sublist], **c_proj)

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    x_ln1 = layer_norm(x, **ln_1)
    mha_output = mha(x_ln1, **attn, n_head=n_head)
    x = [xi + mhai for xi, mhai in zip(x, mha_output)]
    x_ln2 = layer_norm(x, **ln_2)
    return [xi + ffni for xi, ffni in zip(x, ffn(x_ln2, **mlp))]

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    x = [wte[input_id] + wpe[input_id] for input_id in inputs]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    return linear(layer_norm(x, **ln_f), wte)

def generate(inputs, params, n_head, n_tokens_to_generate):
    input_ids, wte, wpe, blocks, ln_f = params.values()
    for _ in tqdm(range(n_tokens_to_generate), "generating"):
        logits = gpt2(inputs, wte, wpe, blocks, ln_f, n_head)
        next_id = logits[-1].index(max(logits[-1]))
        inputs.append(next_id)
    return inputs[len(inputs) - n_tokens_to_generate :]

def main(prompt: str, n_tokens_to_generate: int = 40, model_size: str = "124M", models_dir: str = "models"):
    from utils import load_encoder_hparams_and_params
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)
    return output_text

if __name__ == "__main__":
    import fire
    fire.Fire(main)