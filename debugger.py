from gpt2_femto import generate
from utils import load_encoder_hparams_and_params

def benchmark_generate():
    prompt = "Tolkien wrote that goblins"
    n_tokens_to_generate = 20
    model_size = "124M"
    models_dir = "models"

    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)
    input_ids = encoder.encode(prompt)
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    output_text = encoder.decode(output_ids)

    print(output_text)

#if __name__ == "__main__":
#    benchmark_generate()