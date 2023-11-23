import torch
from transformers import (
    AutoConfig,
    AutoModel,
    LongT5ForConditionalGeneration,
    AutoTokenizer,
    GenerationConfig
)


def get_model(model_type, tokenizer_type, **kwargs):
    if model_type == 'google/long-t5-tglobal-base':
        config = AutoConfig.from_pretrained(model_type)
        config.share_enc_dec_embeddings = kwargs.get("share_enc_dec_embeddings", False)
        config.decode_vocab_size = kwargs.get("decode_vocab_size", None)
        config.use_pawa_decoder = kwargs.get("use_pawa_decoder", None)
        config.pawa_decoder_heads = kwargs.get("pawa_decoder_heads", None)
        config.pawa_num_layers = kwargs.get("pawa_num_layers", None)
        
        logit_mask_fn = kwargs.get("logit_mask_fn", None)
        max_out_length = kwargs.get("max_out_length", None)
        kwargs_dct = {
            "logit_mask_fn": logit_mask_fn,
            "max_out_length": max_out_length
        }
        gen_config = GenerationConfig.from_pretrained(model_type)
        model = LongT5ForConditionalGeneration(config, **kwargs_dct)
        pretrained_model =  AutoModel.from_pretrained(model_type)
        pretrained_params = dict(pretrained_model.named_parameters())

        for name, param in model.named_parameters():
            if config.share_enc_dec_embeddings:
                with torch.no_grad():
                    param.copy_(pretrained_params[name])
            else:
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrained_params[name])

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

        return model, tokenizer, gen_config, config