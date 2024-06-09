import torch
import random
import numpy as np

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

rna_base_dict ={
    0:"A",
    1:"G",
    2:"C",
    3:"U"
}

def get_optimizer(config, params):
    """Return a flax optimizer object based on `config`."""
    if config.optim.optimizer == 'Adam':
        optimizer = torch.optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=config.optim.lr, amsgrad=True, weight_decay=1e-12)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!'
        )
    return optimizer


def get_data_inverse_scaler():
    """Inverse data normalizer."""

    centered = True

    def inverse_scale_fn(seq):
        if centered:
            seq = (seq + 1.) / 2.
        return seq

    return inverse_scale_fn


def post_process(gen_seq, inverse_scaler):
    """Post process generated sequences."""
    gen_seq = inverse_scaler(gen_seq)
    gen_seq = torch.argmax(gen_seq, dim=-1)
    return gen_seq


def expand_dims(v, dims):
    return v[(...,) + (None,) * (dims - 1)]