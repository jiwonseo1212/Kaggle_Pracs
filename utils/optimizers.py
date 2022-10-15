from pyrsistent import v
import torch
import torch.optim as optim


def build_torch_optimizer(model, opt):
    """
    Args :
        model : the model to optimize
        opt : The dictionary of options
    Returns :
        A ``torch.optim.Optimier`` instance.
    """

    params = [p for p in model.parameters() if p.requires_grad]
    betas = [opt.adam_beta1, opt.adam_beta2]
    if opt.optim == 'sgd':
        optimizer = optim.SGD(params, lr=opt.learning_rate)
    elif opt.optim == 'adagrad':
        optimizer = optim.Adagrad(
            params,
            lr=opt.learning_rate,
            initial_accumulator_value=opt.adagrad_accumulator_init)
    elif opt.optim == 'adadelta':
        optimizer = optim.Adadelta(params, lr=opt.learning_rate)
    elif opt.optim == 'adam':
        optimizer = optim.Adam(
            params,
            lr=opt.learning_rate,
            betas=betas,
            eps=1e-9)
    else:
        raise ValueError(f'Invalid optimizer type   : {opt.optim}')

    return optimizer

