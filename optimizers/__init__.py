import torch
from .lr_scheduler import LR_Scheduler
from .lars import LARS
from .larc import LARC


def get_optimizer(name, model, lr, momentum, weight_decay):
    print('learning rate', lr)
    predictor_prefix = ('module.predictor', 'predictor')
    parameters = [{
        'name': 'base',
        'params': [param for name, param in model.named_parameters() if not name.startswith(predictor_prefix)],
        'lr': lr
    }, {
        'name': 'predictor',
        'params': [param for name, param in model.named_parameters() if name.startswith(predictor_prefix)],
        'lr': lr
    }]

    if name == 'lars':
        optimizer = LARS(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name == 'larc':
        optimizer = LARC(torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay),
                         trust_coefficient=0.001, clip=False)
    elif name == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adadelta':
        optimizer = torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        optimizer = torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise NotImplementedError
    return optimizer
