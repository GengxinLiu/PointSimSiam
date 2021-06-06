import numpy as np


class LR_Scheduler(object):
    def __init__(self, optimizer, scheduler='auto',
                 base_lr=None, warmup_epochs=None, warmup_lr=None,
                 num_epochs=None, final_lr=None, iter_per_epoch=None,
                 step_decay_lr=None, step_epochs=None,
                 constant_predictor_lr=False):
        """
        :param optimizer:
        :param warmup_epochs: 学习率衰减的epoch.
        :param warmup_lr:
        :param step_decay_lr: default 0.7, step deccay
        :param step_epochs: decay learning rate after step_epochs.
        """
        self.base_lr = base_lr
        self.optimizer = optimizer
        self.iter = 0
        self.constant_predictor_lr = constant_predictor_lr  # 设置Predictor的学习率为常量
        self.scheduler = scheduler
        if scheduler == 'warmup_cos':
            # 先预热warmup_epochs个epoch，然后进行cos衰减
            warmup_iter = iter_per_epoch * warmup_epochs
            warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
            decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
            cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
            self.lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
        elif scheduler == 'warmup_step':
            # step decay
            # 先预热warmup_epochs个epoch，然后进行cos衰减
            warmup_iter = iter_per_epoch * warmup_epochs
            warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
            step_lr_schedule = []
            for i in range(warmup_epochs, num_epochs):
                step_lr_schedule.extend([self.base_lr * step_decay_lr ** (i // step_epochs)] * iter_per_epoch)
            self.lr_schedule = np.concatenate((warmup_lr_schedule, step_lr_schedule))
        elif scheduler == 'cos':
            # 直接进行cos衰减
            decay_iter = iter_per_epoch * num_epochs
            cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
                    1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
            self.lr_schedule = cosine_lr_schedule
        elif scheduler == 'step':
            step_lr_schedule = []
            for i in range(0, num_epochs):
                step_lr_schedule.extend([self.base_lr * step_decay_lr ** (i // step_epochs)] * iter_per_epoch)
            self.lr_schedule = step_lr_schedule

    def step(self):
        if self.scheduler == 'auto':
            # 优化器自行作衰减
            lr = self.optimizer.param_groups[0]['lr']
        else:
            # 自定义衰减策略
            for param_group in self.optimizer.param_groups:
                if self.constant_predictor_lr and param_group['name'] == 'predictor':
                    param_group['lr'] = self.base_lr
                else:
                    lr = param_group['lr'] = self.lr_schedule[self.iter]
        self.iter += 1
        return lr


if __name__ == "__main__":
    import sys

    sys.path.append("..")
    from encoder.dgcnn import DGCNN
    from pretext import simsiam
    import matplotlib.pyplot as plt
    from optimizers import get_optimizer

    num_epochs = 200
    batch_size = 80
    n_data = 52000
    iter_per_epoch = n_data // batch_size
    base_lr = 0.001
    model = simsiam.SimSiam(
        backbone=DGCNN()
    )
    optimizer = get_optimizer('sgd', model, base_lr, 0.9, 0)

    # warmup_cos
    warmup_lr = 0  # 第一个step开始的学习率
    warmup_epochs = 10
    final_lr = 0
    warmup_lr = warmup_lr * batch_size / 256
    final_lr = final_lr * batch_size / 256

    warmup_cos = LR_Scheduler(optimizer, 'step', base_lr,
                              warmup_epochs, warmup_lr,
                              num_epochs, final_lr,
                              iter_per_epoch, step_decay_lr=0.7, step_epochs=20, constant_predictor_lr=False)
    warmup_cos_scheduler = []
    for epoch in range(num_epochs):
        for it in range(iter_per_epoch):
            lr = warmup_cos.step()
            warmup_cos_scheduler.append(lr)
    plt.plot(warmup_cos_scheduler)

    # # warmup_cos
    # warmup_lr = 0  # 第一个step开始的学习率
    # warmup_epochs = 10
    # final_lr = 0
    # warmup_lr = warmup_lr * batch_size / 256
    # final_lr = final_lr * batch_size / 256
    #
    # warmup_cos = LR_Scheduler(optimizer, 'warmup_cos', 0.05,
    #                           warmup_epochs, warmup_lr,
    #                           num_epochs, final_lr,
    #                           iter_per_epoch, constant_predictor_lr=False)
    # warmup_cos_scheduler = []
    # for epoch in range(num_epochs):
    #     for it in range(iter_per_epoch):
    #         lr = warmup_cos.step()
    #         warmup_cos_scheduler.append(lr)
    # plt.plot(warmup_cos_scheduler)
    #
    # step_decay_lr = 0.7
    # step_epochs = 21
    # warmup_step = LR_Scheduler(optimizer, 'warmup_step', base_lr,
    #                            warmup_epochs, warmup_lr,
    #                            num_epochs, final_lr,
    #                            iter_per_epoch, step_decay_lr, step_epochs, constant_predictor_lr=False)
    # warmup_step_scheduler = []
    # for epoch in range(num_epochs):
    #     for it in range(iter_per_epoch):
    #         lr = warmup_step.step()
    #         warmup_step_scheduler.append(lr)
    # plt.plot(warmup_step_scheduler)
    plt.show()
    # # warmup_cos
    # warmup_lr = 0  # 第一个step开始的学习率
    # warmup_epochs = 10
    # final_lr = 0
    # warmup_lr = warmup_lr * batch_size / 256
    # final_lr = final_lr * batch_size / 256
    # warmup_iter = iter_per_epoch * warmup_epochs
    # warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    # decay_iter = iter_per_epoch * (num_epochs - warmup_epochs)
    # cosine_lr_schedule = final_lr + 0.5 * (base_lr - final_lr) * (
    #         1 + np.cos(np.pi * np.arange(decay_iter) / decay_iter))
    # warm_cos_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    #
    # # GLR
    # step_decay_lr = 0.7
    # step_epochs = 21
    # warmup_iter = iter_per_epoch * warmup_epochs
    # warmup_lr_schedule = np.linspace(warmup_lr, base_lr, warmup_iter)
    # step_lr_schedule = []
    # for i in range(warmup_epochs, num_epochs):
    #     step_lr_schedule.extend([base_lr * step_decay_lr ** (i // step_epochs)] * iter_per_epoch)
    # glr_schedule = np.concatenate((warmup_lr_schedule, step_lr_schedule))
    # plt.plot(warm_cos_schedule, label='warm_cos')
    # plt.plot(glr_schedule, label='glr')
    # plt.legend()
    # plt.show()
