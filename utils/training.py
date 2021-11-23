from torch.optim.optimizer import Optimizer

class StepLRWithWarmUP():
    def __init__(self, optimizer, warmup_size, step_size, min_lr, gamma):
        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.lr_start = self.get_lr()
        self.warmup_size = warmup_size
        self.step_size = step_size
        self.min_lr = min_lr
        self.gamma = gamma
        self.iter_cnt = 0

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _set_lr(self, lr):
        for param_group in  self.optimizer.param_groups:
            param_group['lr'] = lr
        pass

    def step(self):
        if self.iter_cnt < self.warmup_size:
            lr = self.lr_start * pow(self.iter_cnt * 1. / (self.warmup_size), 4)
        elif self.iter_cnt % self.step_size == 0:
            lr = self.get_lr()*self.gamma
            lr = max(lr, self.min_lr)
        else:
            lr = self.get_lr()
        self._set_lr(lr)
