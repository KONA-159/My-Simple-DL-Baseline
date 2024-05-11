from configv1 import Config
from torch.optim import lr_scheduler


def get_scheduler(optimizer):
    scheduler_name = Config.hyper_parameters.learning_rate_schedule.scheduler
    scheduler_config = getattr(Config.hyper_parameters.learning_rate_schedule, scheduler_name)
    if scheduler_name == 'ReduceLROnPlateau':
        # 性能提升（通常是validation loss）小于一定次数————>学习率太大反复横跳了，缩小lr
        # mode：max检测metric是否不再增大，每次调整lr *= factor，patience为可容忍次数，verbose为触发后输出日志
        # threshold_mode：有rel和abs两种阈值计算模式，rel规则：max模式下如果超过best(1+threshold)为显著，min模式下如果低于best(1-threshold)为显著；abs规则：max模式下如果超过best+threshold为显著，min模式下如果低于best-threshold为显著
        # eps为性能提升判断标准，比最好的值提升小于eps则patience++
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=scheduler_config.factor,
                                                   patience=scheduler_config.patience, verbose=True,
                                                   eps=scheduler_config.eps)
    #     见论文
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=scheduler_config.T_max,
                                                   eta_min=scheduler_config.min_lr, last_epoch=-1)
    elif scheduler_name == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=scheduler_config.T_0,
                                                             T_mult=scheduler_config.T_mult,
                                                             eta_min=scheduler_config.min_lr, last_epoch=-1)
    else:
        scheduler=None
    return scheduler
