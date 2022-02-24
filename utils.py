import torch
import os 

class AvgMeter():
    def __init__(self, writer=None, name=None, num_iter_per_epoch=None, per_iter_vis=False):
        self.writer = writer
        self.name = name
        self.num_iter_per_epoch = num_iter_per_epoch
        self.per_iter_vis = per_iter_vis
    #=================================================================
    def reset(self, epoch):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.epoch = epoch
    #=================================================================
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count if self.count !=0 else 0

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def save_model(model, arc_critereon, save_dir, global_step, model_arch):
    torch.save(model.state_dict(), os.path.join(save_dir, f'{model_arch}_step_{global_step}.pth'))
    torch.save(arc_critereon.state_dict(), os.path.join(save_dir, f'{model_arch}_arcLossWeights_step_{global_step}.pth'))