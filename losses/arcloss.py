import torch
import torch.nn as nn
import torch.nn.functional as F

class ArcLoss(nn.Module):
    def __init__(self, embedding_size, num_classes, use_cuda=False, scale=64, margin=0.4, checkpoint_path=None):
        super(ArcLoss, self).__init__()
        self.weights = nn.parameter.Parameter(torch.normal(0, 0.01, (num_classes, embedding_size), requires_grad=True))

        if use_cuda:
            self.weights.cuda()

        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes

        if checkpoint_path is not None:
            checkpoint_arc = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint_arc)

    def forward(self, features, labels):
        norm_features = F.normalize(features).cuda()
        norm_weights = F.normalize(self.weights).cuda()
        cos_theta = F.linear(norm_features, norm_weights).clamp(-1, 1)
        margin_tensor = F.one_hot(labels, num_classes=self.num_classes) * self.margin
        cos_theta = cos_theta - margin_tensor

        return cos_theta * self.scale




