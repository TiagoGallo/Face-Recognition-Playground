from .convnext import ConvNeXt
import torch

def load_model(model_arch='ConvNext', **kwargs):
    if model_arch == 'ConvNext':
        return load_ConvNext(**kwargs)   
        
    
def load_ConvNext(pretrained='ImageNet', checkpoint_path=None, features_dim=512, **kwargs):
    '''
        Load the model
    '''
    model_convNextB = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=features_dim)
    print(checkpoint_path)
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        
        if pretrained == 'ImageNet':
            checkpoint['model'].pop('head.weight', None)
            checkpoint['model'].pop('head.bias', None)
            checkpoint = checkpoint['model']
    
        model_convNextB.load_state_dict(checkpoint, strict=False)
    
    model_convNextB.cuda()
    model_convNextB.train()

    return model_convNextB