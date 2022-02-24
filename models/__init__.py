from .convnext import ConvNeXt
import torch
from .mobilefacenet import get_mbf
from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200

def load_model(model_arch='ConvNext-b', **kwargs):
    if model_arch == 'ConvNext-b':
        model = load_ConvNext_Base(**kwargs)   
    elif model_arch == 'ConvNext-T':
        model = load_ConvNext_Tiny(**kwargs)   
    elif model_arch == 'mobilefacenet':
        model = load_mobile_facenet(**kwargs)
    elif model_arch == 'iresnet18':
        model = iresnet18(num_features=kwargs['features_dim'])
    elif model_arch == 'iresnet34':
        model = iresnet34(num_features=kwargs['features_dim'])
    elif model_arch == 'iresnet50':
        model = iresnet50(num_features=kwargs['features_dim'])
    elif model_arch == 'iresnet100':
        model = iresnet100(num_features=kwargs['features_dim'])
    elif model_arch == 'iresnet200':
        model = iresnet200(num_features=kwargs['features_dim'])

    model.cuda()
    model.train()
    return model
        
def load_mobile_facenet(checkpoint_path=None, features_dim=512, **kwargs):
    '''
        Load Mobile Facenet model
    '''
    model = get_mbf(False, features_dim)

    if checkpoint_path is not None:
        checkpoint = torch.load(kwargs['checkpoint_path'])
        model.load_state_dict(checkpoint, strict=False)

    return model

def load_ConvNext_Base(pretrained='ImageNet', checkpoint_path=None, features_dim=512, **kwargs):
    '''
        Load the model
    '''
    model_convNextB = ConvNeXt(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], num_classes=features_dim)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        
        if pretrained == 'ImageNet':
            checkpoint['model'].pop('head.weight', None)
            checkpoint['model'].pop('head.bias', None)
            checkpoint = checkpoint['model']
    
        model_convNextB.load_state_dict(checkpoint, strict=False)

    return model_convNextB

def load_ConvNext_Tiny(pretrained='ImageNet', checkpoint_path=None, features_dim=512, **kwargs):
    '''
        Load the model
    '''
    model_convNextT = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], num_classes=features_dim)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        
        if pretrained == 'ImageNet':
            checkpoint['model'].pop('head.weight', None)
            checkpoint['model'].pop('head.bias', None)
            checkpoint = checkpoint['model']
    
        model_convNextT.load_state_dict(checkpoint, strict=False)


    return model_convNextT