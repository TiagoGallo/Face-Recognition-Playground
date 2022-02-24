import torch
import torch.nn as nn
from tqdm import tqdm
import argparse

from losses import ArcLoss
from data import load_data
from models import load_model
from evaluation import eval_model
from utils import AvgMeter, get_lr, save_model


def train(args, num_classes, model, test_loader):
    arc_critereon = ArcLoss(args['features_dim'], num_classes, use_cuda=True, scale=args['scale'], margin=args['margin'], checkpoint_path=args['arc_checkpoint'])
    arc_critereon.train()
    critereon = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        params= [
            {"params": model.parameters(), },
            {"params": arc_critereon.parameters(), },
        ])

    scheduler =  torch.optim.lr_scheduler.OneCycleLR(optimizer, args['max_lr'], total_steps=args['epochs'] * len(train_loader)) 

    best_acc = 0
    global_step = 0
    for epoch in range(args['epochs']):
        progress_bar = tqdm(total=len(train_loader), position=epoch)
        model.train()
        
        loss_meter = AvgMeter()
        loss_meter.reset(epoch)
        for samples, labels, _ in train_loader:
            global_step += 1

            # Get everything to the GPU
            samples = samples.cuda()
            labels = labels.cuda()

            # Run model and calc loss
            features = model(samples)
            probs = arc_critereon(features, labels)
            loss = critereon(probs, labels)

            # Update model
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            # Update training stats
            loss_meter.update(loss.item())
            progress_bar.set_description(f'Epoch: {epoch} Loss: {loss_meter.avg:.5f} ({loss.item():.5f}) LR: {get_lr(optimizer):.4f} Best val acc: {best_acc*100:.2f}%')
            progress_bar.update()

            # Evaluate the model
            if global_step % args['eval_steps'] == 0:
                print('evaluating..')
                acc = eval_model(test_loader, model)

                if acc > best_acc:
                    save_model(model, arc_critereon, args['save_dir'], global_step, args['model_arch'])
                    best_acc = acc

                model.train()

if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    # Model parameters
    ap.add_argument('--model-arch', choices=['ConvNext-b', 'ConvNext-T', 'mobilefacenet', 'iresnet50', 'iresnet18', 'iresnet34', 'iresnet100', 'iresnet200'], 
                    default='mobilefacenet', help='Which architecture to use')
    ap.add_argument('--resume', choices=['ImageNet', 'Custom'], default=None,
                    help='If the model will be resumed from a pre-trained weigth trained on ImageNet or custom')
    ap.add_argument('--checkpoint-path', type=str, metavar='PATH', help='Path to the model checkpoint', default=None)
    ap.add_argument('--arc-checkpoint', type=str, metavar='PATH', help='Path to the arc Loss checkpoint', default=None)
    ap.add_argument('--features-dim', type=int, help='The models head dimension', default=512)

    # Data parameters
    ap.add_argument('--train-path', required=True, metavar='Path', type=str,
                    help='Path to the training dataset dir')
    ap.add_argument('--test-path', required=True, metavar='Path', type=str,
                    help='Path to the test dataset dir')
    ap.add_argument('--train-bs', type=int, default=64, help='Training batch size')
    ap.add_argument('--test-bs', type=int, default=64, help='Training batch size')

    # Arc loss parameters
    ap.add_argument('--scale', type=float, default=64.0, help='Arcloss scale')
    ap.add_argument('--margin', type=float, default=0.5, help='Arcloss margin')

    # Training parameters
    ap.add_argument('--max-lr', type=float, default=0.1, help='Max Learning Rate for OneCycleLR scheduler')
    ap.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    ap.add_argument('--eval-steps', type=int, default=2000, help='After how many steps to perform the validation')

    # Other options
    ap.add_argument('--evaluation', action='store_true', help='Perform just evaluaiton')
    ap.add_argument('--save-dir', default='./weights', help='Where to save the model weights')

    args = vars(ap.parse_args())

    # Initialize the model
    model = load_model(model_arch=args['model_arch'], pretrained=args['resume'], 
                       checkpoint_path=args['checkpoint_path'], features_dim=args['features_dim'])

    # Load data
    train_loader, test_loader, num_classes = load_data(args['train_path'], args['test_path'],
                                                       args['train_bs'], args['test_bs'])

    if args['evaluation']:
        print('Evaluating only..')
        _ = eval_model(test_loader, model)
        exit()   
    
    train(args, num_classes, model, test_loader)