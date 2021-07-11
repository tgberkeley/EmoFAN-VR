import numpy as np
from pathlib import Path
import argparse
import os
import sys

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms

from PIL import Image

from emonet.models import EmoNet
# from emonet.data import AffectNet
# from AFEW_VA_dataloader import AffectNet
# from AffWild2_dataloader import AffectNet
from AffectNet_dataloader import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip
from emonet.metrics import CCCLoss, CCC_score

rnd_seed = 42

torch.backends.cudnn.benchmark = True

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5, 8],
                    help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 0
# device = 'cuda:0'
image_size = 256
subset = 'train'
metrics_valence_arousal = {'CCC': CCC, 'PCC': PCC, 'RMSE': RMSE, 'SAGR': SAGR}
metrics_expression = {'ACC': ACC}

learning_rate = 0.001
CCC_Loss = CCCLoss(digitize_num=1)
num_epochs = 4

cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

out_dir = './output'

# LOAD TRAINING DATA  # later we can add some data to validation


model_dir = os.path.join(out_dir, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.manual_seed(rnd_seed)  # fix random seed

# Create the data loaders
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)

# '/vol/bitbucket/tg220/data/AffectNet_val_set/

print('Loading the data')
train_dataset_no_flip = AffectNet(root_path='/vol/bitbucket/tg220/data/train_set/', subset='train', n_expression=n_expression,
                                  transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)

test_dataset_no_flip = AffectNet(root_path='/vol/bitbucket/tg220/data/AffectNet_val_set/', subset='test', n_expression=n_expression,
                                 transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)

train_dataloader = DataLoader(train_dataset_no_flip, batch_size=batch_size, shuffle=False, num_workers=n_workers)

test_dataloader = DataLoader(test_dataset_no_flip, batch_size=batch_size, shuffle=False, num_workers=n_workers)

# Loading the model
# state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')
state_dict_path = Path(__file__).parent.joinpath('pretrained', 'emonet_8.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)

# for param_tensor in net.state_dict():
#    print(param_tensor, "\t", net.state_dict()[param_tensor].size())


params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total number of parameters in the EmoFan: {}".format(params))
print('\n')

net.train()

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


# 0:Neutral 1:Happy 2:Sad 3:Surprise 4:Fear 5:Disgust 6:Anger 7:Contempt
# (looking at the VA cirlce we really dont cover the bottom half very well at all)
expr_to_valence = torch.tensor([ 0,  0.9, -0.81, 0.42, -0.11,  -0.67,  -0.41, -0.57])
# issue here is that only 1 negative arousal, so will favour more +ve ones
expr_to_arousal = torch.tensor([ 0,  0.16, -0.4, 0.88, 0.79,  0.49,  0.78, 0.66])
softm = nn.Softmax(dim=1)

# this is the ratio between the VA prediction and the prediction out from the expr prediction
ratio = 0.3  # will run tests varying this number


total_loss_train = []
CCC_loss_train = []
PCC_loss_train = []
RMSE_loss_train = []

print('START TRAINING...')
for epoch in range(1, num_epochs + 1):

    total_loss_epoch = 0
    CCC_loss_epoch = 0
    PCC_loss_epoch = 0
    RMSE_loss_epoch = 0
    # Training
    for batch_idx, batch_samples in enumerate(train_dataloader):
        print(batch_idx)
        image = batch_samples['image'].to(device)
        valence = batch_samples['valence'].to(device)
        valence = valence.squeeze()
        arousal = batch_samples['arousal'].to(device)
        arousal = arousal.squeeze()
        expr_to_valence = expr_to_valence.to(device)
        expr_to_arousal = expr_to_arousal.to(device)
        #val_from_expr = batch_samples['val_from_expr'].to(device)
        #val_from_expr = val_from_expr.squeeze()
        #aro_from_expr = batch_samples['aro_from_expr'].to(device)
        #aro_from_expr = aro_from_expr.squeeze()


        optimizer.zero_grad()
        prediction = net(image)

        pred_expr = prediction['expression']
        pred_expr_soft = softm(pred_expr)

        new_val = torch.mul(pred_expr_soft, expr_to_valence)
        expr_val = torch.sum(new_val, dim=1)

        new_aro = torch.mul(pred_expr_soft, expr_to_arousal)
        expr_aro = torch.sum(new_aro, dim=1)

        prediction_valence = torch.mul(prediction['valence'], 1 - ratio) + torch.mul(expr_val, ratio)
        prediction_arousal = torch.mul(prediction['arousal'], 1 - ratio) + torch.mul(expr_aro, ratio)


        # maybe look to use shake–shake regularization coefficients α, β and γ (including between valence and arousal
        # so it doesnt favour one over the other

        # remember to change to cuda in loss class

        CCC_valence, PCC_valence = CCC_Loss(valence, prediction_valence)
        CCC_arousal, PCC_arousal = CCC_Loss(arousal, prediction_arousal)

        loss_PCC = 1 - ((PCC_valence + PCC_arousal) / 2)
        loss_CCC = 1 - ((CCC_valence + CCC_arousal) / 2)

        loss_RMSE = F.mse_loss(valence, prediction_valence) + F.mse_loss(arousal, prediction_arousal)

        total_loss = loss_CCC + loss_PCC + loss_RMSE
        total_loss.backward()

        optimizer.step()

        total_loss_epoch += total_loss.item()
        CCC_loss_epoch += loss_CCC.item()
        PCC_loss_epoch += loss_PCC.item()
        RMSE_loss_epoch += loss_RMSE.item()

    total_loss_train.append(total_loss_epoch)
    CCC_loss_train.append(CCC_loss_epoch)
    PCC_loss_train.append(PCC_loss_epoch)
    RMSE_loss_train.append(RMSE_loss_epoch)

    # after 4 epochs had quite a high RMSE but good CCC and PCC (maybe look to favour RMSE loss more)

    print('+ TRAINING \tEpoch: {} \tLoss: {:.6f}'.format(epoch, total_loss_epoch),
          f'\tCCC: {CCC_loss_epoch}, \tPCC: {PCC_loss_epoch}, \tRMSE Loss: {RMSE_loss_epoch}')
    print(f"Total Loss: {total_loss_train}")
    print(f"CCC Loss: {CCC_loss_train}")
    print(f"PCC Loss: {PCC_loss_train}")
    print(f"RMSE Loss: {RMSE_loss_train}")

torch.save(net.state_dict(), os.path.join(model_dir, 'model_affectnet_4_epoch_VA_and_expr_ratio_0.3_(0.001).pth'))

print('\nFinished TRAINING.')

#
# #If want to load a pretrained model of my own
# state_dict_path = Path(__file__).parent.joinpath('output','model','model.pth')
#
# print(f'Loading the model from {state_dict_path}.')
# state_dict = torch.load(str(state_dict_path), map_location='cpu')
#
# state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
#
# net = EmoNet(n_expression=n_expression).to(device)
# net.load_state_dict(state_dict, strict=False)


print('START TESTING...')

net.eval()

for index, data in enumerate(test_dataloader):

    images = data['image'].to(device)
    valence = data.get('valence', None)
    arousal = data.get('arousal', None)

    valence = np.squeeze(valence.cpu().numpy())
    arousal = np.squeeze(arousal.cpu().numpy())

    with torch.no_grad():
        out = net(images)

    # val = out['valence']
    # ar = out['arousal']

    pred_expr = out['expression']
    pred_expr_soft = softm(pred_expr)

    new_val = torch.mul(pred_expr_soft, expr_to_valence)
    expr_val = torch.sum(new_val, dim=1)

    new_aro = torch.mul(pred_expr_soft, expr_to_arousal)
    expr_aro = torch.sum(new_aro, dim=1)

    prediction_valence = torch.mul(out['valence'], 1 - ratio) + torch.mul(expr_val, ratio)
    prediction_arousal = torch.mul(out['arousal'], 1 - ratio) + torch.mul(expr_aro, ratio)

    val = np.squeeze(prediction_valence.cpu().numpy())
    ar = np.squeeze(prediction_arousal.cpu().numpy())

    if index:
        valence_pred = np.concatenate([val, valence_pred])
        arousal_pred = np.concatenate([ar, arousal_pred])
        valence_gts = np.concatenate([valence, valence_gts])
        arousal_gts = np.concatenate([arousal, arousal_gts])

    else:
        valence_pred = val
        arousal_pred = ar
        valence_gts = valence
        arousal_gts = arousal

# Clip the predictions
valence_pred = np.clip(valence_pred, -1.0, 1.0)
arousal_pred = np.clip(arousal_pred, -1.0, 1.0)

# Squeeze if valence_gts is shape (N,1)
valence_gts = np.squeeze(valence_gts)
arousal_gts = np.squeeze(arousal_gts)

CCC_valence, PCC_valence = CCC_score(valence_gts, valence_pred)
RMSE_valence = RMSE(valence_gts, valence_pred)

CCC_arousal, PCC_arousal = CCC_score(arousal_gts, arousal_pred)
RMSE_arousal = RMSE(arousal_gts, arousal_pred)

print('+ TESTING', f'\tCCC Valence: {CCC_valence}, \tPCC Valence: {PCC_valence}, \tRMSE Valence: {RMSE_valence}')
print(f'\tCCC Arousal: {CCC_arousal}, \tPCC Arousal: {PCC_arousal}, \tRMSE Arousal: {RMSE_arousal}')

print('\nFinished TESTING.')

# evaluate(net, test_dataloader, device=device, metrics_valence_arousal=metrics_valence_arousal, metrics_expression=metrics_expression)
