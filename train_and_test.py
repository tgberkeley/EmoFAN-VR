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
import matplotlib.pyplot as plt

from emonet.models import EmoNet
from AFEW_VA_dataloader import AFEW_VA
from AffectNet_dataloader import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip
from emonet.metrics import CCCLoss, CCC_score

rnd_seed = 42

torch.backends.cudnn.benchmark = True


# Parameters of the experiments
n_expression=8
batch_size = 32
n_workers = 0
# device = 'cuda:0'
image_size = 256
subset = 'train'
metrics_valence_arousal = {'CCC': CCC, 'PCC': PCC, 'RMSE': RMSE, 'SAGR': SAGR}
metrics_expression = {'ACC': ACC}


learning_rate = 0.00008
weight_decay = 0.001
print(learning_rate)
CCC_Loss = CCCLoss(digitize_num=1)
num_epochs = 10


cuda_dev = '0'  # GPU device 0 (can be changed if multiple GPUs are available)
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:" + cuda_dev if use_cuda else "cpu")
print('Device: ' + str(device))
if use_cuda:
    print('GPU: ' + str(torch.cuda.get_device_name(int(cuda_dev))))

out_dir = './output'

# LOAD TRAINING DATA  

model_dir = os.path.join(out_dir, 'model')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

torch.manual_seed(rnd_seed)  # fix random seed

# Create the data loaders
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)



print('Loading the data')
train_dataset_no_flip = AffectNet(root_path='/vol/bitbucket/tg220/data/train_set/', subset='train', n_expression=n_expression,
                                  transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)

test_dataset_no_flip = AFEW_VA(root_path='/vol/bitbucket/tg220/data/AFEW_VA_all/', subset='test', n_expression=n_expression,
                                 transform_image_shape=transform_image_shape_no_flip, transform_image=transform_image)


# Loading the model
###Here we either use emonet_8.pth or my_model.pth
model = '34421_epoch_6_lr_0.00008_with_dropout_with_train.pth'
state_dict_path = Path(__file__).parent.joinpath('pretrained', model)

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')

state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

# as have added in the drop out layer and we are using pretrained weights
if model == 'emonet_8.pth':
  state_dict['emo_fc_2.4.weight'] = state_dict['emo_fc_2.3.weight']
  del state_dict['emo_fc_2.3.weight']
  state_dict['emo_fc_2.4.bias'] = state_dict['emo_fc_2.3.bias']
  del state_dict['emo_fc_2.3.bias']


net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)


###Uncomment these to change which parameters of the model we train
# for model_block in list(net.children())[10:20]:
#     for param in model_block.parameters():
#         param.requires_grad = True

###Uncomment these to view all the parameters of the model
# for k,v in net.named_parameters():
#    print('{}: {}'.format(k, v.requires_grad))

params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total number of parameters in the EmoFan: {}".format(params))
print('\n')




optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)


# 0:Neutral 1:Happy 2:Sad 3:Surprise 4:Fear 5:Disgust 6:Anger 7:Contempt
# (looking at the VA cirlce we really dont cover the bottom half very well at all)
#expr_to_valence = torch.tensor([ 0,  0.9, -0.81, 0.42, -0.11,  -0.67,  -0.41, -0.57])
# issue here is that only 1 negative arousal, so will favour more +ve ones
#expr_to_arousal = torch.tensor([ 0,  0.16, -0.4, 0.88, 0.79,  0.49,  0.78, 0.66])
#softm = nn.Softmax(dim=1)



###All code below is for training, uncomment it if you wish to train a model further

# total_loss_train = []
# CCC_loss_train = []
# PCC_loss_train = []
# RMSE_loss_train = []
# CE_loss_train = []

# print('START TRAINING...')
# for epoch in range(1, num_epochs + 1):

#   train_dataloader = DataLoader(train_dataset_no_flip, batch_size=batch_size, shuffle=True, num_workers=n_workers)
test_dataloader = DataLoader(test_dataset_no_flip, batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
#     net.train()
    
#     total_loss_epoch = 0
#     CCC_loss_epoch = 0
#     PCC_loss_epoch = 0
#     RMSE_loss_epoch = 0
#     CE_loss_epoch = 0
#     # Training
#     for batch_idx, batch_samples in enumerate(train_dataloader):
#         #print(batch_idx)
#         image = batch_samples['image'].to(device)
#         valence = batch_samples['valence'].to(device)
#         valence = valence.squeeze()
#         arousal = batch_samples['arousal'].to(device)
#         arousal = arousal.squeeze()
#         expression = batch_samples['expression'].to(device)
#         expression = expression.squeeze()



#         optimizer.zero_grad()
#         prediction = net(image)

#         pred_expr = prediction['expression']

#         # printing heat maps relative to occluded image
#         # x = 29
#         # heatmap = prediction['heatmap']
#         # #print(heatmap.size())
#         # heat_1 = heatmap[x,:,:,:]
#         #
#         # heat_1 = heat_1.squeeze().detach().cpu().numpy()
#         #
#         #
#         # # sum them so that we can get all landmarks on one heat map
#         # heat_1 = np.sum(heat_1, axis=0)
#         #
#         #
#         # img = image[x,:,:,:].mul(255).byte()
#         # img = img.cpu().numpy().transpose((1, 2, 0))
#         #
#         # image = Image.fromarray(img, 'RGB')
#         # image.show()
#         # Tensor_a = sns.heatmap(heat_1, linewidth=0.2)
#         # plt.show()
#         # sys.exit()

#         # binary cross entrpy loss (for discrete emtions)

#         #loss_CE = F.cross_entropy(pred_expr, expression)


#         ### test on it non_occluded images

#         # pred_expr_soft = softm(pred_expr)
#         #
#         # new_val = torch.mul(pred_expr_soft, expr_to_valence)
#         # expr_val = torch.sum(new_val, dim=1)
#         #
#         # new_aro = torch.mul(pred_expr_soft, expr_to_arousal)
#         # expr_aro = torch.sum(new_aro, dim=1)
#         #
#         # prediction_valence = torch.mul(prediction['valence'], 1 - ratio) + torch.mul(expr_val, ratio)
#         # prediction_arousal = torch.mul(prediction['arousal'], 1 - ratio) + torch.mul(expr_aro, ratio)
#         #



#         CCC_valence, PCC_valence = CCC_Loss(valence, prediction['valence'])
#         CCC_arousal, PCC_arousal = CCC_Loss(arousal, prediction['arousal'])

#         loss_PCC = 1 - ((PCC_valence + PCC_arousal) / 2)
#         loss_CCC = 1 - ((CCC_valence + CCC_arousal) / 2)

#         loss_RMSE = F.mse_loss(valence, prediction['valence']) + F.mse_loss(arousal, prediction['arousal'])

#         # shake–shake regularization coefficients α, β and γ

#         alpha = np.random.uniform()
#         beta = np.random.uniform()
#         gamma = np.random.uniform()
#         total = alpha + beta + gamma

#         total_loss = torch.mul(loss_CCC, alpha/total) + torch.mul(loss_PCC, beta/total) + \
#                      torch.mul(loss_RMSE, gamma/total) #+  loss_CE

#         total_loss.backward()

#         optimizer.step()

#         total_loss_epoch += total_loss.item()
#         CCC_loss_epoch += loss_CCC.item()
#         PCC_loss_epoch += loss_PCC.item()
#         RMSE_loss_epoch += loss_RMSE.item()
#         #CE_loss_epoch += loss_CE.item()

#     total_loss_train.append(total_loss_epoch)
#     CCC_loss_train.append(CCC_loss_epoch)
#     PCC_loss_train.append(PCC_loss_epoch)
#     RMSE_loss_train.append(RMSE_loss_epoch)
#     #CE_loss_train.append(CE_loss_epoch)


#     print('+ TRAINING \tEpoch: {} \tLoss: {:.6f}'.format(epoch, total_loss_epoch),
#           f'\tCCC: {CCC_loss_epoch}, \tPCC: {PCC_loss_epoch}, \tRMSE Loss: {RMSE_loss_epoch}',
#           f'\tCE Loss: {CE_loss_epoch}')
#     #print(f"Total Loss: {total_loss_train}")
#     print(f"CCC Loss: {CCC_loss_train}")
#     #print(f"PCC Loss: {PCC_loss_train}")
#     print(f"RMSE Loss: {RMSE_loss_train}")
#     #print(f"CE Loss: {CE_loss_train}")



#     torch.save(net.state_dict(), os.path.join(model_dir, f'new_with_L2(0.001)_epoch_{epoch}_lr_0.00008_with_dropout.pth'))
    



print('START TESTING...')
net.eval()


for index, data in enumerate(test_dataloader):
    #print(index)
    images = data['image'].to(device)
    valence = data.get('valence', None)
    arousal = data.get('arousal', None)
    expression = data.get('expression', None)

    valence = np.squeeze(valence.cpu().numpy())
    arousal = np.squeeze(arousal.cpu().numpy())
    expression = np.squeeze(expression.cpu().numpy())

    with torch.no_grad():
        out = net(images)

    val = out['valence']
    ar = out['arousal']
    expr = out['expression']

    val = np.squeeze(val.cpu().numpy())
    ar = np.squeeze(ar.cpu().numpy())
    expr = np.squeeze(expr.cpu().numpy())

    if index:
        valence_pred = np.concatenate([val, valence_pred])
        arousal_pred = np.concatenate([ar, arousal_pred])
        valence_gts = np.concatenate([valence, valence_gts])
        arousal_gts = np.concatenate([arousal, arousal_gts])
        expression_pred = np.concatenate([expr, expression_pred])
        expression_gts = np.concatenate([expression, expression_gts])

    else:
        valence_pred = val
        arousal_pred = ar
        valence_gts = valence
        arousal_gts = arousal
        expression_pred = expr
        expression_gts = expression


# Clip the predictions
valence_pred = np.clip(valence_pred, -1.0, 1.0)
arousal_pred = np.clip(arousal_pred, -1.0, 1.0)

# remove this later, this is only so plot looks good
vale = np.array([1, -1])
valence_pred = np.concatenate([valence_pred, vale])
valence_gts = np.concatenate([valence_gts, vale])
arousal_pred = np.concatenate([arousal_pred, vale])
arousal_gts = np.concatenate([arousal_gts, vale])


# Squeeze if valence_gts is shape (N,1)
valence_gts = np.squeeze(valence_gts)
arousal_gts = np.squeeze(arousal_gts)
expression_gts = np.squeeze(expression_gts)

print(expression_pred)
print(expression_gts)
expression_pred = np.argmax(expression_pred, axis=1)
print(expression_pred)
num_correct = (expression_pred == expression_gts).sum()
print(num_correct)
print(len(expression_gts))
accuracy = num_correct / len(expression_gts)
print(accuracy)





plt.hist2d(valence_pred, arousal_pred, bins=(21, 21), cmap=plt.cm.jet)
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.colorbar()
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Predictions Distribution")
plt.savefig("/vol/bitbucket/tg220/results/ground_truth_AFEW_VA_21.png")
plt.draw()

plt.hist2d(valence_pred, arousal_pred, bins=(42, 42), cmap=plt.cm.jet)
plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Predictions Distribution")
plt.ylim(-1,1)
plt.xlim(-1,1)
plt.savefig("/vol/bitbucket/tg220/results/predictions_AFEW_VA_42.png")
plt.show()












CCC_valence, PCC_valence = CCC_score(valence_gts, valence_pred)
RMSE_valence = RMSE(valence_gts, valence_pred)

CCC_arousal, PCC_arousal = CCC_score(arousal_gts, arousal_pred)
RMSE_arousal = RMSE(arousal_gts, arousal_pred)


print('+ TESTING',
      f'\tCCC Valence: {CCC_valence}, \tPCC Valence: {PCC_valence}, \tRMSE Valence: {RMSE_valence}')
print(f'\tCCC Arousal: {CCC_arousal}, \tPCC Arousal: {PCC_arousal}, \tRMSE Arousal: {RMSE_arousal}')

print('\nFinished TESTING.')


