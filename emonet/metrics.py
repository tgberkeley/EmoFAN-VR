import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable






def ACC(ground_truth, predictions):
    """Evaluates the mean accuracy
    """
    return np.mean(ground_truth.astype(int) == predictions.astype(int))

def RMSE(ground_truth, predictions):
    """
        Evaluates the RMSE between estimate and ground truth.
    """
    return np.sqrt(np.mean((ground_truth-predictions)**2))


def SAGR(ground_truth, predictions):
    """
        Evaluates the SAGR between estimate and ground truth.
    """
    return np.mean(np.sign(ground_truth) == np.sign(predictions))


def PCC(ground_truth, predictions):
    """
        Evaluates the Pearson Correlation Coefficient.
        Inputs are numpy arrays.
        Corr = Cov(GT, Est)/(std(GT)std(Est))
    """
    return np.corrcoef(ground_truth, predictions)[0,1]


def CCC(ground_truth, predictions):
    """
        Evaluates the Concordance Correlation Coefficient.
        Inputs are numpy arrays.
    """
    mean_pred = np.mean(predictions)
    mean_gt = np.mean(ground_truth)

    std_pred= np.std(predictions)
    std_gt = np.std(ground_truth)

    pearson = PCC(ground_truth, predictions)
    return 2.0*pearson*std_pred*std_gt/(std_pred**2+std_gt**2+(mean_pred-mean_gt)**2)

def ICC(labels, predictions):
    """Evaluates the ICC(3, 1) 
    """
    naus = predictions.shape[1]
    icc = np.zeros(naus)

    n = predictions.shape[0]

    for i in range(0,naus):
        a = np.asmatrix(labels[:,i]).transpose()
        b = np.asmatrix(predictions[:,i]).transpose()
        dat = np.hstack((a, b))
        mpt = np.mean(dat, axis=1)
        mpr = np.mean(dat, axis=0)
        tm  = np.mean(mpt, axis=0)
        BSS = np.sum(np.square(mpt-tm))*2
        BMS = BSS/(n-1)
        RSS = np.sum(np.square(mpr-tm))*n
        tmp = np.square(dat - np.hstack((mpt,mpt)))
        WSS = np.sum(np.sum(tmp, axis=1))
        ESS = WSS - RSS
        EMS = ESS/(n-1)
        icc[i] = (BMS - EMS)/(BMS + EMS)

    return icc



####### my stuff from here



# good one :
#Custom Losses: CCC (with PCC inside it)
# https://github.com/wtomin/Multitask-Emotion-Recognition-with-Incomplete-Labels/blob/3d18c3f1eea6f93522d00e7a58d669e1051c3610/Multitask-CNN-RNN/utils/model_utils.py#L93

class CCCLoss(nn.Module):
    def __init__(self, digitize_num, range=[-1, 1]):
        super(CCCLoss, self).__init__()
        self.digitize_num =  digitize_num
        self.range = range
        self.eps = 0.000000001 # used to prevent a nan return when e.g. all the gts are 0
        if self.digitize_num !=0:
            bins = np.linspace(*self.range, num= self.digitize_num)
            #self.bins = Variable(torch.as_tensor(bins, dtype = torch.float32).cuda()).view((1, -1))
            self.bins = Variable(torch.as_tensor(bins, dtype=torch.float32)).view((1, -1))
    def forward(self, x, y):
        # the target y is continuous value (BS, )
        # the input x is either continuous value (BS, ) or probability output(digitized)
        y = y.view(-1)

        if self.digitize_num !=1:
            x = F.softmax(x, dim=-1)
            x = (self.bins * x).sum(-1) # expectation
        x = x.view(-1)

        vx = x - torch.mean(x) + self.eps
        vy = y - torch.mean(y)
        rho =  torch.sum(vx * vy) / (torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))))
        x_m = torch.mean(x) + self.eps
        y_m = torch.mean(y)
        x_s = torch.std(x) + self.eps
        y_s = torch.std(y)
        ccc = 2*rho*x_s*y_s/(torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))

        return ccc, rho

#CCC_loss = CCCLoss(digitize_num=1)




def CCC_score(x, y):
    vx = x - np.mean(x)
    vy = y - np.mean(y)
    rho = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))
    x_m = np.mean(x)
    y_m = np.mean(y)
    x_s = np.std(x)
    y_s = np.std(y)
    ccc = 2*rho*x_s*y_s/(x_s**2 + y_s**2 + (x_m - y_m)**2)
    return ccc, rho






