import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from dfsd_repro.utils.ood_utils import nng_score



def get_msp_score(inputs, model, method_args):
    with torch.no_grad():
        outputs = model(inputs)
    scores = np.max(F.softmax(outputs, dim=1).detach().cpu().numpy(), axis=1)

    return scores



def get_kpca_score(inputs, model, method_args, u, NS, alpha, args, seed):
    numclasses = method_args['num_classes']
    with torch.no_grad():
        if method_args['model_arch'] not in ['resnet18','efficientnet']:
            features = model.features(inputs)
            out = F.adaptive_avg_pool2d(features, 1)
            features = out.view(out.size(0), -1)
            outputs = model.fc(features)
        else:
            feature1 = F.relu(model.bn1(model.conv1(inputs)))
            feature2 = model.layer1(feature1)
            feature3 = model.layer2(feature2)
            feature4 = model.layer3(feature3)
            feature5 = model.layer4(feature4)
            feature5 = model.avgpool(feature5)
            feature = feature5.clip(max=1)
            features = feature.view(feature.size(0), -1)
            outputs = model.fc(features)
        outputs = outputs.cpu().detach().numpy()
        features = features.cpu().detach().numpy()
        energy_ood = logsumexp(outputs, axis=-1)
        vlogit_ood = np.zeros((energy_ood.shape[0], int(numclasses)))

        for i in range(int(numclasses)):
            
            a = NS[i].transform(features - u)
            vlogit_ood[:, i] = norm(
                ((features - u) - NS[i].inverse_transform(NS[i].transform(features - u))),
                axis=-1
            ) * alpha[i]
        scores = np.max(-vlogit_ood, axis=-1) + energy_ood
    return scores

def get_odin_score(inputs, model, method_args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    temper = method_args['temperature']
    noiseMagnitude1 = method_args['magnitude']

    criterion = nn.CrossEntropyLoss()
    inputs = Variable(inputs, requires_grad = True)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    outputs = model(Variable(tempInputs))
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_energy_score(inputs, model, method_args,args):
    # Calculating the perturbation we need to add, that is,
    # the sign of gradient of cross entropy loss w.r.t. input

    temper = method_args['temperature']

    inputs = Variable(inputs, requires_grad = True)
    features = model.features(inputs)
    out = F.adaptive_avg_pool2d(features, 1)
    features = out.view(out.size(0), -1)
    outputs = model.fc(features)
    # Using temperature scaling
    outputs = outputs / temper
    nnOutputs = outputs.data.cpu()
    scores = torch.logsumexp(nnOutputs, dim=1).numpy()
    if args.use_nng == True:
        scores = nng_score(scores, features, args)
    return scores
def get_ash_score(inputs, model, method_args,args):

    inputs = Variable(inputs, requires_grad = True)
    features = model.features(inputs)
    features = F.adaptive_avg_pool2d(features, 1)
    features = scale(features.view(features.size(0), -1, 1, 1), 90)
    features = features.view(features.size(0), -1)

    outputs = model.fc(features)
    # Using temperature scaling
    outputs = outputs
    nnOutputs = outputs.data.cpu()
    scores = torch.logsumexp(nnOutputs, dim=1).numpy()
    return scores
def ash_s(x, percentile=65):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    x = x * torch.exp(scale[:, None, None, None])

    return x

def scale(x, percentile=65):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2

    return input * torch.exp(scale[:, None, None, None])
def get_vim_score(inputs, model, method_args,u,NS,alpha):
    with torch.no_grad():
        if method_args['model_arch'] not in ['swin','efficientnet']:
            features = model.features(inputs)
            out = F.adaptive_avg_pool2d(features, 1)
            features = out.view(out.size(0), -1)
            outputs = model.fc(features)
        else:
            features = model.extract_feat(inputs)
            outputs = model.head(features)
    outputs = outputs.cpu().detach().numpy()
    features = features[0].cpu().detach().numpy()
    energy_ood = logsumexp(outputs, axis=-1)
    vlogit_ood = norm(np.matmul(features - u, NS), axis=-1) * alpha
    scores = -vlogit_ood + energy_ood
    return scores


def get_score(inputs, model, method, method_args, raw_score=False,u=None,NS=None,alpha=None,args=None,seed=0):
    if method == "msp":
        scores = get_msp_score(inputs, model, method_args)
    elif method == "odin":
        scores = get_odin_score(inputs, model, method_args)
    elif method == "energy":
        scores = get_energy_score(inputs, model, method_args,args=args)
    elif method == "ash":
        scores = get_ash_score(inputs, model, method_args,args=args)
    elif method == "vim":
        scores = get_vim_score(inputs, model, method_args,u=u,NS=NS,alpha=alpha)
    elif method == "kpca":
        scores = get_kpca_score(inputs, model, method_args, u=u, NS=NS, alpha=alpha,args=args,seed=seed)

    return scores
