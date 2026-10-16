import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from dfsd_repro.config import ExperimentConfig
from dfsd_repro.dfsd import _train_dataset, train_feature_paths, feature_dim_for, score_numpy, fit_or_load_subspaces, classifier_params, extract_features_and_logits
from dfsd_repro.utils.ood_utils import transform_test, transform_test_largescale
from dfsd_repro.models.route_test import RouteDICE
from dfsd_repro.models.mobilenetv2 import mobilenet_v2


def config(**kwargs):
    return ExperimentConfig(experiment_name='test', in_dataset='CIFAR-100', **kwargs)


def test_same_preprocessing_in_fitting_and_evaluation():
    with patch('dfsd_repro.dfsd.torchvision.datasets.CIFAR100') as dataset:
        _train_dataset(config())
        assert dataset.call_args.kwargs['transform'] is transform_test
    cfg = ExperimentConfig(experiment_name='test', in_dataset='imagenet', model_arch='resnet50')
    with patch('dfsd_repro.dfsd.torchvision.datasets.ImageFolder') as dataset:
        _train_dataset(cfg)
        assert dataset.call_args.kwargs['transform'] is transform_test_largescale


def test_mask_ties_and_logits():
    head = RouteDICE(4, 2, p=50, info=np.ones(4))
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1., 2., 2., 3.], [1., 2., 2., 3.]]))
        head.bias.zero_()
    out = head(torch.ones(1, 4))
    assert head.thresh == 2
    assert torch.allclose(out, torch.tensor([[3., 3.]]))


def test_scoring_uses_shift_and_selected_models():
    class ZeroInverse:
        def transform(self, x):return x
        def inverse_transform(self, x):return np.zeros_like(x)
    features=np.array([[3.,4.]])
    actual=score_numpy(features,np.zeros((1,2)),np.array([2.,3.]),[ZeroInverse(),ZeroInverse()],np.array([3.,0.]),'csp_kpca')
    np.testing.assert_allclose(actual,[-8.])


def test_seeded_feature_paths_are_separate():
    a=ExperimentConfig(experiment_name='t',in_dataset='imagenet',model_arch='resnet50',seed=0)
    b=ExperimentConfig(experiment_name='t',in_dataset='imagenet',model_arch='resnet50',seed=1)
    assert train_feature_paths(a)!=train_feature_paths(b)
    assert 'shared_transforms_v2' in str(train_feature_paths(config())[0])


def test_mobilenet_adapter_without_download():
    cfg=ExperimentConfig(experiment_name='t',in_dataset='imagenet',model_arch='mobilenetv2',p=None)
    model=mobilenet_v2(pretrained=False).eval()
    assert feature_dim_for(cfg)==1280
    w,b=classifier_params(model,cfg)
    assert w.shape==(1000,1280)
    features, logits=extract_features_and_logits(model,torch.zeros(1,3,32,32),cfg)
    assert features.shape==(1,1280) and logits.shape==(1,1000)


def test_evr_fit_uses_class_data_and_shared_logit_reference(tmp_path):
    rng=np.random.default_rng(10)
    features=rng.normal(size=(16,4)); labels=np.repeat([0,1],8)
    cfg=config(dim=2,evr=0.8,subspace_cache_dir=str(tmp_path))
    w=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]]);b=np.array([2.,2.])
    with patch('dfsd_repro.dfsd.load_or_extract_train_features',return_value=(features,labels,features.mean(0))),patch('dfsd_repro.dfsd.classifier_params',return_value=(w,b)),patch('dfsd_repro.dfsd.feature_dim_for',return_value=4):
        alpha,estimators,u=fit_or_load_subspaces(cfg,None,'kpca',2)
    numerator=(features@w.T+b).max(axis=-1).mean()
    from dfsd_repro.dfsd import class_order
    for i,c in enumerate(class_order(2,1.,cfg.seed)):
        x=features[labels==c]-u
        residual=np.linalg.norm(x-estimators[i].inverse_transform(estimators[i].transform(x)),axis=-1)
        np.testing.assert_allclose(alpha[i]*residual.mean(),numerator)
        assert 1<=estimators[i].n_components<=7
