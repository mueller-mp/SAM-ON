# [Normalization Layers Are All That Sharpness Aware Minimization Needs](https://arxiv.org/abs/2306.04226)

**Maximilian Müller\*, Tiffany Vlaar\*, David Rolnick\*, Matthias Hein**

**NeurIPS 2023**

Paper: [https://arxiv.org/abs/2306.04226](https://arxiv.org/abs/2306.04226)  

![teaser.png](teaser.png)

# SAM-ON
This repository contains the SAM-ON optimizer for CIFAR data. SAM-ON performs the (A)SAM perturbation _only_ for the normalization parameters and achieves superior performance compared to its counterparts using all layers. The optimizer is adapted from the original ASAM optimizer (https://github.com/SamsungLabs/ASAM/)

### Dependencies
You can install the required packages via conda:
```
conda env create -f samon_env.yml
conda activate samon-env
```
In case this doesn't work for you, the required packages can also be found in `requirements.txt`.
### How to run the code
The code is set up to train a WRN28-10 on CIFAR100, and the desired options can be passed with flags:
```
  python train.py --dataset=CIFAR100 --data_path=/path/to/CIFAR100/ --minimizer=ASAM_ON --p=2 --elementwise --normalize_bias --autoaugment --rho=10. --only_norm
 ```
 Currently, SAM_ON, ASAM_ON, AdamW and SGD can be selected as optimizers. If ASAM_ON or SAM_ON is chosen and neither the only_norm nor the no_norm flag are set, the conventional (A)SAM optimizer is used.
### New Models
The available models are wrn28_10, resnet56, resnext vit_t, vit_s. Those can be chosen via the `--model` flag. If you would like to try SAM-ON on other models, you need to add them in `models.py`. Since for now the optimizer is selecting the normalization-layers by name, you need to make sure that _all_ normalization parameters contain the string 'norm' or 'bn' in their name, otherwise the optimizer does not recognize them. You can check this by calling model.named_parameters().
### Citations
```
@inproceedings{mueller2023normalization,
      title={Normalization Layers Are All That Sharpness-Aware Minimization Needs}, 
      author={Maximilian Mueller and Tiffany Vlaar and David Rolnick and Matthias Hein},
      year={2023},
      booktitle = {Advances in Neural Information Processing Systems},
}
```
### License
