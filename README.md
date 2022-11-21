# SAM-BN
This repository contains the SAM-BN optimizer used for the experiments in the Neurips 2022 paper 'Perturbing BatchNorm and Only BatchNorm benefits Sharpness-Aware Minimization' at the HITY-Workshop. SAM-BN performs the (A)SAM perturbation _only_ for the BatchNorm parameters and achieves superior performance compared to its counterparts using all layers. The optimizer is adapted from the original ASAM optimizer (https://github.com/SamsungLabs/ASAM/)
### How to run the code
The code is set up to train a WRN28-10 on CIFAR100, and the desired options can be passed with flags:
```
  python train.py --dataset=CIFAR100 --data_path=/path/to/CIFAR100/ --minimizer=ASAM_BN --p=2 --elementwise --normalize_bias --autoaugment --rho=10. --only_bn
 ```
 Currently, SAM_BN, ASAM_BN and SGD can be selected as optimizers. If ASAM_BN or SAM_BN is chosen and neither the only_bn nor the no_bn flag are set, the conventional (A)SAM optimizer is used.
### New Models
For now, the optimizer is selecting the BN-layers by name. If you would like to try it on other models than the implemented WRN28-10, make sure that _all_ BatchNorm parameters contain the string 'norm' or 'bn' in their name, otherwise the optimizer does not recognize them. You can check this by calling model.parameters().
# Citations
# License
