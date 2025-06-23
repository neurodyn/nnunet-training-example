
# Training NNUNET V2 on an old Windows PC

I'm using WSL on a Windows 11 PC. So let's try the method described here:

https://docs.nvidia.com/cuda/wsl-user-guide/index.html

then here:

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local


Create and activate a conda environment:
```
conda activate nnunet
```

The machine has CUDA 12.2 which isn't listed on the PyTorch website https://pytorch.org/get-started/locally/. But we can try the pytorch installation from:

https://joelognn.medium.com/installing-wsl2-pytorch-and-cuda-on-windows-11-65a739158d76:
(this seemed to work in a fresh conda environment)

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
Then install nnunuet
```
pip install nnunetv2
```
Set environment variables:

Create and run: 
```
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```
```
source set-env.sh
``` 
at the bash prompt.

Follow: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how_to_use_nnunet.md

1)
```
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```
E.g.:

```
nnUNetv2_plan_and_preprocess -d 200 --verify_dataset_integrity
```
Output:
```
Fingerprint extraction...
Dataset200_NanoTracerTiny
Using <class 'nnunetv2.imageio.natural_image_reader_writer.NaturalImage2DIO'> as reader/writer

####################
verify_dataset_integrity Done.
If you didn't see any error messages then your dataset is most likely OK!
####################

Using <class 'nnunetv2.imageio.natural_image_reader_writer.NaturalImage2DIO'> as reader/writer
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3488/3488 [00:23<00:00, 145.67it/s]Experiment planning...

############################
INFO: You are using the old nnU-Net default planner. We have updated our recommendations. Please consider using those instead! Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
############################

2D U-Net configuration:
{'data_identifier': 'nnUNetPlans_2d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 12, 'patch_size': (512, 512), 'median_image_size_in_voxels': array([500., 500.]), 'spacing': array([1., 1.]), 'normalization_schemes': ['CTNormalization'], 'use_mask_for_norm': [False], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_seg_kwargs': {'is_seg': True, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'resampling_fn_probabilities': 'resample_data_or_seg_to_shape', 'resampling_fn_probabilities_kwargs': {'is_seg': False, 'order': 1, 'order_z': 0, 'force_separate_z': None}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages': 8, 'features_per_stage': (32, 64, 128, 256, 512, 512, 512, 512), 'conv_op': 'torch.nn.modules.conv.Conv2d', 'kernel_sizes': ((3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3), (3, 3)), 'strides': ((1, 1), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2)), 'n_conv_per_stage': (2, 2, 2, 2, 2, 2, 2, 2), 'n_conv_per_stage_decoder': (2, 2, 2, 2, 2, 2, 2), 'conv_bias': True, 'norm_op': 'torch.nn.modules.instancenorm.InstanceNorm2d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch.nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ('conv_op', 'norm_op', 'dropout_op', 'nonlin')}, 'batch_dice': True}

Using <class 'nnunetv2.imageio.natural_image_reader_writer.NaturalImage2DIO'> as reader/writer
Plans were saved to /mnt/g/awoo016/training/nnunet/nnUNet_preprocessed/Dataset200_NanoTracerTiny/nnUNetPlans.json
Preprocessing...
Preprocessing dataset Dataset200_NanoTracerTiny
Configuration: 2d...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3488/3488 [01:13<00:00, 47.18it/s]Configuration: 3d_fullres...
INFO: Configuration 3d_fullres not found in plans file nnUNetPlans.json of dataset Dataset200_NanoTracerTiny. Skipping.
Configuration: 3d_lowres...
INFO: Configuration 3d_lowres not found in plans file nnUNetPlans.json of dataset Dataset200_NanoTracerTiny. Skipping.
```
2) Now we will train a 2D U-Net. We will specify this as the first fold. Syntax: nnUNetv2_train DATASET_NAME_OR_ID 2d FOLD [--npz]
```
nnUNetv2_train 200 2d 0 --npz
```
The training will begin (hopefully).

GPU utilization:

```
Mon Jun 23 16:25:33 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.54.04              Driver Version: 536.23       CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  NVIDIA GeForce RTX 2060        On  | 00000000:50:00.0 Off |                  N/A |
| 38%   59C    P2             151W / 190W |   5934MiB /  6144MiB |     99%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+

+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A        41      G   /Xwayland                                 N/A      |
|    0   N/A  N/A       837      C   /python3.10                               N/A      |
+---------------------------------------------------------------------------------------+
```

Command line output:

```

############################
INFO: You are using the old nnU-Net default plans. We have updated our recommendations. Please consider using those instead! Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md
############################

Using device: cuda:0

#######################################################################
Please cite the following paper when using nnU-Net:
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
#######################################################################

2025-06-23 16:20:48.433895: Using torch.compile...
2025-06-23 16:20:49.701162: do_dummy_2d_data_aug: False
2025-06-23 16:20:49.728989: Using splits from existing split file: /mnt/g/awoo016/training/nnunet/nnUNet_preprocessed/Dataset200_NanoTracerTiny/splits_final.json
2025-06-23 16:20:49.740389: The split file contains 5 splits.
2025-06-23 16:20:49.745311: Desired fold for training: 0
2025-06-23 16:20:49.755418: This split has 2790 training and 698 validation cases.
using pin_memory on device 0
using pin_memory on device 0

This is the configuration used by this training:
Configuration name: 2d
 {'data_identifier': 'nnUNetPlans_2d', 'preprocessor_name': 'DefaultPreprocessor', 'batch_size': 12, 'patch_size': [512, 512], 'median_image_size_in_voxels': [500.0, 500.0], 'spacing': [1.0, 1.0], 'normalization_schemes': ['CTNormalization'], 'use_mask_for_norm': [False], 'resampling_fn_data': 'resample_data_or_seg_to_shape', 'resampling_fn_seg': 'resample_data_or_seg_to_shape', 'resampling_fn_data_kwargs': {'is_seg': False, 'order': 3, 'order_z': 0, 'force_separaNone}, 'architecture': {'network_class_name': 'dynamic_network_architectures.architectures.unet.PlainConvUNet', 'arch_kwargs': {'n_stages':s': 'resample_dat 8, 'features_per_stage': [32, 64, 128, 256, 512, 512, 512, 512], 'conv_op': 'torch.nn.modules.conv.Conv2d', 'kernel_sizes': [[3, 3], [3, 3ure': {'network_c], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], 'strides': [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]], 'n_conv_pe 128, 256, 512, 5r_stage': [2, 2, 2, 2, 2, 2, 2, 2], 'n_conv_per_stage_decoder': [2, 2, 2, 2, 2, 2, 2], 'conv_bias': True, 'norm_op': 'torch.nn.modules.inst 'strides': [[1, ancenorm.InstanceNorm2d', 'norm_op_kwargs': {'eps': 1e-05, 'affine': True}, 'dropout_op': None, 'dropout_op_kwargs': None, 'nonlin': 'torch2, 2, 2, 2, 2, 2].nn.LeakyReLU', 'nonlin_kwargs': {'inplace': True}}, '_kw_requires_import': ['conv_op', 'norm_op', 'dropout_op', 'nonlin']}, 'batch_dice': op': None, 'dropo
True}                                                                                                                                      opout_op', 'nonli

These are the global plan.json settings:
 {'dataset_name': 'Dataset200_NanoTracerTiny', 'plans_name': 'nnUNetPlans', 'original_median_spacing_after_transp': [999.0, 1.0, 1.0], 'original_median_shape_after_transp': [1, 500, 500], 'image_reader_writer': 'NaturalImage2DIO', 'transpose_forward': [0, 1, 2], 'transpose_backginal_median_shapward': [0, 1, 2], 'experiment_planner_used': 'ExperimentPlanner', 'label_manager': 'LabelManager', 'foreground_intensity_properties_per_cha 'experiment_plannnel': {'0': {'max': 4095.0, 'mean': 160.918681859126, 'median': 100.0, 'min': 49.0, 'percentile_00_5': 95.0, 'percentile_99_5': 1488.0, 's': 160.9186818591td': 208.2107709285002}}}

2025-06-23 16:21:54.137305: Unable to plot network architecture: nnUNet_compile is enabled!
2025-06-23 16:21:54.153713: 
2025-06-23 16:21:54.157436: Epoch 0
2025-06-23 16:21:54.172770: Current learning rate: 0.01
W0623 16:22:06.753000 837 site-packages/torch/_inductor/utils.py:1250] [0/0] Not enough SMs to use max_autotune_gemm mode
/root/anaconda3/envs/nnunet_p3102/lib/python3.10/site-packages/torch/_inductor/lowering.py:7007: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
/root/anaconda3/envs/nnunet_p3102/lib/python3.10/site-packages/torch/_inductor/lowering.py:7007: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
/root/anaconda3/envs/nnunet_p3102/lib/python3.10/site-packages/torch/_inductor/lowering.py:7007: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
/root/anaconda3/envs/nnunet_p3102/lib/python3.10/site-packages/torch/_inductor/lowering.py:7007: UserWarning: 
Online softmax is disabled on the fly since Inductor decides to
split the reduction. Cut an issue to PyTorch if this is an
important use case and you want to speed it up with online
softmax.

  warnings.warn(
2025-06-23 16:29:13.153991: train_loss 0.0306
2025-06-23 16:29:13.170832: val_loss -0.0722
2025-06-23 16:29:13.182576: Pseudo dice [np.float32(0.1527)]
2025-06-23 16:29:13.196836: Epoch time: 439.0 s
2025-06-23 16:29:13.209171: Yayy! New best EMA pseudo Dice: 0.1527000069618225
2025-06-23 16:29:16.698571: 
2025-06-23 16:29:16.702106: Epoch 1
2025-06-23 16:29:16.713968: Current learning rate: 0.00999
2025-06-23 16:34:10.173313: train_loss -0.1956
2025-06-23 16:34:10.177030: val_loss -0.3086
2025-06-23 16:34:10.190441: Pseudo dice [np.float32(0.4852)]
2025-06-23 16:34:10.204829: Epoch time: 293.48 s
2025-06-23 16:34:10.218703: Yayy! New best EMA pseudo Dice: 0.1860000044107437
```