# Severstal-Steel-Defect-Detection
Can you detect and classify defects in steel? Segmentation in Pytorch
https://www.kaggle.com/c/severstal-steel-defect-detection/overview

31 place solution

Project structure:
 * **common_blocks** - classes and functions for training
    - **dataloader.py** - dataloader of competition data    
    - **metric.py** - function with metric functions
    - **training_helper.py** - main class for training with cross validation or train_test_split
    - **utils.py** - useful utils for logs, model loading, mask transformation
    - **losses.py** - different segmentation losses
    - **optimizers.py** - SOTA optimizers
* **configs**
   - **train_params.py** (in development)
   1. **isDebug - then is True, pipelines finishes to work in minutes (useful for code testing)
   2. **unet_encoder** - you can specify unet encoder (resnet, resnext, se-resnext are available)
   3. **crop_image_size** - if specified will train on crops, othwerwise on full image size
   4. **attention_type** - will add scse blocks to decoder
   5. **path, folder** - paths to competition data
   
* **train.py** - main code for training
* **inference.py** (TODO - currently in kaggle kernels)


**SOLUTION**

**Team - [ods.ai] stainless**

- Insaf Ashrapov
- Igor Krashenyi
- Pavel Pleskov
- Anton Zakharenkov
- Nikolai Popov

**Models** 
We tried almost every type of model from qubvel`s segmentation model library - unet, fpn, pspnet with different encoders from resnet to senet152. FPN with se-resnext50 outperformed other models. Lighter models like resnet34 performed aren't well enough but were useful in the final blend. Se-resnext101 possibly could perform much better with more time training, but we didn’t test that.

**Augmentations and Preprocessing**
From **Albumentations** library:
Hflip, VFlip, RandomBrightnessContrast – training speed was not to fast so these basic augmentations performed well enough. In addition, we used big crops for training or/and finetuning on the full image size, because attention blocks in image tasks rely on the same input size for the training and inference phase.

**Training**
- We used both pure pytorch and Catalyst framework for training.
- Losses: bce and bce with dice performed quite well, but lovasz loss dramatically outperformed them in terms of validation and public score. However, combining with classification model bce with dice gave a better result, that could be because Lovasz helped the model to filter out false-positive masks. Focal loss performed quite poor due to not very good labeling.
- Optimizer: Adam with RAdam. LookAHead, Over900 didn’t work well to use.
- Crops with a mask, BalanceClassSampler with upsampler mode from catalyst significantly increased training speed.

- We tried own classification model (resnet34 with CBAM) by setting the goal to improve f1 for each class. The optimal threshold was disappointingly unstable but we reached averaged f1 95.1+. As a result, Cheng`s classification was used.

- Validation: kfold with 10 folds. Despite the shake-up – local, public and private correlated surprisingly good.

- Pseudolabing; We did two rounds of pseudo labeling by training on the best public submit and validating on the out of fold. It didn’t work for the third time but gave us a huge improvement.

- Postprocessing: filling holes, removing the small mask by the threshold. We tried to remove small objects by connected components with no improvements.

- Hardware: bunch of nvidia cards

**Ensembling**
Simple segmentation models averaging with different encoders, both FPN and Unet applied to images classified having a mask. One of the unchosen submit could give as 16th place.
