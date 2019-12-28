# Understanding Clouds from Satellite Images
 
Code for https://www.kaggle.com/c/understanding_cloud_organization competition

Example of usage:
>>> python train.py --encoder resnet50 --bs 20 --num_epochs 100 --train True --optimize_postprocess True --make_prediction True

>>> python train.py --encoder densenet169 --bs 20 --num_epochs 100 --train True --task classification --loss BCE --height 224 --width 224 --lr 1e-4



Code mostly taken from https://github.com/Erlemar/Understanding-Clouds-from-Satellite-Images
