# [Gentle guide on how YOLO Object Localization works with Keras](https://heartbeat.fritz.ai/gentle-guide-on-how-yolo-object-localization-works-with-keras-part-2-65fe59ac12d)

Complementary source code for the post.
Keras+TensorFlow YOLO object Localization implementation guided walk through.

## How to Run
Require [Python 3.5+](https://www.python.org/ftp/python/3.6.4/python-3.6.4.exe) and [Jupyter notebook](https://jupyter.readthedocs.io/en/latest/install.html) installed
### Clone or download this repo
```
git clone https://github.com/Tony607/YOLO_Object_Localization
```
### Install required libraries
`pip3 install -r requirements.txt`
### Download the Pre-trained YOLO model
Download the trained model weight from the releases, [yolo.h5](https://github.com/Tony607/YOLO_Object_Localization/releases/download/V0.1/yolo.h5).
Put it to `model_data` folder in the project directory.

### Start the notebook
In the project directory start a command line, then run
```
jupyter notebook
```
In the opened browser window open
```
Gentle guide on how YOLO Object Localization works with Keras.ipynb
```

Side note, if the model doesn't load correctly try to follow the instruction on YAD2K
*  Clone/Download [YAD2K](https://github.com/allanzelener/YAD2K).
*   Download Darknet model cfg and weights from the [official YOLO website](http://pjreddie.com/darknet/yolo/).
*   Convert the Darknet YOLO_v2 model to a Keras model.
*   Test the converted model on the small test set in `images/`.

```source-shell
git clone https://github.com/allanzelener/YAD2K
wget http://pjreddie.com/media/files/yolo.weights
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
./test_yolo.py model_data/yolo.h5  # output in images/out/
```

See `./yad2k.py --help` and `./test_yolo.py --help` for more options.

Happy coding! Leave a comment if you have any question.