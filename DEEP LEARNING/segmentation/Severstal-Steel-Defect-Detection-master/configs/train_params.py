sample_submission_path = (
    "./input/severstal-steel-defect-detection/sample_submission.csv"
)
train_df_path = "./input/severstal-steel-defect-detection/train.csv"
data_folder = "./input/severstal-steel-defect-detection/train_images"
test_data_folder = "./input/severstal-steel-defect-detection/test_images"
FOLDS_ids = "./input/folds.pkl"
lb_test = "./input/severstal-steel-defect-detection/submission_0.91625.csv"
isDebug = False
unet_encoder = "se_resnext50_32x4d"
ATTENTION_TYPE = None
num_epochs = 100
LEARNING_RATE = 5e-4 / 100
BATCH_SIZE = {"train": 4, "val": 1}
TOTAL_FOLDS = 10
model_weights = "imagenet"
EARLY_STOPING = 30


crop_image_size = None  # (256, 1600)
INITIAL_MINIMUM_DICE = 0.9

if isDebug:
    num_epochs = 1
