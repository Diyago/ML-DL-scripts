### Google Landmark Retrieval Challenge

# export PATH=~/anaconda3/bin:$PATH
# pip install --ignore-installed --upgrade "https://github.com/sigilioso/tensorflow-build/raw/master/tensorflow-1.4.0-cp36-cp36m-linux_x86_64.whl"

from tqdm import tqdm
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.layers import Flatten, Input
from keras.models import Model
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from google.cloud import storage
from io import BytesIO
import time

start = time.time()


model = ResNet50(weights="imagenet", pooling=max, include_top=False)


client = storage.Client()
bucket = client.get_bucket("landsbyconst")

bucket_list = list(bucket.list_blobs())

f = BytesIO(file.download_as_string())

X_test = []

####### GENERATING FEATURES
# file creation
train_filenames = open("train_filenames.txt", "w+")
test_filenames = open("test_filenames.txt", "w+")


train_featues = open("train_featues.txt", "w+")
test_features = open("test_features.txt", "w+")

i = 0
start = time.time()

# for file in tqdm(bucket_list[0:5000]):

for file in bucket_list[0:501]:
    try:
        f = BytesIO(file.download_as_string())
        img = image.load_img(f, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        features_reduce = features.squeeze()
        X_test.append(features_reduce)
        file_name = file.path.split("/o/")[1].split("%2F")
        if file_name[0] == "train":
            # print(file.path.split('/o/')[1].split('%2F')[1])
            train_filenames.write(file_name[1] + "\n")
            train_featues.write(" ".join(str(x) for x in features.squeeze()) + "\n")
        else:
            test_filenames.write(file_name[1] + "\n")
            test_features.write(" ".join(str(x) for x in features.squeeze()) + "\n")
        i = i + 1
        # if i % 100:
        #     print(i)
    except:
        pass

print(i)
end = time.time()
print("\n\ntime spend: ", (end - start) / 60, " minutes \n\n")

train_filenames.close()
test_filenames.close()
train_featues.close()
test_features.close()

# my_file = open('test_filenames.txt', 'r')
# print(my_file.read())
# my_file.close()

# sum(1 for line in open('test_filenames.txt'))
# file_len('test_filenames.txt')


# xb for the database, that contains all the vectors that must be indexed, and that we are going to search in. Its size is nb-by-d
# xq for the query vectors, for which we need to find the nearest neighbors. Its size is nq-by-d. If we have a single query vector, nq=1.

# need to contactinate tests as well
xb = np.load("train_filenames.txt")
xq = np.load("train_featues.txt")
print(xb.shape)
print(xq.shape)


print(xq.shape)
import faiss

res = faiss.StandardGpuResources()  # use a single GPU

# build a flat (CPU) index
index_flat = faiss.IndexFlatL2(d)
# make it into a gpu index
gpu_index_flat = faiss.index_cpu_to_gpu(res, 0, index_flat)

gpu_index_flat.add(xb)  # add vectors to the index
print(gpu_index_flat.ntotal)

k = 100  # we want to see 4 nearest neighbors
D, I = gpu_index_flat.search(xq, k)  # actual search
# print(I[:5])                   # neighbors of the 5 first queries
# print(I[-5:])                  # neighbors of the 5 last queries

np.save("output/I.npy", I)
np.save("output/D.npy", D)

### make submission
index_path = "input/index/"
index_list = sorted(glob.glob(index_path + "*"))  # 1091756
index_list = pd.DataFrame(index_list, columns=["id"])
index_list["id"] = index_list["id"].apply(lambda x: os.path.basename(x)[:-4])
index_list = np.array(index_list["id"])
query_path = "input/query/"
query_list = sorted(glob.glob(query_path + "*"))  # 114943

sub = pd.DataFrame(query_list, columns=["id"])
sub["id"] = sub["id"].apply(lambda x: os.path.basename(x)[:-4])

images_list = index_list[I]
images_list = images_list + " "
images_list = np.sum(images_list, axis=1)

sub["images"] = images_list
sub2 = pd.read_csv("input/sample_submission.csv")
sub2["images"] = ""
sub = pd.concat([sub, sub2])
sub = sub.drop_duplicates(["id"])
# sub.to_csv("output/sub_{}_{}.csv".format(model_name, feature_layer), index=None)
sub.to_csv("output/resnet50_output.csv", index=None)
