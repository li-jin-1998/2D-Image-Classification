import os
import shutil

from sklearn.model_selection import train_test_split

origin_path = r'/mnt/algo_storage_server/ScanSceneClassification/data'

result_path = r'/mnt/algo_storage_server/ScanSceneClassification/dataset'

if os.path.exists(result_path):
    shutil.rmtree(result_path)
os.mkdir(result_path)

train_path = os.path.join(result_path, 'train')
test_path = os.path.join(result_path, 'test')

os.mkdir(train_path)
os.mkdir(test_path)

for item in ['intra', 'extra']:
    os.mkdir(os.path.join(train_path, item))
    os.mkdir(os.path.join(test_path, item))
    paths = os.listdir(os.path.join(origin_path, item))
    train, test = train_test_split(paths, test_size=0.3, random_state=12)

    print(len(train), len(test))

    for path in train:
        train_origin_path = os.path.join(origin_path, item, path)
        if item == 'intra':
            for p in os.listdir(train_origin_path)[::6]:
                shutil.copy(os.path.join(train_origin_path, p), os.path.join(train_path, item, path + '_' + p))
        else:
            for p in os.listdir(train_origin_path)[::3]:
                shutil.copy(os.path.join(train_origin_path, p), os.path.join(train_path, item, path + '_' + p))

    for path in test:
        test_origin_path = os.path.join(origin_path, item, path)
        if item == 'intra':
            for p in os.listdir(test_origin_path)[::6]:
                shutil.copy(os.path.join(test_origin_path, p), os.path.join(test_path, item, path + '_' + p))
        else:
            for p in os.listdir(test_origin_path)[::3]:
                shutil.copy(os.path.join(test_origin_path, p), os.path.join(test_path, item, path + '_' + p))
