import datetime
import os
import shutil
import time

import cv2
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json

import torch
import tqdm
import matplotlib.pyplot as plt
from parse_args import parse_args, get_model


def predict(args, verbose=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)
    print(class_indict)
    # create model
    model = get_model(args)
    # load model weights
    model_weight_path = "./weights/{}_best_model.pth".format(args.arch)
    model.load_state_dict(torch.load(model_weight_path, map_location=device))

    error_path = './error/' + args.arch
    if os.path.exists(error_path):
        shutil.rmtree(error_path)
    os.mkdir(error_path)

    start_time = time.time()
    # load image
    for item in ['extra', 'intra'][::-1]:
        os.mkdir(os.path.join(error_path, item))
        print(item)
        path = "/mnt/algo_storage_server/ScanSceneClassification/dataset/test/" + item
        paths = os.listdir(path)
        s = len(paths)
        wrong = 0
        for i in tqdm.tqdm(range(s)):
            img_path = os.path.join(path, paths[i])
            assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
            # origin_img = Image.open(img_path)
            # img = torch.Tensor(preprocessing(origin_img, args.image_size))
            origin_image = cv2.imread(img_path)
            # print(origin_image.shape)
            origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(origin_image, (192, 192), interpolation=cv2.INTER_CUBIC)
            image = np.array(image, np.float32)
            image = image / 127.5 - 1
            img = torch.Tensor(image)
            img = img.permute(2, 0, 1)
            # expand batch dimension
            img = torch.unsqueeze(img, dim=0)
            model.eval()
            with torch.no_grad():
                # predict class
                output = torch.squeeze(model(img.to(device))).cpu()
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()
            if class_indict[str(predict_cla)] == item:
                continue
            else:
                wrong += 1
            if verbose:
                print(img_path)
                for j in range(len(predict)):
                    print("class: {:10}   prob: {:.4}".format(class_indict[str(j)],
                                                              predict[j].numpy()))
            plt.figure()
            plt.imshow(origin_image)
            print_res = "class: {}   prob: {:.4}".format(class_indict[str(predict_cla)],
                                                         predict[predict_cla].numpy())
            plt.title(print_res)

            plt.xticks([])
            plt.yticks([])
            plt.savefig(os.path.join(error_path, item, paths[i]), dpi=100)
            # plt.show()
            plt.close()
        print("{} acc:{:.4f}".format(item, (s - wrong) / s * 100))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("training time {}".format(total_time_str))


if __name__ == '__main__':
    print('>' * 10, 'predict')
    args = parse_args()

    predict(args, False)
