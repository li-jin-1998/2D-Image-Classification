import json
import os
import shutil
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime
import tqdm

from parse_args import parse_args, getModel

# read class_indict
json_path = './class_indices.json'
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

with open(json_path, "r") as f:
    class_indict = json.load(f)
print(class_indict)

error_path = './onnx/'
if os.path.exists(error_path):
    shutil.rmtree(error_path)
os.mkdir(error_path)

# Load ONNX model
args = parse_args()
model = getModel(args)
onnx_file_name = "./weights/{}_best_model_ssr.onnx".format(args.arch)
simplify_onnx_file_name = "./weights/{}_best_model_ssr_simplify.onnx".format(args.arch)
session = onnxruntime.InferenceSession(onnx_file_name)
start_time = time.time()
# x = random.randint(1, 10000)
x = 1000
print(x)

# load image
for item in ['extra', 'intra'][::-1]:
    os.mkdir(os.path.join(error_path, item))
    print(item)
    path = "/mnt/algo_storage_server/ScanSceneClassification/dataset/test/" + item
    paths = os.listdir(path)[x:x + 1000]
    s = len(paths)
    wrong = 0
    for i in tqdm.tqdm(range(s)):
        img_path = os.path.join(path, paths[i])
        assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)

        origin_image = cv2.imread(img_path)
        # print(origin_image.shape)
        origin_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(origin_image, (192, 192), interpolation=cv2.INTER_CUBIC)
        # image = cv2.bilateralFilter(image, 2, 50, 50)  # remove images noise.
        # img = cv2.applyColorMap(img, cv2.COLORMAP_BONE)  # produce a pseudocolored image. 伪彩色
        image = np.array(image, np.float32)
        image = image / 127.5 - 1

        image = np.expand_dims(image, 0)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})
        result = np.squeeze(result)
        # print(result)
        # Compare ONNX model output with golden image
        final_result = class_indict[str(result.argmax(0))]
        if final_result == item:
            continue
        else:
            wrong += 1
        # if True:
        #     print(img_path)
        #     for j in range(len(result)):
        #         print("class: {:10}   prob: {:.4}".format(class_indict[str(j)], result[j]))
        shutil.copy(img_path, os.path.join(error_path, paths[i]))
        plt.figure()
        plt.imshow(origin_image)
        print_res = "class: {:10}   prob: {:.4}".format(class_indict[str(result.argmax(0))], result[result.argmax(0)])
        plt.title(print_res)

        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(error_path, item, paths[i]), dpi=100)
        # plt.show()
        plt.close()
    print("{} acc:{:.4f}".format(item, (s - wrong) / s * 100))
total_time = time.time() - start_time
print("time {}s, fps {}".format(total_time, 100 / total_time))
