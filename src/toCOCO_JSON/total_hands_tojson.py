import json
import os
import cv2
import csv

test_flag = 1
dataset = {'categories': [], 'images': [], 'annotations': []}
class_file = 'classes.txt'
with open(class_file) as f:
    classes = f.read().strip().split()
    for i, cls in enumerate(classes, 1):
        dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'mark'})

root = 'images_total'

print("root:", root)
train_root_path = '{}/train'.format(root)
test_root_path = '{}/test'.format(root)
train_labels = '{}/train_labels.csv'.format(root)
test_labels = '{}/test_labels.csv'.format(root)

if test_flag == 0:
    indexes = [f for f in os.listdir(train_root_path)]
    csvFile = open(train_labels, "r")

else:
    indexes = [f for f in os.listdir(test_root_path)]
    csvFile = open(test_labels, "r")

reader = csv.reader(csvFile)

annos = []
for item in reader:
    if item[0] == '' or len(item) == 0:
        continue
    print(item)
    annos.append([item[0], 1, int(float(item[4])), int(float(item[5])), int(float(item[6])), int(float(item[7]))])
csvFile.close()

for k, index in enumerate(indexes):
    print(index)
    # 用opencv读取图片，得到图像的宽和高
    if test_flag == 0:
        im = cv2.imread(os.path.join(train_root_path, index))
    else:
        im = cv2.imread(os.path.join(test_root_path, index))

    height, width, _ = im.shape

    # 添加图像的信息到dataset中
    dataset['images'].append({'file_name': index,
                                'id': k,
                                'width': width,
                                'height': height})

    for ii, anno in enumerate(annos):

        # 如果图像的名称和标记的名称对上，则添加标记
        if anno[0] == index:
            cls_id = anno[1]
            x1 = int(anno[2])
            y1 = int(anno[3])
            x2 = int(anno[4])
            y2 = int(anno[5])
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            dataset['annotations'].append({
                'area': width * height,
                'bbox': [x1, y1, width, height],
                'category_id': int(cls_id),
                'id': ii,
                'image_id': k,
                'iscrowd': 0,
                # mask, 矩形是从左上角点按顺时针的四个顶点
                'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
            })
if test_flag == 0:
    json_name = os.path.join('hands_train.json')
else:
    json_name = os.path.join('hands_val.json')

with open(json_name, 'w') as f:
    json.dump(dataset, f)
