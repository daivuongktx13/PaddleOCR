import os 
import glob
import re
import json
import numpy as np

gt_path1 = 'train_data/bkainaver/train_data/training_gt/*.txt'
image_path1 = 'bkainaver/train_data/training_img'


gt_files = glob.glob(gt_path1)
with open('train_data/bkainaver/train_data.txt', 'w') as o_file:
    for gt_file in gt_files:
        index = re.findall('\d+', gt_file)[0]
        with open(gt_file, 'r') as file:
            data = [line.strip().split(',') for line in file.readlines()]
            data = [row[0:8]+ [",".join(row[8:])] for row in data]
            annotations = []
            for row in data:
                points = np.array(row[0:8], dtype=int).reshape(-1, 2)
                annotations.append({
                    'transcription': row[-1],
                    'points': points.tolist()
                })
            annotations = json.dumps(annotations, ensure_ascii=False)
            filename = os.path.join(image_path1, f"img_{index}.jpg")
            o_file.write(f"{filename}\t{annotations}\n")

gt_path2 = 'train_data/bkainaver/vietnamese/labels/*.txt'
image_path2_train = 'bkainaver/vietnamese/train_images'
image_path2_valid = 'bkainaver/vietnamese/test_image'
image_path2_test = 'bkainaver/vietnamese/unseen_test_images'

gt_files = glob.glob(gt_path2)

train_o = open('train_data/bkainaver/vietnamese_train.txt', 'w')
test_o = open('train_data/bkainaver/vietnamese_test.txt', 'w')
for gt_file in gt_files:
    index = int(re.findall('\d+', gt_file)[0])
    with open(gt_file, 'r') as file:
        data = [line.strip().split(',') for line in file.readlines()]
        data = [row[0:8]+ [",".join(row[8:])] for row in data]
        annotations = []
        for row in data:
            points = np.array(row[0:8], dtype=int).reshape(-1, 2)
            annotations.append({
                'transcription': row[-1],
                'points': points.tolist()
            })
        annotations = json.dumps(annotations, ensure_ascii=False)
        if index <= 1200:
            filename = os.path.join(image_path2_train, "im{:04d}.jpg".format(index))
            train_o.write(f"{filename}\t{annotations}\n")
        elif index <= 1500:
            filename = os.path.join(image_path2_valid, "im{:04d}.jpg".format(index))
            train_o.write(f"{filename}\t{annotations}\n")
        else:
            filename = os.path.join(image_path2_test, "im{:04d}.jpg".format(index))
            test_o.write(f"{filename}\t{annotations}\n")
train_o.close()
test_o.close()

        
    


    

