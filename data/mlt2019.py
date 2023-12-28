import os 
import glob
import re
import json
import numpy as np

image_path1 = 'mlt2019/ImagesPart1'
image_path2 = 'mlt2019/ImagesPart2'

gt_path = 'train_data/mlt2019/Labels/*.txt'

gt_files = glob.glob(gt_path)
with open('train_data/mlt2019/mlt2019.txt', 'w') as o_file:
    for gt_file in gt_files:
        index = int(re.findall('\d+', os.path.basename(gt_file))[0])
        with open(gt_file, 'r') as file:
            data = [line.strip().split(',') for line in file.readlines()]
            data = [row[0:9]+ [",".join(row[9:])] for row in data]
            annotations = []
            for row in data:
                points = np.array(row[0:8], dtype=int).reshape(-1, 2)
                annotations.append({
                    'transcription': row[-1],
                    'points': points.tolist()
                })
            annotations = json.dumps(annotations, ensure_ascii=False)
            if index <= 5000:   
                if os.path.exists(os.path.join('train_data', image_path1, "tr_img_{:05d}.jpg".format(index))):   
                    filename = os.path.join(image_path1, "tr_img_{:05d}.jpg".format(index))
                elif os.path.exists(os.path.join('train_data', image_path1, "tr_img_{:05d}.png".format(index))):
                    filename = os.path.join(image_path1, "tr_img_{:05d}.png".format(index))
                else:
                    filename = os.path.join(image_path1, "tr_img_{:05d}.gif".format(index))
            else:
                if os.path.exists(os.path.join('train_data', image_path2, "tr_img_{:05d}.jpg".format(index))):   
                    filename = os.path.join(image_path2, "tr_img_{:05d}.jpg".format(index))
                elif os.path.exists(os.path.join('train_data', image_path2, "tr_img_{:05d}.png".format(index))):
                    filename = os.path.join(image_path2, "tr_img_{:05d}.png".format(index))
                else:
                    filename = os.path.join(image_path2, "tr_img_{:05d}.gif".format(index))

            o_file.write(f"{filename}\t{annotations}\n")
