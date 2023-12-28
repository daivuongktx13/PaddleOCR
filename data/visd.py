import os 
import glob
import re
import json
import numpy as np

image_path1 = 'visd/10K/image'

gt_path = 'train_data/visd/10K/text/*.txt'

gt_files = glob.glob(gt_path)
with open('train_data/visd/visd.txt', 'w') as o_file:
    for gt_file in gt_files:
        name = os.path.basename(gt_file).split('.')[0]
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

            filename = os.path.join(image_path1, name + '.jpg')

            o_file.write(f"{filename}\t{annotations}\n")
