import os 
import glob
import numpy as np
import json


label_path = 'train_data/synthtext3d/Synth3D-10K/label/*'
path_write = 'synthtext3d/Synth3D-10K/img'

def toPoints(text):
    return map(float, text.strip().split(','))

gt_files = glob.glob(label_path)
with open('train_data/synthtext3d/synthtext3d.txt', 'w') as o_file:
    for gt_file in gt_files:
        name = os.path.basename(gt_file).split('.')[0]
        with open(gt_file, 'r') as file:
            data = file.readlines()
        data = np.array(data).reshape((-1, 5)).tolist()
        annotations = []
        for row in data:
            p1 = [float(x) for x in row[0].strip().split(',')]
            p2 = [float(x) for x in row[1].strip().split(',')]
            p3 = [float(x) for x in row[2].strip().split(',')]
            p4 = [float(x) for x in row[3].strip().split(',')]
            is_hard = row[4].strip()
            points = np.array([p1, p2, p3, p4]).reshape(-1, 2)
            if not np.any(points):
                continue
            annotations.append({
                'transcription': row[-1].strip(),
                'points': points.tolist()
            })
        annotations = json.dumps(annotations, ensure_ascii=False)
        filename = os.path.join(path_write, name + '.jpg')
        o_file.write(f"{filename}\t{annotations}\n")
