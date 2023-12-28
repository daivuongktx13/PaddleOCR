import os 
import glob
import numpy as np
import json

label_path = 'train_data/unrealtext/*/labels/*.json'


gt_files = glob.glob(label_path)
with open('train_data/unrealtext/unrealtext.txt', 'w') as o_file:
    for gt_file in gt_files:
        with open(gt_file, 'r') as file:
            data = json.load(file)
        path_write = "/".join(gt_file.split('/')[1:-2])
        path_write = os.path.join(path_write, data['imgfile'])
        annotations = []
        for bbox, text in zip(data['bbox'], data['text']):
            points = np.array(bbox).reshape(-1, 2)
            annotations.append({
                'transcription': text,
                'points': points.tolist()
            })
        annotations = json.dumps(annotations, ensure_ascii=False)
        o_file.write(f"{path_write}\t{annotations}\n")
