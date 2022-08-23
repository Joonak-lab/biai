import os
import pandas as pd
from tqdm.auto import tqdm
import shutil as sh

## requirements ##
#git clone https://github.com/ultralytics/yolov5
#pip install -U pycocotools
#pip install -qr yolov5/requirements.txt
#cp yolov5/requirements.txt ./

img_h, img_w= (380, 676)
df = pd.read_csv(".\\input\\car-object-detection\\data\\train_solution_bounding_boxes (1).csv")
df['image'] = df['image'].apply(lambda x: x.split('.')[0])
df['x_center'] = (df['xmin'] + df['xmax'])/2
df['y_center'] = (df['ymin'] + df['ymax'])/2
df['w'] = df['xmax'] - df['xmin']
df['h'] = df['ymax'] - df['ymin']
df['classes'] = 0
df['x_center'] = df['x_center']/img_w
df['w'] = df['w']/img_w
df['y_center'] = df['y_center']/img_h
df['h'] = df['h']/img_h
df.head()

index = list(set(df.image))

source = 'training_images'
val_index = index[0:len(index)//5]
for name,mini in tqdm(df.groupby('image')):
    if name in val_index:
        path2save = "val2017\\"
    else:
        path2save = "train2017\\"
    if not os.path.exists(".\\yolov5\\labels\\"+path2save):
        os.makedirs(".\\yolov5\\labels\\"+path2save)
    with open(".\\yolov5\\labels\\"+path2save+name+".txt", 'w+') as f:
        row = mini[['classes','x_center','y_center','w','h']].astype(float).values
        row = row.astype(str)
        for j in range(len(row)):
            text = ' '.join(row[j])
            f.write(text)
            f.write("\n")
    if not os.path.exists(".\\yolov5\\images\\{}".format(path2save)):
        os.makedirs(".\\yolov5\\images\\{}".format(path2save))
    sh.copy(".\\input\\car-object-detection\\data\\{}\\{}.jpg".format(source,name),".\\yolov5\\images\\{}\\{}.jpg".format(path2save,name))


##TRAINING MODEL##
#py yolov5\\train.py --batch 16 --epochs 10 --data cardetect.yaml --cfg yolov5\\models\\yolov5s.yaml --name yolov5presentation --cache --nosave

##TESTING##
#py yolov5\\detect.py --weights yolov5\\runs\\yolov5s.pt --conf 0.4 --source "Path to images"