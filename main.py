import os
import pandas as pd
from tqdm.auto import tqdm
import shutil as sh

git clone https://github.com/ultralytics/yolov5
pip install -U pycocotools
pip install -qr yolov5/requirements.txt
cp yolov5/requirements.txt ./

img_h, img_w= (380, 676)
df = pd.read_csv('C:\\Users\\pawko\\PycharmProjects\\biai\\input\\car-object-detection\\data\\train_solution_bounding_boxes (1).csv')
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
    if not os.path.exists("C:\\Users\\pawko\\PycharmProjects\\biai\\labels\\"+path2save):
        os.makedirs("C:\\Users\\pawko\\PycharmProjects\\biai\\labels\\"+path2save)
    with open("C:\\Users\\pawko\\PycharmProjects\\biai\\labels\\"+path2save+name+".txt", 'w+') as f:
        row = mini[['classes','x_center','y_center','w','h']].astype(float).values
        row = row.astype(str)
        for j in range(len(row)):
            text = ' '.join(row[j])
            f.write(text)
            f.write("\n")
    if not os.path.exists("C:\\Users\\pawko\\PycharmProjects\\biai\\images\\{}".format(path2save)):
        os.makedirs("C:\\Users\\pawko\\PycharmProjects\\biai\\images\\{}".format(path2save))
    sh.copy("C:\\Users\\pawko\\PycharmProjects\\biai\\input\\car-object-detection\\data\\{}\\{}.jpg".format(source,name),"C:\\Users\\pawko\\PycharmProjects\\biai\\images\\{}\\{}.jpg".format(path2save,name))



py yolov5\\train.py --batch 16 --epochs 10 --data C:\\Users\\pawko\\PycharmProjects\\biai\\cardetect.yaml --cfg yolov5\\models\\yolov5s.yaml --name yolov5presentation --cache --nosave <-- uczenie sie

py yolov5\\detect.py --weights yolov5\\runs\\yolov5s.pt --conf 0.4 --source "C:\\Users\\pawko\\Desktop\\Zrzut ekranu 2022-06-20 142738.png" <-- odpalenie zdjecia ze stop chama