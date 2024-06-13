import os
from glob import glob
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from sklearn.utils import compute_class_weight

df = pd.read_csv('./dataset/Data_Entry_2017.csv')
print(df.head(5))
bbox_path = pd.read_csv('/dataset/BBox_List_2017.csv')

#이미지 가져오기
imgs_paths = {os.path.basename(x): x for x in
                   glob(os.path.join('.', 'dataset','images*', 'images', '*.png'))}
print('Scans found:', len(imgs_paths)) #112120개

#이미지 경로 추출
df['Image Path'] = df['Image Index'].map(imgs_paths.get)
#환자 나이 전처리 5<n<100
df['Patient Age'] = np.clip(df['Patient Age'], 5, 100)
#print(df.head())

df = df[['Image Path', 'Finding Labels']]
print(df.head())

#라벨 추출
from itertools import chain
labels = np.unique(list(chain(*df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
print(labels)

#진단 여부 기록
for label in labels:
    if len(label)>1:
        df[label] = df['Finding Labels'].map(lambda finding: 1.0 if label in finding else 0)

columns = df.columns[2:]
print(columns)

dfs = list()

#증상별 분리
for col in columns:
    one_df = df[df[col] == 1.0]
    dfs.append(one_df)

dfs_ds = list()

#증상별 n개씩 선별
for one_df in dfs:
    dfs_ds.append(one_df.head(200))
df_ds = pd.concat(dfs_ds)

df_ds.to_csv('./down_sampled_df.csv')

df_ds = pd.read_csv('./down_sampled_df.csv')
print('Total Images: ', df_ds['Image Path'].count())

SIZE = 224
x_dataset = []
i = 0

#이미지 전처리
for path in df_ds['Image Path']:
    i = i + 1
    img = image.load_img(path, target_size=(SIZE, SIZE, 3))
    img = image.img_to_array(img)
    img = img / 255.
    x_dataset.append(img)

# 훈련세트, 테스트 세트 분할 
x = np.array(x_dataset)
y = np.array(df_ds.drop(['Unnamed: 0', 'Image Path', 'Finding Labels'], axis=1))
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=20, test_size=0.3)

#VGG16 pretrained_model 사용
pretrained_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False ,input_shape=x.shape[1:])

#모델 생성
model = tf.keras.Sequential([
    pretrained_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(15, activation = 'relu')
])

# Adam옵티마이저 / 이진 교차 엔트로피
model.compile(
    optimizer='adam',
    loss = 'binary_crossentropy',
    metrics=['binary_accuracy']
)
# 클래스 가중치 계산
class_weights = compute_class_weight('balanced', np.unique(np.argmax(y_train, axis=1)), np.argmax(y_train, axis=1))
class_weight_dict = dict(enumerate(class_weights))

model.summary()

history = model.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test), batch_size=64, class_weight=class_weight_dict)

model.save("model.h5")

_, acc = model.evaluate(x_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

'''
MODEL_DIR = "/model"
version = 1
export_path = "."+os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)
'''
