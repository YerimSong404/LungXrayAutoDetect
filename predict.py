import os.path

import pandas as pd
from keras.preprocessing import image

import numpy as np
import cv2

SIZE = 224
target_size = (224,224,3)
#img_path = './dataset/images_001/images/00001286_005.png'


val1 = np.array(['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 'Emphysema', 'Fibrosis',
                'Hernia', 'Infiltration', 'Mass', 'No Finding', 'Nodule','Pleural_Thickening', 'Pneumonia',
                 'Pneumothorax'])
val2 = np.array(['무기폐', '심비대증', '폐경화', '부종', '삼출', '기종', '섬유증',
                 '탈장', '침투', '종양', '발견항목없음', '결절', '흉막삼출', '폐렴', '기흉'])

bbox_path = pd.read_csv('./dataset/BBox_List_2017.csv')
def processing_image(img_path):
    img = image.load_img(img_path)
    img = img.resize((SIZE, SIZE))

    img = image.img_to_array(img)
    img = img/225.
    img = np.expand_dims(img, axis = 0)
    return img

'''
def result_output(predict):
    sorted_prediction = np.argsort(prediction[0])[:-11:-1]
    result = ""
    for i in range(5):
        line = ("{}".format(val1[sorted_prediction[i]]) + "({})".format(val2[sorted_prediction[i]]) + "({:.3})".format(prediction[0][sorted_prediction[i]]))
        result += line
    return result
'''

def bounding_box(img_path):
    image = cv2.imread(img_path)
    img_index = os.path.basename(img_path).split('.')[0]

    bbox_info = bbox_path[bbox_path['img_ind'] == img_index]

    if not bbox_info.empty:
        # 바운딩 박스 정보 추출
        x, y, w, h = bbox_info.iloc[0, 1:].astype(int)

        # 이미지에 바운딩 박스 그리기
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return image

def result_output_(predict):
    sorted_prediction = np.argsort(predict[0])[:-15:-1]
    result_data = []
    for i in range(5):
        disease_label = val1[sorted_prediction[i]]
        disease_name = val2[sorted_prediction[i]]
        percentage = round(predict[0][sorted_prediction[i]] *2, 3)
        result_data.append([disease_label, disease_name, percentage])

    df = pd.DataFrame(result_data, columns=['disease_label', 'disease_name', 'percentage'])
    return df


