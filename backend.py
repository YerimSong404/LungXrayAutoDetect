from flask import Flask, request, render_template, redirect, send_file
import glob
import os, json
import numpy as np
import pandas as pd
from keras.models import load_model
from werkzeug.utils import secure_filename
import predict

app = Flask(__name__)

app.config['IMG_FOLDER'] = os.path.join('static', 'images')

# imgpaths2 = ('./saved_image')

model = load_model('model.h5')
resultSavePath = './detect_result/'
# 경로 체크하기
@app.route('/')
def main():
        return render_template("index2.html")

@app.route('/fileupload', methods=['POST'])
def fileupload():
    if request.method == 'POST':
        f = request.files['file']
        file_path = './saved_image/' + secure_filename(f.filename)
        print("file Path: " + file_path)
        f.save(file_path)
        #f.save('./saved_image/' + secure_filename(f.filename))
        return 'upload Success'
        #return render_template('index.html')

@app.route('/detect', methods=['GET'])
def detect():
    # img = request.files['file']
    img = request.args.get('file_path')
    print("img : "+ img)
    if img is None:
        return 'Img Not Found'
    img_path = './saved_image/' + img.strip()
    print("img_path :" + img_path)
    if img_path is None:
        return 'File not found'

    path = img.replace('png', 'txt')

    result_path = os.path.join('./detect_result/', path)
    print("result Path : "+result_path)
    if os.path.exists(result_path):
        return 'Result Exist'

    processed_img = predict.processing_image(img_path) # 이미지 전처리
    prediction = model.predict(processed_img) # 탐지
    df_txt = predict.result_output_(prediction) # 탐지 결과 txt

    # result_path = img.filename.split('/')[8]
    # result_path = os.path.basename(img).split('/')[8]
    result_path = os.path.basename(img)

    print(result_path)

    txtname = resultSavePath + result_path
    txtname = txtname.replace('png', 'txt')
    print("txt Name : "+ txtname)
    if not os.path.exists(txtname):
        os.makedirs(resultSavePath, exist_ok=True)

    df_txt.to_csv(txtname, sep = ' ', index = False, header = False)

    return 'detected'
    #return redirect('/showing')

@app.route('/showing', methods=['GET'])
def showing():

    file_list = glob.glob('./detect_result/*.txt')
    # for file_path in file_list:
    #    print(file_path)
    if file_list is None:
        print("File Not Exist")
    else:

        last_file = max(file_list, key = os.path.getctime)
        print(last_file)
    with open(last_file, encoding='utf-8') as f:
        lines = f.read()
        '''
        result = lines.split(' ')
        col = ['d_label', 'd_name', 'percent']
        
        result = dict(zip(col, result))
        info = json.dumps(result)
        '''
        result = lines.split('\n')  # 결과를 줄 단위로 분리
        col = ['d_label', 'd_name', 'percent']
        result_dict = []

        for line in result:
            parts = line.split(' ')  # 각 줄을 공백으로 분리하여 결과 추출
            if len(parts) == 3:
                d_label, d_name, percent = parts
                result_dict.append({
                    'd_label': d_label,
                    'd_name': d_name,
                    'percent': float(percent)
                })

        info = json.dumps(result_dict)
        return info
        #return render_template("index.html", info = info)


@app.route('/showing2', methods=['GET'])
def showing2():
    file = request.args.get('file_name')
    file = file.replace('png', 'txt')
    file_path = './detect_result/' + file
    if not file_path:
        print("File Not Exist")
    else :
        print(file_path)

    with open(file_path, encoding='utf-8') as f:
        lines = f.read()
    result = lines.split('\n')  # 결과를 줄 단위로 분리
    col = ['d_label', 'd_name', 'percent']
    result_dict = []

    for line in result:
        parts = line.split(' ')  # 각 줄을 공백으로 분리하여 결과 추출
        if len(parts) == 3:
            d_label, d_name, percent = parts
            result_dict.append({
                'd_label': d_label,
                'd_name': d_name,
                'percent': float(percent)
            })
    info = json.dumps(result_dict)
    return info

@app.route('/download/<filename>', methods=['GET'])
def download(file_name):
    file = request.args.get('file_name')
    file_path = f'./detect_result/{file}'

    try:
        # 파일 다운로드
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host='localhost', port='5000', debug=True)
    #app.run(host='0.0.0.0', port=8000, debug=True)
