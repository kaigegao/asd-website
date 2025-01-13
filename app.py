import os
import datetime
from datetime import datetime, timedelta
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, send_file
from werkzeug.utils import secure_filename
import zipfile
import csv

import torch
import numpy as np
from models.MyModel import MyModel  # Ensure this import matches your directory structure
from flask_cors import CORS


from nilearn.image import load_img
from nilearn import masking
from nilearn.image import resample_to_img
from nilearn.input_data import NiftiLabelsMasker
from nilearn import plotting, datasets
from nilearn import image
import nibabel as nib
import matplotlib

matplotlib.use("TkAgg")


app = Flask(__name__)
app.secret_key = 'supersecretkey'
CORS(app)
# 允许上传的文件类型
ALLOWED_EXTENSIONS = {'csv', 'txt'}
# 存储问卷数据的列表
surveys = []
# 存储用户数据的列表
users = []
UPLOAD_FOLDER = "./uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER  # 正确设置配置变量
app.config['case_save_file'] = "./cases_save_file.csv"
app.config['user_info_file'] = "./user_info_file.csv"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
cases = []
# global viewing_file


# Configuration class for model parameters
class DefaultConfig(object):
    dataset = "Kaggle"
    d_model = 1024
    n_layer = 24
    ssm_cfg = dict()
    norm_epsilon = 1e-5
    rms_norm = True
    residual_in_fp32 = True
    fused_add_norm = True
    initializer_cfg = None
    lr = 4.3895647763297976e-05
    hidden_1 = 1024
    hidden_2 = 1024
    batch_size = 32
    num_workers = 0
    embed = "fixed"
    freq = "h"
    dropout = 0.3762915331187696
    num_class = 2
    pred_len = 0
    if dataset == "Kaggle":
        enc_in = 110
        seq_len = 176
    elif dataset == "ABIDE_preprocessed":
        enc_in = 111
        seq_len = 316
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    model_path = "models/Mamba_GCN_OF_dmodel_1024_nlayer_24_lr_4.3895647763297976e-05_dropout_0.3762915331187696_hidden1_1024_hidden2_1024.pth"

args = DefaultConfig()
model = MyModel(args)
state_dict = torch.load(args.model_path, map_location=args.device)
model.load_state_dict(state_dict)
model.to(args.device)
model.eval()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/team')
def team():
    return render_template('team.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # 检查是否有文件部分
        if 'data_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['data_file']
        # 如果用户没有选择文件，浏览器也会提交一个空的文件名
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            # 处理上传的文件（例如：读取数据、存储到数据库等）
            process_uploaded_file(file_path)
            return f'File {filename} has been uploaded successfully.'
    return render_template('upload.html')


def process_uploaded_file(file_path):
    # 这里可以添加处理上传文件的逻辑
    print(f"Processing file: {file_path}")
    # 示例：读取CSV文件并打印内容
    data = pd.read_csv(file_path)
    # print(data.head())


@app.route('/survey')
def survey():
    return render_template('survey.html')

def get_next_case_id(filename):
    if not os.path.exists(filename):
        return 0
    try:
        existing_df = pd.read_csv(filename)
        if existing_df.empty:
            return 0
        else:
            return existing_df['caseId'].max() + 1
    except FileNotFoundError:
        print('FileNotFound')
        return 0
    except pd.errors.EmptyDataError:
        print('EmptyData')
        return 0
    except pd.errors.ParserError:
        print('ParserError')
        return 0

@app.route('/api/upload_case', methods=['POST'])
def file_upload_destination():
    file = request.files.get("case-file")
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config.get("UPLOAD_FOLDER"), filename)
        file.save(file_path)

        # 读取文件内容，将第一列设置为索引
        data = pd.read_csv(file_path)
        # 假设你有一个预测函数 predict
        diagnosis, risk = predict(data.values)

        data = {
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'name': request.form.get("name"),
            'fmri.image': "None",
            'file': filename,
            'uploadDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            #保留预测功能
            'diagnosis':diagnosis,
            # 保留预测功能
            'risk':str(risk),
            'doctor':session['username']
        }
        cases.append(data)
        file_path = app.config.get("case_save_file")
        case_id = get_next_case_id(file_path)

        data_with_case_id = {'caseId': case_id, **data}
        columns_order = ['caseId', 'age', 'gender', 'name', 'fmri.image','file', 'uploadDate','diagnosis','risk','doctor']
        df = pd.DataFrame([data_with_case_id], columns=columns_order)
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(file_path, index=False)

        except pd.errors.EmptyDataError:
            df.to_csv(file_path,  index=False)

        return {'success': True, 'result': str(case_id)}, 200

    except Exception as e:
        return {'success': False, 'errorMsg': f'Error reading file {"caseInfo"}: {str(e)}'}, 200


@app.route('/upload_case', methods=['GET', 'POST'])
def upload_case():
    return render_template('upload_case.html')






@app.route('/return_to_upload')
def return_to_upload():
    return redirect(url_for('upload_case'))


@app.route('/bulk_upload_cases')
def bulk_upload_cases():
    return render_template('bulk_upload_cases.html')

bulk_cases = []
@app.route('/bulk_upload', methods=['GET', 'POST'])
def upload_files():

    if request.method == 'POST':

        zip_file = request.files.get('zipFile')
        csv_file = request.files.get('csvFile')

        if not zip_file or not csv_file:
            flash('No file part')
            # return redirect(request.url)
            return "success"

        if zip_file.filename.endswith('.zip') and csv_file.filename.endswith('.csv'):

            zip_filename = secure_filename(zip_file.filename)
            csv_filename = secure_filename(csv_file.filename)

            zip_path = os.path.join(UPLOAD_FOLDER, zip_filename)
            csv_path = os.path.join(UPLOAD_FOLDER, csv_filename)

            zip_file.save(zip_path)
            csv_file.save(csv_path)

            upload_date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_contents = zip_ref.namelist()
                for item in zip_contents:
                    if item in get_csv_filenames(csv_path):
                        extracted_path = os.path.join(UPLOAD_FOLDER, item)
                        zip_ref.extract(item, UPLOAD_FOLDER)

                        cases.append({
                            'age': get_csv_row_value(csv_path, item, 'age'),
                            'gender': get_csv_row_value(csv_path, item, 'gender'),
                            'name': get_csv_row_value(csv_path, item, 'name'),
                            'file': item,
                            'uploadDate': upload_date
                        })
                        data={
                            'age': get_csv_row_value(csv_path, item, 'age'),
                            'gender': get_csv_row_value(csv_path, item, 'gender'),
                            'name': get_csv_row_value(csv_path, item, 'name'),
                            'file': item,
                            'uploadDate': upload_date
                        }
                        file_path = app.config.get("case_save_file")
                        case_id = get_next_case_id(file_path)
                        data_with_case_id = {'caseId': case_id, **data}
                        columns_order = ['caseId', 'age', 'gender', 'name', 'file', 'uploadDate']
                        df = pd.DataFrame([data_with_case_id], columns=columns_order)
                        try:
                            existing_df = pd.read_csv(file_path)
                            updated_df = pd.concat([existing_df, df], ignore_index=True)
                            updated_df.to_csv(file_path, index=False)
                        except pd.errors.EmptyDataError:
                            df.to_csv(file_path, index=False)
            os.remove(zip_path)
            os.remove(csv_path)

            flash('Files successfully processed')
            return redirect(url_for('upload_files'))

        else:
            flash('Invalid file format')
            return redirect(request.url)

    return render_template('bulk_upload2.html')


@app.route('/api/fmri_image_upload', methods=['POST'])
def upload_image_files():

    file = request.files.get("case-file")
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    try:
        filename = file.filename
        filename2 = filename.replace('.nii.gz', '')
        file_path = os.path.join(app.config.get("UPLOAD_FOLDER"), filename)

        file.save(file_path)
        convert_fmri_image_to_timeseries(file_path, filename2, app.config.get("UPLOAD_FOLDER"))

        file_path2 = os.path.join(app.config.get("UPLOAD_FOLDER"), filename2+ '_timeseries.csv')
        data = pd.read_csv(file_path2)
        diagnosis, risk = predict(data.values)

        data = {
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'name': request.form.get("name"),
            'fmri.image': filename,
            'file':filename2+ '_timeseries.csv',
            'uploadDate': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # 保留预测功能
            'diagnosis': diagnosis,
            # 保留预测功能
            'risk': str(risk),
            'doctor': session['username']
        }

        file_path = app.config.get("case_save_file")
        case_id = get_next_case_id(file_path)

        data_with_case_id = {'caseId': case_id, **data}
        columns_order = ['caseId', 'age', 'gender', 'name', 'fmri.image', 'file', 'uploadDate', 'diagnosis', 'risk',
                            'doctor']
        df = pd.DataFrame([data_with_case_id], columns=columns_order)
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(file_path, index=False)

        except pd.errors.EmptyDataError:
            df.to_csv(file_path, index=False)

        counter = 0
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        target_folder = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        os.makedirs(target_folder, exist_ok=True)
        img = load_img(file_path)
        num_time_points = img.shape[-1]
        for brain_index in range(0,num_time_points,int(num_time_points/5)):
            # 提取当前时间点的数据
            first_volume = image.index_img(img, brain_index)
            html_view = plotting.view_img(first_volume)
            save_name = f"brain_image_{counter}.html"
            final_path = os.path.join(target_folder, save_name)
            html_view.save_as_html(final_path)
            counter += 1

        return {'success': True, 'result': str(case_id)}, 200
    except Exception as e:
        return {'success': False, 'errorMsg': f'Error reading file {"caseInfo"}: {str(e)}'}, 200
    
@app.route('/api/query_fmri_image_html_content', methods=['POST'])
def query_fmri_image_html_content():
    data=request.get_json()
    index=data['index']
    name=data['image_name']
    app.config['UPLOAD_FOLDER']
    file_name = f"brain_image_{index}.html"
    target_folder = os.path.join(app.config['UPLOAD_FOLDER'], name)
    file_path = os.path.join(target_folder, file_name)
    return send_file(file_path, mimetype='text/html')

def convert_fmri_image_to_timeseries(image_path, file_name, fmri_save_path):

    dataset = datasets.fetch_atlas_aal()

    atlas_filename = dataset.maps
    labels = dataset.labels
    fMRIData = load_img(image_path)
    mask = masking.compute_background_mask(fMRIData)
    Atlas = resample_to_img(atlas_filename, mask, interpolation='nearest')
    masker = NiftiLabelsMasker(labels_img=Atlas, standardize=True,
                            memory='nilearn_cache', verbose=0)
    time_series = masker.fit_transform(fMRIData)
    if time_series.shape[1] != len(labels):
        print('Error: time_series.shape[1] != len(labels)')
        print(time_series[0,:])
        print("====================================================")
        print(labels)
    df = pd.DataFrame(time_series, columns=labels)
    save_path = os.path.join(fmri_save_path, file_name + '_timeseries.csv')
    df.to_csv(save_path, index=True)




def get_csv_filenames(csv_path):
    filenames = set()
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filenames.add(row['file'])
    return filenames

def get_csv_row_value(csv_path, filename, column_name):
    with open(csv_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['file'] == filename:
                return row[column_name]
    return None




@app.route('/view_cases')
def view_cases():
    records = []
    try:
        file_path = app.config.get("case_save_file")

        df_final = pd.read_csv(file_path)
        df_final.fillna('——', inplace=True)
        filtered_df = df_final[df_final['doctor'] == session.get('username')]

        camel = filtered_df.to_html(classes='table table-striped', index=False, escape=False, formatters={
            'caseId': lambda x: f'<a href="{url_for("case_detail", case_id=x)}">{x}</a>'
        })
        records.append(("Medical cases information", camel))
    except Exception as e:
        flash(f'Error reading file {"caseInfo"}: {str(e)}', 'danger')

    return render_template('view_cases.html', records=records, filtered_df=filtered_df)

@app.route('/api/query_cases', methods=['POST'])
def query_cases():
    # 获取参数处理筛选项逻辑
    filter = request.get_json()
    try:
        file_path = app.config.get("case_save_file")

        df_final = pd.read_csv(file_path)
        df_final = df_final.applymap(lambda x: None if pd.isna(x) else x)
        filtered_df = df_final[df_final['doctor'] == session.get('username')]

        # 根据filter中提供的条件进行筛选
        mask = df_final['doctor'] == session.get('username')
        if filter.get('caseId') not in [None, '']:
            case_id_mask = df_final['caseId']== int(filter['caseId'])
            mask &= case_id_mask
        if filter.get('name') not in [None, '']:
            name_mask = df_final['name'] == filter['name']
            mask &= name_mask

        filtered_df = df_final[mask]

        records = filtered_df.to_dict(orient='records')
    except Exception as e:
        return {'success': False, 'errorMsg': f'Error reading file {"caseInfo"}: {str(e)}'}, 200

    return {'success': True, 'result': {'list': records,'total': len(records)}}, 200


@app.route('/case_detail/<int:case_id>')
def case_detail(case_id):
    file_path = app.config.get("case_save_file")
    df = pd.read_csv(file_path)
    result = df.loc[df['caseId'] == case_id, 'file']
    result2 = df.loc[df['caseId'] == case_id, 'fmri.image']
    viewing_file = secure_filename(result.iloc[0])
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(result.iloc[0]))

    try:
        # 读取文件内容
        data = pd.read_csv(file_path, index_col=0)
        table_html = data.to_html(index=True)
    except Exception as e:
        flash(f'Error reading file : {str(e)}', 'danger')
        return redirect(url_for('view_cases'))

    return render_template('case_detail.html', img=result2.iloc[0],case= result.iloc[0],table_html=table_html)



@app.route('/delete_case', methods=['POST'])
def delete_case():
    # 检查用户是否登录
    username = session.get('username')
    if not username:
        flash('请先登录以继续操作.', 'warning')
        return redirect(url_for('login'))

    try:
        data = request.get_json()
        file_path = app.config.get("case_save_file")
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError("病例保存文件路径未配置或文件不存在.")

        # 读取CSV文件
        df_final = pd.read_csv(file_path)

        # 确认当前医生有权删除该病例
        if df_final[df_final['caseId'] == data['case_id']]['doctor'].values[0] != username:
            flash('您无权删除此病历.', 'danger')
            return redirect(url_for('view_cases'))

        row = df_final.loc[df_final['caseId'] == data['case_id']].iloc[0]
        fmri_image = row['fmri.image']
        file_value = row['file']
        # 删除fmri.image对应的文件（如果存在）
        if pd.notna(fmri_image):
            fmri_image_path = os.path.join(app.config.get("UPLOAD_FOLDER"), fmri_image)
            if os.path.exists(fmri_image_path):
                os.remove(fmri_image_path)
                print(f"已删除文件: {fmri_image_path}")
            else:
                print(f"文件不存在: {fmri_image_path}")

        # # # 删除file列对应的文件（如果存在）
        if pd.notna(file_value) :
            csv_file_path = os.path.join(app.config.get("UPLOAD_FOLDER"), file_value)
            if os.path.exists(csv_file_path):
                os.remove(csv_file_path)
                print(f"已删除文件: {csv_file_path}")
            else:
                print(f"文件不存在: {csv_file_path}")
        # 删除对应的行
        df_updated = df_final[df_final['caseId'] != data['case_id']]

        # 保存更新后的DataFrame回CSV文件
        df_updated.to_csv(file_path, index=False)

        return {'success': True}, 200
    except Exception as e:
        app.logger.error(f'Error deleting case {data["case_id"]}: {str(e)}')
        return {'success': False, 'errorMsg': f'删除病历时发生错误: {str(e)}'}, 200

@app.route('/api/query_csv_data', methods=['POST'])
def query_csv_data():
    data = request.get_json()
    records = []
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        df_final = pd.read_csv(file_path)
        records = df_final.dropna().to_dict('records')
    except Exception as e:
        {'success': False, 'errorMsg': f'Error reading file {"caseInfo"}: {str(e)}'}

    return {'success': True, 'result': {'list': records,'total': len(records)}}

@app.route('/api/get_column_data', methods=['POST'])
def get_column_data():
    data = request.get_json()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], data['viewing_file'])

    try:
        # 读取文件内容，将第一列设置为索引
        df_final = pd.read_csv(file_path, index_col=0)

        if data['column_header'] not in df_final.columns:
            return {'success': False, 'errorMsg': f'Column {data["column_header"]} not found'}, 500

        index = df_final.index.tolist()
        values = df_final[data['column_header']].tolist()

        return {'success': True, 'result': {'index': index, 'values': values}}
    except Exception as e:
        return {'success': False, 'errorMsg': str(e)}, 500


@app.route('/predict', methods=['POST'])
def predict_case():
    req = request.get_json()
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], req['viewing_file'])
    try:
        # 读取文件内容，将第一列设置为索引
        data = pd.read_csv(file_path)
        # 假设你有一个预测函数 predict
        diagnosis, risk = predict(data.values)

        #diagnosis, risk ='ASD',0.99

        csv_path = app.config.get("case_save_file")
        df= pd.read_csv(csv_path)

        # 找到df中‘file’列的值等于req['viewing_file']的那一行
        match = df['file'] == req['viewing_file']

        if match.any():  # 如果找到了匹配的行
            # 更新diagnosis和risk值
            df.loc[match, 'diagnosis'] = diagnosis
            df.loc[match, 'risk'] = str(risk)

            # 将更新后的数据写回到csv文件中
            df.to_csv(csv_path, index=False)

        return {'success': True, 'result': {'diagnosis': diagnosis, 'risk': str(risk)}}, 200
    except Exception as e:
        return {'success': False,'errorMsg': str(e)}, 200


@app.route('/get_nii_data/<filename>')
def get_nii_data(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404

    try:

        nii_img = nib.load(file_path)
        data = nii_img.get_fdata()

        # 发送整个数据到前端
        data_list = data.tolist()
        dimensions = list(data.shape)  # 保留原始数据的完整维度

        # small_data_subset = data[:, :, 0, :10]
        # dimensions = list(small_data_subset.shape)  # 保留原始数据的完整维度
        # small_data_subset_list=small_data_subset.tolist()

        return jsonify({'data': data_list, 'dimensions': dimensions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_brain_image', methods=['GET'])
def get_brain_image():
    # 加载示例图像
    brain_index = int(request.args.get('index'))
    print(brain_index)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], request.args.get('filename'))
    img = load_img(file_path)
    first_volume = image.index_img(img, brain_index)

    # 创建交互式视图
    html_view = plotting.view_img(first_volume)

    html_view.save_as_html("brain_image.html")

    return send_file("brain_image.html", mimetype='text/html')


# Preprocess the data to fit the model input requirements
def preprocess_data(data):
    seq_len = args.seq_len
    enc_in = args.enc_in
    current_seq_len, current_enc_in = data.shape

    if current_seq_len < seq_len:
        data = np.vstack((data, np.zeros((seq_len - current_seq_len, current_enc_in))))
    elif current_seq_len > seq_len:
        data = data[:seq_len, :]

    if current_enc_in < enc_in:
        data = np.hstack((data, np.zeros((seq_len, enc_in - current_enc_in))))
    elif current_enc_in > enc_in:
        data = data[:, :enc_in]

    inputs = torch.tensor(data)
    return inputs

# Predict function using the pre-trained model
def predict(data):
    inputs = preprocess_data(data)
    inputs = inputs.unsqueeze(0).float().to(args.device)
    with torch.no_grad():
        try:
            logit, _, _, _ = model(inputs)
            _, predicted = torch.max(logit, 1)
            proba = torch.nn.functional.softmax(logit, dim=1).detach().cpu().numpy()
            diagnosis = 'ASD' if predicted.item() == 0 else 'Normal'
            risk = float(proba[0][predicted.item()]).__round__(4)
        except Exception as e:
            print("Error in predict:", e)
            raise e
    if diagnosis == 'ASD':
        return diagnosis, risk
    else:
        asd_risk = (1 - risk).__round__(4)
        return diagnosis, asd_risk


@app.route('/change_password', methods=['GET', 'POST'])
def change_password():
    if 'username' not in session:
        flash('请先登录。')
        return redirect(url_for('login'))

    if request.method == 'POST':
        current_password = request.form.get('current_password')
        new_password = request.form.get('new_password')
        confirm_new_password = request.form.get('confirm_new_password')

        # 验证当前密码是否正确
        user = next((user for user in users if
                     user['username'] == session['username'] and user['password'] == current_password), None)
        if not user:
            flash('当前密码不正确，请重试。')
            return redirect(url_for('change_password'))

        # 验证新密码和确认新密码是否一致
        if new_password != confirm_new_password:
            flash('新密码和确认新密码不一致，请重试。')
            return redirect(url_for('change_password'))

        # 更新用户密码
        user['password'] = new_password
        flash('密码更改成功！')
        return redirect(url_for('index'))

    return render_template('change_password.html')


@app.route('/case_management')
def case_management():
    return render_template('case_management.html')


@app.route('/submit_survey', methods=['POST'])
def submit_survey():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        feedback = request.form.get('feedback')

        # 将问卷数据存储到列表中
        surveys.append({
            'name': name,
            'email': email,
            'feedback': feedback,
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })

        flash('感谢您的反馈！')
        return redirect(url_for('index'))
    return "Invalid request", 400


@app.route('/survey2')
def survey2():
    return render_template('survey2.html')


@app.route('/upload_video')
def upload_video():
    return render_template('upload_video.html')


@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    print(data)
    username = data['username']
    email = data['email']
    password = data['password']
    role = data['role']
    hospital = data['hospital'] if role == 'doctor' else None

    # 简单的验证：检查用户名是否已存在
    if any(user['username'] == username for user in users):
        return {'success': False, 'errorMsg': '用户名已存在，请选择其他用户名。'}, 200

    # 将用户数据存储到列表中
    data = {
        'username': username,
        'email': email,
        'password': password,
        'role': role,
        'hospital': hospital,
        'registration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    users.append(data)
    try:
        df = pd.DataFrame([data])
        df['registration_date'] = pd.to_datetime(df['registration_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
        # print(df)
        file_path = app.config.get("user_info_file")

        raw_df = pd.read_csv(file_path)
        if df['email'].iloc[0] in raw_df['email'].values or df['username'].iloc[0] in raw_df['username'].values :
            print(f"Email {email} already exists.")
        else:
            # 将新的DataFrame追加到读取的DataFrame中
            df.to_csv(file_path, mode='a', header=False, index=False)
        return {'success': True}, 200
    except Exception as e:
        print(str(e))


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    file_path = app.config.get("user_info_file")
    raw_df = pd.read_csv(file_path)
    if raw_df['username'].eq(data['username']).any():
        # 用户名存在，进一步判断密码是否一致
        df = raw_df[ raw_df['username'] == data['username']].to_dict('records')[0]
        if (df['username'] == data['username']) and (str(df['password']) == data['password']):
            print(f"用户 {data['username']} 存在，并且密码一致。")
            session['username'] = data['username']
            session['role'] = df['role']
            if df['role'] == 'doctor':
                return {'success': True}, 200
            else:
                return {'success': False, 'errorMsg': "Not a doctor, temporarily unable to log in"}, 200
        else:
            return {'success': False, 'errorMsg': f"User {data['username']} exists, But the passwords don't match."}, 200
    else:
        return {'success': False, 'errorMsg': 'The username or password is incorrect, please try again.'}, 200

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    session.pop('role', None)
    return {'success': True}, 200


@app.route('/doctor_dashboard')
def doctor_dashboard():
    if 'role' in session and session['role'] == 'doctor':
        file_path = app.config.get("case_save_file")
        df = pd.read_csv(file_path)
        num_data_rows = len(df)
        session['total_cases'] = num_data_rows

        now = datetime.now()
        one_week_ago = now - timedelta(days=7)
        one_week_ago2=one_week_ago.strftime('%Y-%m-%d %H:%M:%S')
        recent_cases = df[df['uploadDate'] >= one_week_ago2]
        weekly_new_cases = len(recent_cases)
        session['weekly_new_cases'] = weekly_new_cases

        return render_template('doctor_dashboard2.html')
    else:
        flash('您无权访问此页面。')
        return redirect(url_for('index'))


if __name__ == '__main__':
    # 设置上传文件夹路径
    # app.config['UPLOAD_FOLDER'] = r"E:\working_dir\py_project\zibizheng\uploads"
    # 确保上传文件夹存在
    app.run(debug=False)
