import os
import datetime
from datetime import datetime, timedelta
import logging
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
import zipfile
import csv

app = Flask(__name__)
app.secret_key = 'supersecretkey'

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
        return 0
    except FileNotFoundError:
        return 0
    except pd.errors.EmptyDataError:
        return 0
    except pd.errors.ParserError:
        return 0

@app.route('/api/upload_case', methods=['GET', 'POST'])
def file_upload_destination():
    # print(request.form)
    file = request.files.get("file")
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config.get("UPLOAD_FOLDER"), filename)
        file.save(file_path)
        data = {
            'age': request.form.get("age"),
            'gender': request.form.get("gender"),
            'name': request.form.get("name"),
            'file': file.filename,
            'uploadDate': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        cases.append(data)
        # df = pd.DataFrame([data])
        # df['uploadDate'] = pd.to_datetime(df['uploadDate']).dt.strftime('%Y-%m-%d %H:%M:%S')
        # print(cases)
        # file_path = app.config.get("case_save_file")
        # df.to_csv(file_path, mode='a', header=False, index=True)
        file_path = app.config.get("case_save_file")
        # print(file_path)
        case_id = get_next_case_id(file_path)
        # print(case_id)
        data_with_case_id = {'caseId': case_id, **data}
        columns_order = ['caseId', 'age', 'gender', 'name', 'file', 'uploadDate']
        df = pd.DataFrame([data_with_case_id], columns=columns_order)
        try:
            existing_df = pd.read_csv(file_path)
            updated_df = pd.concat([existing_df, df], ignore_index=True)
            updated_df.to_csv(file_path, index=False)
        except pd.errors.EmptyDataError:
            df.to_csv(file_path,  index=False)

        # print(cases)
        # print('----------')
        # print(updated_df)
    except Exception as e:
        flash(f'Error reading file {"caseInfo"}: {str(e)}', 'danger')

    return "success"



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

    # print(cases)
    try:
        # indexed_data = [(i, case) for i, case in enumerate(cases)]
        # df = pd.DataFrame(indexed_data, columns=['caseId', 'data'])
        file_path = app.config.get("case_save_file")
        # # df = pd.read_csv(file_path)
        #
        # print('---------------------------------')
        # df_expanded = df.join(pd.json_normalize(df['data']))
        # df_final = df_expanded.drop(columns=['data'])
        # df_final.rename(columns={'uploadDate': 'date'}, inplace=True)
        # print(df_final)
        df_final = pd.read_csv(file_path)
        camel = df_final.to_html(classes='table table-striped', index=False, escape=False, formatters={
            'caseId': lambda x: f'<a href="{url_for("case_detail", case_id=x)}">{x}</a>'
        })
        records.append(("病历信息", camel))
    except Exception as e:
        flash(f'Error reading file {"caseInfo"}: {str(e)}', 'danger')

    return render_template('view_cases.html', records=records)


@app.route('/case_detail/<int:case_id>')
def case_detail(case_id):
    if case_id < 0 or case_id >= len(cases):
        flash('无效的病例ID', 'danger')
        return redirect(url_for('view_cases'))

    case = cases[case_id]
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], case['file'])

    try:
        # 读取文件内容
        data = pd.read_csv(file_path)
        table_html = data.to_html(index=False)
    except Exception as e:
        flash(f'Error reading file {case["file"]}: {str(e)}', 'danger')
        return redirect(url_for('view_cases'))

    return render_template('case_detail.html', case=case, table_html=table_html)


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


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        hospital = request.form.get('hospital') if role == 'doctor' else None

        # 简单的验证：检查用户名是否已存在
        if any(user['username'] == username for user in users):
            flash('用户名已存在，请选择其他用户名。')
            return redirect(url_for('register'))

        # 将用户数据存储到列表中
        data = {
            'username': username,
            'email': email,
            'password': password,
            'role': role,
            'hospital': hospital,
            'registration_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
            flash('注册成功！')
            return redirect(url_for('index'))
        except Exception as e:
            print(str(e))
    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        file_path = app.config.get("user_info_file")
        raw_df = pd.read_csv(file_path)
        if raw_df['username'].eq(username).any():
            # 用户名存在，进一步判断密码是否一致
            df = raw_df[ raw_df['username'] == username].to_dict('records')[0]
            if (df['username'] == username) and (str(df['password']) == password):
                print(f"用户 {username} 存在，并且密码一致。")
                session['username'] = username
                session['role'] = df['role']
                flash('登录成功！')
                if df['role'] == 'doctor':
                    return redirect(url_for('doctor_dashboard'))
                else:
                    return redirect(url_for('index'))
            else:
                print(f"用户 {username} 存在，但密码不一致。")
                flash('用户名或密码错误，请重试。')
                return redirect(url_for('login'))
        else:
            print(f"用户 {username} 不存在。")
            flash('用户名或密码错误，请重试。')
            return redirect(url_for('login'))
        # 验证用户是否存在且密码正确
        user = next((user for user in users if user['username'] == username and user['password'] == password), None)
        if user:
            session['username'] = user['username']
            session['role'] = user['role']
            flash('登录成功！')
            if user['role'] == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            flash('用户名或密码错误，请重试。')
            return redirect(url_for('login'))

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.pop('username', None)
    session.pop('role', None)
    flash('已退出登录。')
    return redirect(url_for('index'))


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
    app.run(debug=True)
