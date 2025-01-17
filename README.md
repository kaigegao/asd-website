# asd_webproject

## 项目简介

利用Flask搭建的自闭症病历管理网站。使用了flask框架构建网站，前端使用css等渲染。

## 功能实现
- 用户管理： 注册；登录；个人信息管理
- 病历上传： 上传病例，上传fMRI的时序数据（CSV文件）；上传病例的基本信息（如年龄、性别、病例编号等）
- 病历管理： 病例列表，按编号、名字、上传时间等排序列出所有病例并提供筛选功能
- 医生账户管理： 包括对病历操作界面以及密码修改
## 预留功能
- 诊断功能
- 管理员管理功能
- 首页功能入口保留

### 项目首页，如图所示 ![](pic/mainwebsite.png)

### 注册界面，如图所示![](pic/regis.png)

### 登录界面，如图所示![](pic/login.png)

### 用户（医生）界面，如图所示![](pic/dashboard.png)

### 单一病历上传界面，如图所示![](pic/upload.png)

### 批量病历上传界面，如图所示![](pic/bulk-upload.png)

### 病历管理界面，如图所示![](pic/view-cases.png)

### 修改密码界面，如图所示![](pic/passchange.png)

## 开发环境

- Python == 3.10.4
- 其他依赖项请参阅requirements.txt