# asd_webproject

## 项目简介

利用Flask搭建的自闭症病历管理网站。使用了flask框架构建网站，前端使用css等渲染。

## 开发环境

- Python=3.10.13
- 创建环境：
- conda env create -f environment.yml
- conda activate mamba_env
- 如果causal_conv1d_cuda和selective_scan_cuda报错
- 参考https://blog.csdn.net/qq_45538220/article/details/143159283?utm_medium=distribute.pc_relevant.none-task-blog-2~default~baidujs_baidulandingword~default-0-143159283-blog-144476424.235^v43^pc_blog_bottom_relevance_base7&spm=1001.2101.3001.4242.1&utm_relevant_index=3中第六点修改源码

## 功能实现
- 用户管理： 注册；登录；个人信息管理
- 病历上传： 上传病例，上传fMRI的时序数据（CSV文件）；上传fMRI的时序数据（nii文件）；上传病例的基本信息（如年龄、性别、病例编号等）
- 病历管理： 病例列表，按编号、名字、上传时间等排序列出所有病例并提供筛选功能
- 医生账户管理： 包括对病历操作界面以及密码修改
## 预留功能
- 管理员管理功能

### 项目首页，如图所示 ![]()

### 注册界面，如图所示![]()

### 登录界面，如图所示![]()

### 用户（医生）界面，如图所示![]()

### 单一病历上传界面，如图所示![]()

### 批量病历上传界面，如图所示![]()

### 病历管理界面，如图所示![]()

### 修改密码界面，如图所示![]()

