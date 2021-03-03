# Instanced-Level-image-retrieval-with-DeLF
## Group member:
* **Nguyen Hoang An** - *18520430@gm.uit.edu.vn*
* **Nguyen Huynh Anh** - *18520456@gm.uit.edu.vn*
* **Duong Trong Van** - *18521630@gm.uit.edu.vn*

# Table of contents
=================

<!--ts-->
   * [Introduction](#introduction)
   * [Install requirements](#install-requirements)
   * [Data](#Data)
   * [Demo](#Demo)
   * [Checklist](#Checklist)
   * [Run code](#Run-code)
   
<!--te-->

## Introduction
This project is on the field of Computer Vision, an sub-domain of Computer Science. On this project, we build our image retriveval system based on the famous paper public in ICCV 2017: [Large-Scale Image Retrieval with Attentive Deep Local Features](https://arxiv.org/abs/1612.06321). 

Evaluation our system by Average Precision and Mean Average Precision

Futhermore, we use [Flask](https://flask.palletsprojects.com/en/1.1.x/) - an web framework using for python to build our GUI

## Install requirements
```Shell
pip install -r requirements.txt
```

## Data
Our data is around 1300 landscape images of 20 class from [Google Landmarks Dataset v2](https://github.com/cvdfoundation/google-landmark)

## Demo

### Main screen
![alt text](https://github.com/NguyenHoangAn0511/Instanced-Level-image-retrieval-with-DeLF/blob/main/demo/main.jpeg)

### Example
![alt text](https://github.com/NguyenHoangAn0511/Instanced-Level-image-retrieval-with-DeLF/blob/main/demo/1.png)
![alt text](https://github.com/NguyenHoangAn0511/Instanced-Level-image-retrieval-with-DeLF/blob/main/demo/2.png)
![alt text](https://github.com/NguyenHoangAn0511/Instanced-Level-image-retrieval-with-DeLF/blob/main/demo/3.png)

## Run code:
Extract Database (if you update or create new database):
```
python extract.py
```
To run main code:
```
python server.py
```
###### Output:
```
* Serving Flask app "server" (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
INFO:werkzeug: * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
```
Go to http://127.0.0.1:5000/
## Checklist
- [x] Upload Image
- [x] Process uploaded image
- [x] Save Image to Database
- [x] Interactive Web app
- [x] Retrieve Image
- [x] Average Precision and Mean Average Precision
- [x] Optimize processing time
