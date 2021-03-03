# Instanced-Level-image-retrieval-with-DeLF
## Group member:
* **Nguyen Hoang An** - *18520430@gm.uit.edu.vn*
* **Nguyen Huynh Anh** - *18520456@gm.uit.edu.vn*
* **Duong Trong Van** - *18521630@gm.uit.edu.vn*

# Table of contents
=================

<!--ts-->
   * [Install requirements](#install-requirements)
   * [Demo](#Demo)
   * [Checklist](#Checklist)
   * [Results](#results)
   * [Run streamlit](#Run-streamlit-(Change-to-Makeup-directory-)-)
   
<!--te-->

## Install requirements
```Shell
pip install -r requirements.txt
```

## Demo

### Main screen
![alt text](https://github.com/NguyenHoangAn0511/Make-up-application-DeepLearning-Flask/blob/main/Makeup/example/main.jpeg)

### Image Enhance + Filter + Dye hair example
![alt text](https://github.com/NguyenHoangAn0511/Make-up-application-DeepLearning-Flask/blob/main/Makeup/example/POSTERIZE%20%2B%20OBLUE.jpeg)
![alt text](https://github.com/NguyenHoangAn0511/Make-up-application-DeepLearning-Flask/blob/main/Makeup/example/PURRPLE-hair.jpeg)
![alt text](https://github.com/NguyenHoangAn0511/Make-up-application-DeepLearning-Flask/blob/main/Makeup/example/MAKEUP-adjust.jpeg)

## Run streamlit (Change to Makeup directory):
```
streamlit run main.py
```
###### Output:
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.5:8501
```
Link to streamlit: https://www.streamlit.io/
## Checklist
- [x] Upload Image
- [x] Process uploaded image
- [x] Save Image to Database
- [x] Interactive Web app
- [x] Face detection (Facial Landmark, Face parsing)
- [x] Image filter (Vintage, ...)
- [x] Dye hair & better eyebrow color
