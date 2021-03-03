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
   * [Run code](#Run-code)
   
<!--te-->

## Install requirements
```Shell
pip install -r requirements.txt
```

## Demo

### Main screen
![alt text]()

### Example
![alt text]()
![alt text]()
![alt text]()

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
