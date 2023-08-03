from flask import Flask, request, jsonify, url_for
import base64
from ultralytics import YOLO
import numpy as np
import io
from PIL import Image
mymodel=YOLO("kona_yv8.pt")


app = Flask(__name__)

@app.route('/')
def home():
    return {"msg":"kona API app is ok now"}

@app.route('/process_image', methods=['POST','GET'])
def process_image():
    request_data = request.get_json()
    base64_image = request_data.get('image')

    if base64_image:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))

        result=mymodel.predict(image)
        cc_data=np.array(result[0].boxes.data)

        if len(cc_data) != 0:
            x1r=[]
            y1r=[]
            wr=[]
            hr=[]
            confr=[]
            clasr=[]
            xywh=np.array(result[0].boxes.xywh).astype("int32")
            for (_, _, w, h), (x1, y1,_,_,conf,clas) in zip(xywh,cc_data):
                x1=str(x1)
                x1r.append(x1)
                y1=str(y1)
                y1r.append(y1)
                w=str(w)
                wr.append(w)
                h=str(h)
                hr.append(h)
                conf=str(conf)
                confr.append(conf)
                clas=str(clas)
                clasr.append(clas)
                
            return jsonify({'x1':x1r,'y1':y1r,'w':wr,'h':hr,'conf':confr,'clas':clasr})
        return jsonify({'error':"no pridiction found"})
    else:
        return jsonify({'e': 'No image data found in the request'})




if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
