import os
import time
from flask import Flask,render_template,request
from werkzeug.utils import secure_filename
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
app = Flask(__name__)
fld="uploads/"
app.config['UPLOAD_FOLDER']=fld

@app.route("/")
def home():
    return render_template("index.html",data="")

@app.route("/upload",methods=["GET","POST"])
def upload():
    if(request.method=="POST"):
        stt=time.time()
        fl=request.files["file"]
        fl_name=secure_filename(fl.filename)
        path=os.path.join(app.config['UPLOAD_FOLDER'],fl_name)
        fl.save(path)
        model=load_model("model.h5")
        model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
        #img_width, img_height = 224,224
        img=image.load_img(path,target_size=(224,224))
        x=image.img_to_array(img)
        x=np.expand_dims(x,axis=0)
        class_names=["ABE","ART","BAS","BLA","EBO","EOS","FGC","HAC","KSC","LYI","LYT","MMZ","MON","MYB","NGB","NGS"]
        classes=np.argmax(model.predict(x),axis=-1)
        for i in classes:
            name=class_names[i]
        os.remove(path)
        d={
            "ABE":"Abnormal Eosinophil",
            "ART":"Artefact",
            "BAS":"Basophil",
            "BLA":"Blast",
            "EBO":"Erythroblast",
            "EOS":"Eosinophil",
            "FGC":"Faggott cell",
            "HAC":"Hairy Cell",
            "KSC":"Smudge Cell",
            "LYI":"Immature Lymphocyte",
            "LYT":"Lymphocyte",
            "MMZ":"Metamyelocyte",
            "MON":"Monocyte",
            "MYB":"Myelocyte",
            "NGB":"Band neutrophil",
            "NGS":"Segmented neutrophil",
        }
        e=time.time()
        t=e-stt;
        k={
            "Prediction_Class":d[name],
            "Time_Required":round(t,2)
        }
        return render_template("index.html",data=k)
        

if __name__ == "__main__":
    app.run()
