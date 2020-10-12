import io
import json

from oasis import app
from flask import send_file, render_template
from flask import Flask, jsonify, request, render_template
from oasis.model.microbe_net import ResNet18


model = ResNet18()


@app.route("/", endpoint="index")
def index():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        (class_id, class_name), confidence = model.get_prediction(
            image_bytes=img_bytes)

        confidence = sorted(confidence.items(), key=(
            lambda x: x[1]), reverse=True)

        # print(confidence)
        result = ""
        for key, val in confidence:
            result += str(key)+"&emsp;&emsp;&emsp;" + \
                str(val) + "%" + "<br />"
        # print(result)
        return jsonify({'class_id': class_id, 'class_name': class_name, "confidence": result})


@app.route("/finger", methods=["GET"])
def finger_get():
    return send_file("static/img/finger.jpg", mimetype='image/jpg')
