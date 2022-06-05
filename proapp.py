from flask import Flask,jsonify,request
from proclassifier import get_prediction
proapp = Flask(__name__)
@proapp.route("/predict-letter",methods = ["POST"])
def predictdata():
    image = request.files.get("letter")
    prediction = get_prediction(image)
    return jsonify({
        "prediction":prediction
    }),477
if __name__ == "__main__":
    proapp.run(debug = True)

    