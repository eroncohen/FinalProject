from flask import Flask, request
from flask_cors import CORS
from video_main import VideoMain
from model_predictor import PredictionType
app = Flask(__name__)
CORS(app)


@app.route("/start", methods=['POST'])
def start():
    eron = request
    print(request.json["times"])
    algo = request.json["algo"]
    if algo == 'CNN':
        algo = PredictionType.CNN
    elif algo == 'YE_ALGORITHM':
        algo = PredictionType.YE_ALGORITHM
    elif algo == 'MTCNN':
        algo = PredictionType.MTCNN
    else:
        algo = PredictionType.DLIB
    main_video = VideoMain(is_doll=request.json["doll"], email=request.json["email"], algo=algo, time=request.json["times"])
    main_video.start_detecting()
    return "Welcome to Python Flask!"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
