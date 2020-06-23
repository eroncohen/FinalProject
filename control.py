from flask import Flask, request
from flask_cors import CORS
from video_main import VideoMain
from Utils.model_manager import PredictionType
app = Flask(__name__)
CORS(app)

main_video = None
counter = 0


@app.route("/start", methods=['POST'])
def start():
    global main_video
    global counter
    counter += 1
    algo = request.json["algo"]
    if algo == 'CNN':
        algo = PredictionType.CNN
    elif algo == 'YE_ALGORITHM':
        algo = PredictionType.YE_ALGORITHM
    elif algo == 'MTCNN':
        algo = PredictionType.MTCNN
    else:
        algo = PredictionType.DLIB
    main_video = VideoMain(window_name=str(counter), is_doll=(request.json["doll"] == 'True'), algo=algo,
                           time=float(request.json["times"]), email=request.json["email"])
    main_video.start_detecting()
    return "start"


@app.route("/stop", methods=['POST'])
def stop():
    global main_video
    #main_video = VideoMain(is_doll=request.json["doll"], email=request.json["email"], algo=algo, time=float(request.json["times"]))
    main_video.stop_detecting()
    return "stop"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
