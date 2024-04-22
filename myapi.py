from flask import Flask, jsonify
from flask import request
from werkzeug.utils import secure_filename


from machinelearning import preprocess_image, read_image, predict_image

ALLOWED_EXTENSIONS = {"png", "jpg", "bmp", "jpeg"}

app = Flask(__name__)


@app.route("/", methods=["GET"])
def hello_world():
    data = {"Message" : "Service online!"}
    return jsonify(data)
    # return {"Message" : "Service online!"}

@app.route("/upload", methods=["POST"])
def upload_file():
    f = request.files["file"] 
    if f.filename == '':
        return "No file selected"
    if not allowed_file(f.filename):
        return "File not supported"
    f.filename = secure_filename(f.filename) # To ensure filename is safe

    image = read_image(f.read())
    image = preprocess_image(image)
    img_class = predict_image(image)
    return img_class

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 

# ========================================================================
# ngrok http --domain=ideal-modest-gazelle.ngrok-free.app <port_number>   
# <port_number> is the port from localhost or the port you are letting it listen to
# ========================================================================
