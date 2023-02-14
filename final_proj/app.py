from flask import Flask, request, render_template, send_file, flash
import cv2
from ai import processAI
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'dlgkals2633'

image = ""


@app.route('/', methods=["GET"])
def homepage():
    return render_template("index.html", image_file="picture.jpg", result_file="picture.jpg")


@app.route('/upload', methods=["POST", "GET"])
def upload_file():
    global image
    if request.method == "POST":
        try:
            data = request.files['file']
            image = request.files['file']
            data.save('./static/'+secure_filename(data.filename))
            return render_template("index.html", image_file=data.filename, result_file="picture.jpg")
        except:
            flash("image를 넣고 UPLOAD 하세요!")
            return render_template("index.html", image_file="picture.jpg", result_file="picture.jpg")
    else:
        return render_template("index.html", image_file="picture.jpg", result_file="picture.jpg")


@app.route('/fileDown', methods=['GET', 'POST'])
def download_file():
    if request.method == 'POST':
        try:
            path = "./static/"
            return send_file(path + 'result.jpg',
                             as_attachment=True)
        except:
            flash("Download할 파일이 없습니다!")
            return render_template("index.html", image_file="picture.jpg")
    return render_template("index.html", image_file=image.filename)


@app.route('/process', methods=["POST", "GET"])
def processing():
    if request.method == "POST":
        # try:
        img = cv2.imread('static/'+image.filename)
        ai = processAI(img)
        cv2.imwrite('static/result.jpg', ai)
        return render_template("index.html", result_file="result.jpg", image_file=image.filename)
        # except:
        #     flash("Restomer를 실패하였습니다.")
        #     return render_template("index.html", image_file="picture.jpg")
    else:
        return render_template("index.html", image_file="picture.jpg")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
