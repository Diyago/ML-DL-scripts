from flask import (
    Flask,
    escape,
    flash,
    request,
    jsonify,
    redirect,
    url_for,
    render_template,
    send_file,
)
import numpy as np
import os
from sklearn.externals import joblib

knn = joblib.load("knn.pkl")
app = Flask(__name__)


@app.route("/")
def hello():
    print("Started hello")
    name = request.args.get("name", "my friend")
    return f"<h2>Hello, {escape(name)}!</h2>"


@app.route("/square/<username>")
def squar_val(username):
    username = float(username) * float(username)
    return str(username)


def average(lst):
    return sum(lst) / len(lst)


@app.route("/avg/<nums>")
def avg(nums):
    nums = nums.split(",")
    nums = [float(num) for num in nums]
    return str(average(nums))


@app.route("/iris/<params>")
def fit_predict_iris(params):
    params = params.split(",")
    params = np.array([float(num) for num in params]).reshape(1, -1)

    print("Input params:", params)
    predict = knn.predict(params)
    img_path = '<br><img src="/static/setosa.jpg" alt="Setoca iris flower" width="500" height="600">'
    return str(predict) + img_path


@app.route("/show_image")
def show_image():
    print("image loaded")
    return '<img src="/static/setosa.jpg" alt="Setoca iris flower" width="500" height="600">'


@app.route("/iris_post", methods=["POST"])
def add_message():
    try:
        content = request.get_json()
        params = content["flower"].split(",")
        params = np.array([float(num) for num in params]).reshape(1, -1)

        print("Input params:", params)
        predict = {"class": str(knn.predict(params)[0])}
    except:
        return redirect(url_for("bad_request"))
    return jsonify(predict)


from flask import abort


@app.route("/badrequest400")
def bad_request():
    abort(400)


from flask_wtf import FlaskForm
from wtforms import StringField, FileField
from werkzeug.utils import secure_filename
from wtforms.validators import DataRequired
import pandas as pd

UPLOAD_FOLDER = ""
ALLOWED_EXTENSIONS = set(["txt", "pdf", "png", "jpg", "jpeg", "gif"])
app.config.update(
    dict(SECRET_KEY="powerful secretkey", WTF_CSRF_SECRET_KEY="a csrf secret key")
)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


class MyForm(FlaskForm):
    name = StringField("name", validators=[DataRequired()])
    file = FileField()


@app.route("/submit", methods=("GET", "POST"))
def submit():
    form = MyForm()
    if form.validate_on_submit():

        f = form.file.data
        filename = form.name.data + ".csv"
        # f.save(os.path.join(
        #     filename
        # ))

        df = pd.read_csv(f, header=None)
        print(df.head())

        predict = knn.predict(df)

        result = pd.DataFrame(predict)
        result.to_csv(filename, index=False)

        return send_file(
            filename,
            mimetype="text/csv",
            attachment_filename=filename,
            as_attachment=True,
        )

    return render_template("submit.html", form=form)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            return "file uploaded"

    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """
