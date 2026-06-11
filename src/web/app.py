import os

from flask import Flask
from flask import render_template
from flask import request

from predict import predict_image


app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


def get_recommendation(label):

    recommendations = {

        "recycle":
        "Sampah dapat dipilah dan dikirim ke fasilitas daur ulang.",

        "reuse":
        "Sampah masih dapat digunakan kembali untuk keperluan lain.",

        "reduce":
        "Kurangi penggunaan barang sejenis untuk meminimalkan limbah."
    }

    return recommendations.get(
        label,
        "-"
    )


@app.route("/")
def index():

    return render_template(
        "index.html"
    )


@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    if "image" not in request.files:

        return render_template(
            "index.html",
            error="Tidak ada file yang dipilih."
        )

    file = request.files["image"]

    if file.filename == "":

        return render_template(
            "index.html",
            error="Silakan pilih gambar terlebih dahulu."
        )

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    # =========================
    # PREDIKSI
    # =========================

    result = predict_image(
        filepath
    )

    recommendation = get_recommendation(
        result["label"]
    )

    return render_template(
        "index.html",

        image_path=filepath,

        label=result["label"],

        confidence=round(
            result["confidence"],
            2
        ),

        recommendation=recommendation,

        glcm=result["glcm"],

        probabilities=result["probabilities"]
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )