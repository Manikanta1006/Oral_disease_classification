import os
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from predict import predict_image


BASE_DIR = Path(__file__).resolve().parents[1]
CLIENT_DIST_DIR = BASE_DIR / "client" / "dist"
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}


app = Flask(__name__, static_folder=str(CLIENT_DIST_DIR), static_url_path="")
CORS(app)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def is_allowed_file(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.get("/")
def index():
    if not CLIENT_DIST_DIR.exists():
        return jsonify(
            {
                "message": "React build not found. Run `npm run build` inside client/ first."
            }
        ), 503
    return send_from_directory(CLIENT_DIST_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/api/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded."}), 400

    image = request.files["image"]
    if image.filename == "":
        return jsonify({"error": "No selected file."}), 400

    if not is_allowed_file(image.filename):
        return jsonify({"error": "Use a JPG, PNG, or WEBP image."}), 400

    original_name = secure_filename(image.filename)
    suffix = Path(original_name).suffix.lower()
    saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    image.save(saved_path)

    try:
        result = predict_image(saved_path)
    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500

    return jsonify(result)


@app.errorhandler(404)
def fallback(_error):
    if CLIENT_DIST_DIR.exists():
        return send_from_directory(CLIENT_DIST_DIR, "index.html")
    return jsonify({"error": "Not found"}), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False)
