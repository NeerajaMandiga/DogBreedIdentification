import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
from utils import load_trained_model, predict_image

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'replace-me-with-a-secure-key'

# Load model at startup (if available)
MODEL, CLASS_MAP = load_trained_model()
if MODEL is None:
    print('Warning: Model not found. Run `python train_model.py` to create dogbreed.h5')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict')
def predict_page():
    return render_template('predict.html')


@app.route('/result', methods=['POST'])
def result():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # make unique
        import time
        filename = f"{int(time.time())}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # If model not loaded, show helpful message
        if MODEL is None or CLASS_MAP is None:
            flash('Model not found. Please run training first: python train_model.py')
            return redirect(url_for('predict_page'))

        try:
            label, conf = predict_image(filepath, MODEL, CLASS_MAP)
            confidence_pct = round(conf * 100, 2)
            return render_template('output.html', filename=filename, label=label, confidence=confidence_pct)
        except Exception as e:
            flash('Prediction failed: ' + str(e))
            return redirect(url_for('predict_page'))
    else:
        flash('Invalid file type. Allowed types: png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)
