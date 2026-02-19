# Dog Breed Prediction (Flask + VGG19 transfer learning)


# # Download the Trained model 
https://drive.google.com/file/d/1v7YuTvqguRU1I6qDMVZmvpH3NpgKQXR8/view?usp=sharing

Place it inside /model folder before running.

This project provides a Flask web application that predicts dog breeds from uploaded images using a VGG19 transfer learning model (TensorFlow / Keras).

Quick start

1. Create & activate virtual environment (recommended):

```powershell
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install tensorflow==2.12.0 keras==2.12.0 flask numpy pandas pillow
```

3. Prepare dataset (folder structure):

```
dataset/
  train/
    breed1/
    breed2/
    ...
  test/
    ...
```

4. Train model (creates `dogbreed.h5` and `class_names.json`):

```powershell
python train_model.py --dataset dataset --epochs 6
```

5. Run the app:

```powershell
python app.py
```

Open `http://127.0.0.1:5000/` in your browser.

Notes
- If `dogbreed.h5` is missing the app will still run but will prompt you to train the model first.
- The training script saves the best model (based on validation accuracy) and writes a class mapping to `class_names.json`.
#