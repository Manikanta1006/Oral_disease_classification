# Oral Disease Classification

TensorFlow image classification project for:

- gingivitis
- healthy
- mouth_ulcer
- tooth_decay

The app trains a MobileNetV2 transfer learning model, serves predictions through a Flask API, and uses a React client for the full UI.

## Project Structure

```text
dataset/
  train/
  valid/
  test/
backend/
  model/
  uploads/
  train.py
  predict.py
  app.py
  requirements.txt
client/
  src/
  package.json
  vite.config.js
```

Each `dataset` split must contain the same class folders:

```text
healthy/
mouth_ulcer/
tooth_decay/
gingivitis/
```

The code uses this fixed class order:

```text
gingivitis
healthy
mouth_ulcer
tooth_decay
```

Keep those exact folder names. After training, the same order is saved in `backend/model/class_names.json` and used during prediction, so predictions do not shift from one class name to another.

If your dataset is in COCO format with `_annotations.coco.json`, first convert or copy the images into these class folders. The current training script is an image classification trainer, not an object-detection/COCO trainer.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Install React client dependencies:

```bash
cd client
npm install
cd ..
```

On Linux or Lightning AI:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
cd client
npm install
cd ..
```

## Train

Put your images inside the class folders under `dataset/train` and `dataset/valid`, then run:

```bash
python backend/check_dataset.py
```

```bash
python backend/train.py --epochs 15
```

This creates:

```text
backend/model/oral_disease_model.h5
backend/model/class_names.json
```

## Test Prediction From Terminal

```bash
python backend/predict.py path/to/image.jpg
```

## Run In Development

Start Flask API:

```bash
python backend/app.py
```

In a second terminal, start React:

```bash
cd client
npm run dev
```

Open:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to Flask on port `8080`.

## Run Production Build

Build the React app:

```bash
cd client
npm run build
cd ..
```

Start Flask:

```bash
python backend/app.py
```

Open:

```text
http://localhost:8080
```

## Deploy On Lightning AI

1. Upload or clone this project into a Lightning AI Studio.
2. Upload your dataset into the `dataset/` folder.
3. Install dependencies:

```bash
pip install -r requirements.txt
cd client
npm install
cd ..
```

4. Train the model:

```bash
python backend/train.py --epochs 15
```

5. Build the React UI:

```bash
cd client
npm run build
cd ..
```

6. Start the app:

```bash
python backend/app.py
```

Lightning usually exposes the running `PORT` automatically. The app also defaults to port `8080` when `PORT` is not set.

## Important

This project is for AI/ML learning and demonstration. It is not a medical diagnostic system.