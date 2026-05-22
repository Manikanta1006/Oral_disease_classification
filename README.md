# Oral Region And Disease Classification

TensorFlow image classification project for predicting two things from one oral image:

- detected oral region, for example `inner_cheek`
- possible disease, for example `oral_ulcer`

The backend trains a MobileNetV2 transfer-learning CNN with two output heads:

- `region`: oral organ/region prediction
- `disease`: oral disease prediction

The API and React UI show results like:

```text
Detected Region: Inner cheek
Possible Disease: Oral ulcer
Prediction Value: 84.2%
```

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
client/
  src/
```

## Dataset Format

For region and disease prediction, every image needs both labels. Put images in nested folders:

```text
dataset/
  train/
    inner_cheek/
      oral_ulcer/
        image1.jpg
      healthy/
        image2.jpg
    gums/
      gingivitis/
        image3.jpg
  valid/
    inner_cheek/
      oral_ulcer/
        image4.jpg
  test/
    inner_cheek/
      oral_ulcer/
        image5.jpg
```

Current region labels are defined in `backend/config.py`:

```text
inner_cheek
gums
tongue
teeth
lips
palate
```

Current disease labels are:

```text
gingivitis
healthy
oral_ulcer
tooth_decay
```

If your folders still use `mouth_ulcer`, the trainer accepts it as an alias for `oral_ulcer`.

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

## Train The CNN

1. Add images to `dataset/train/<region>/<disease>/`.
2. Add validation images to `dataset/valid/<region>/<disease>/`.
3. Make sure every region and every disease has at least one train and valid image.

Check the dataset:

```bash
python backend/check_dataset.py
```

Train:

```bash
python backend/train.py --epochs 15
```

This creates:

```text
backend/model/oral_region_disease_model.keras
backend/model/labels.json
```

For better accuracy, use more images per class and train longer:

```bash
python backend/train.py --epochs 30 --batch-size 16 --learning-rate 0.0001
```

## Test Prediction From Terminal

```bash
python backend/predict.py path/to/image.jpg
```

Example output:

```text
Detected Region: Inner cheek
Possible Disease: Oral ulcer
Prediction Value: 84.2%
```

For full probabilities:

```bash
python backend/predict.py path/to/image.jpg --json
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

## Important

This project is for AI/ML learning and demonstration. It is not a medical diagnostic system.
