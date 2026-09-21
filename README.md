# Undertone AI Makeup Recommender

This repo contains a Gradio app that analyzes a face image or webcam frame, predicts undertone, mood, and skin depth, then recommends matching makeup products from the local makeup dataset.

## Google Drive Setup

The current app scripts expect this exact Google Drive path:

```text
/content/drive/MyDrive/undertone-ai-project/undertone-ai
```

In Google Drive, place or clone the repo so the folder structure looks like this:

```text
MyDrive/
  undertone-ai-project/
    undertone-ai/
      AI_Undertone.ipynb
      scripts/
      data/
```

This path matters because `scripts/makeup_recommender.py` and `scripts/gradio_ver.py` load data and images from that location.

## Run In Colab

1. Open `AI_Undertone.ipynb` in Google Colab.
2. Run the Drive mount cell.
3. Run the project path setup cell.
4. Run the dependency install cell.
5. Run the "Run The App" cell.
6. Open the public Gradio link printed by the notebook.

The app supports webcam and uploaded images.

## Optional Notebook Sections

The notebook also includes optional sections for:

- checking whether makeup JSON and product image paths load correctly
- rebuilding `data/undertone_faces` from `data/undertone`
- retraining the undertone model and saving `data/undertone_faces/undertone_resnet18_best.pth`

You do not need to run those optional cells just to start the recommender app.

## Main Files

- `AI_Undertone.ipynb`: Colab notebook for setup, app launch, and optional training
- `scripts/gradio_ver.py`: Gradio app entry point
- `scripts/makeup_recommender.py`: makeup product loading and recommendation logic
- `scripts/undertone_detector.py`: undertone detection
- `scripts/mood_detector.py`: mood detection
- `scripts/shade_detector.py`: skin depth detection
- `data/makeup`: product images and JSON metadata
- `data/undertone`: raw undertone image dataset
- `data/undertone_faces`: processed undertone face patches and trained model
