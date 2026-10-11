# Oxford-IIIT Pet: Full-Spectrum Computer Vision

One dataset, one project, two frameworks. Each notebook covers classification, transfer learning, explainability, segmentation, localization, detection, retrieval, autoencoders, robustness and export on the Oxford-IIIT Pet dataset (7,349 images, 37 cat/dog breeds, with pixel masks).

| File                              | Framework                  |
| --------------------------------- | -------------------------- |
| `oxford_pets_cv_pytorch.ipynb`    | PyTorch + torchvision      |
| `oxford_pets_cv_tensorflow.ipynb` | TensorFlow 2.16+ / Keras 3 |
| `requirements.txt`                | Python dependencies        |

## What each notebook covers

1. Data and EDA
2. Joint augmentation of image, mask and box
3. CNN from scratch (baseline)
4. Transfer learning with ResNet-50
5. Evaluation: top-k, confusion matrix, error analysis, calibration
6. Grad-CAM explainability
7. Semantic segmentation (U-Net)
8. Multi-task model: breed + box + mask
9. Object detection with a pretrained COCO detector
10. Second architecture (ViT-B/16 in PyTorch, ConvNeXt-Tiny in TensorFlow)
11. Embeddings, image retrieval and t-SNE
12. Denoising autoencoder and latent interpolation
13. Robustness to noise, blur, darkening, occlusion and pixelation
14. Export (ONNX, or SavedModel and TFLite), latency, and prediction on your own photo

## Setup

Use Python 3.10-3.12.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

PyTorch and TensorFlow can conflict, so one virtual environment per notebook is safest (install the shared block plus only that notebook's block from `requirements.txt`).

**Colab:** upload a notebook, choose Runtime > Change runtime type > GPU, and uncomment the `!pip install` line in the first cell.

## Running

1. Set `FAST_DEV = True` in the config cell and run everything. This uses a tiny subset and 1 epoch per stage, so it takes a few minutes and catches setup problems early.
2. Set `FAST_DEV = False` and run for real. Expect roughly 40-70 minutes on a T4 GPU.

The dataset (~800 MB) downloads automatically on the first run into `./data` (PyTorch) or `~/tensorflow_datasets` (TensorFlow).

## Outputs

Each notebook writes its trained models (`.pt` / `.keras`), an exported model (`.onnx` / SavedModel + `.tflite`), and a results table (`results_pytorch.json` / `results_tensorflow.json`) to the working directory.

To try your own photo, set `IMAGE_SOURCE` in the second-to-last section to a file path or URL.

## Notes

- A GPU is strongly recommended. CPU works but is very slow.
- The pretrained-detector section needs internet access (weights or TF Hub). If it fails, it skips itself and the rest still runs.
- The notebooks have been syntax-checked but not trained end to end. If something breaks, the most likely cause is a library version mismatch.
