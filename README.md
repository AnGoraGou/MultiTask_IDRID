
```markdown
# 🧠 Multitask Learning on IDRiD Dataset

This repository presents a multitask deep learning framework applied to the **IDRiD (Indian Diabetic Retinopathy Image Dataset)**. It simultaneously performs:

- ✅ **Disease Grading (Classification)**: Predict the grade of diabetic retinopathy.
- ✅ **Lesion Segmentation (Binary or Multi-channel)**: Segment optic disc or multiple lesion types like microaneurysms, hemorrhages, and exudates.

---

## 🏗️ Model Architecture

This implementation uses **soft parameter sharing** with:

- 🔗 A shared ResNet-based encoder (e.g., `resnet18`, `resnet34`).
- 🎯 Two task-specific heads:
  - 🧬 **Classification Head**: Adaptive pooling + fully connected layers.
  - 🧼 **Segmentation Head**: U-Net decoder for binary or multi-class lesion maps.
- ⚖️ A **dynamic routing network** to weight task losses during training.

---

## 📁 Folder Structure


Multitask-IDRiD/
├── data/
│   ├── classification\_dataset.py
│   └── segmentation\_dataset.py
├── models/
│   ├── backbone.py
│   ├── expert.py
│   ├── gating.py
│   └── multitask\_model.py
├── transforms/
│   └── joint\_transform.py
├── utils/
│   ├── metrics.py
│   └── loss.py
├── main.py
├── requirements.txt
└── README.md



---

## ⚙️ Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/Multitask-IDRiD.git
cd Multitask-IDRiD
````

2. **(Recommended)** Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

---

## 💾 Data Preparation

Download the dataset from: [IDRiD Grand Challenge](https://idrid.grand-challenge.org/)

Ensure the following structure is maintained:

```
IDRiD/
├── DiseaseGrading/
│   ├── OriginalImages/
│   └── Groundtruths/
└── Segmentation/
    ├── OriginalImages/
    └── AllSegmentationGroundtruths/
        ├── Microaneurysms/
        ├── Haemorrhages/
        ├── Hard_Exudates/
        ├── Soft_Exudates/
        └── Optic_Disc/
```

Update paths in `main.py` or pass via `argparse` if applicable.

---

## 🚀 Usage

Train the multitask model with:

```bash
python main.py
```

This will:

* Train both classification and segmentation branches
* Log loss, accuracy, and Dice score
* Save best model and plots

---

## 📊 Outputs

After training, you will get:

* 🧠 `best_mtlr_model_TIMESTAMP.pth` – Best model checkpoint
* 🧠 `final_mtlr_model_TIMESTAMP.pth` – Final model weights
* 📈 `mtlr_metrics_plot_TIMESTAMP.png` – Loss & metric curves

---

## 🧪 Evaluation & Visualization

The framework provides:

* 🟦 Classification Accuracy
* 🔴 Dice Score for segmentation
* 🖼️ Overlay visualizations of predictions vs. ground truth (optional extension)

---

## 🔗 Dependencies

Key libraries:

* [`torch`](https://pytorch.org/)
* [`torchvision`](https://pytorch.org/vision/)
* [`segmentation_models_pytorch`](https://github.com/qubvel/segmentation_models.pytorch)
* `matplotlib`, `pandas`, `scikit-learn`

Install all via:

```bash
pip install -r requirements.txt
```

---

## ✍️ Author

**Gouranga Bala**
📧 [gouranga.bala23@gmail.com](mailto:gouranga.bala23@gmail.com)
📘 https://www.linkedin.com/in/gouranga-bala-5871b8191/

---

## 📄 License

This project is licensed under the **MIT License**.
See [`LICENSE`](LICENSE) for full terms.

```
