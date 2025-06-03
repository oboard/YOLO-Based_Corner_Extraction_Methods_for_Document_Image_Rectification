# YOLO-Based Corner Extraction Methods for Document Image Rectification

This repository contains the source code and dataset related to our paper:

**"YOLO-Based Corner Extraction Methods for Document Image Rectification"**  
📄 Presented at ICEICT 2025

## 📦 Contents

- `models/`: Pre-trained model weights
  - `pose-gscl-best.pt`: GSCL pose estimation model
  - `pose-origin-best.pt`: Original pose estimation model
  - `segment-best.pt`: Segmentation model
- `pose/`: Pose estimation module
  - `preprocess.py`: Data preprocessing scripts
  - `train.py`: Training script
  - `ui.py`: User interface
  - `weight/`: Model weights
- `segmentation/`: Document segmentation module
  - `preprocess.py`: Data preprocessing scripts
  - `train.py`: Training script
  - `ui.py`: User interface
  - `weights/`: Model weights
- `test/`: Evaluation and comparison scripts
  - `experiment1.py` to `experiment6.py`: Different experiment configurations
  - Comparison result visualizations
- `requirements.txt`: Project dependencies
- `README.md`: This file

---

## 📂 Dataset

Due to the large size of the dataset (~4GB), it is hosted externally.

🔗 **Download Dataset:**

- **Baidu Pan (百度网盘)**: [datasets-YOLO-Based_Corner_Extraction_Methods_for_Document_Image_Rectification.zip](https://pan.baidu.com/s/1Unr6m97wjuNBZnSIftOviQ?pwd=417s)  
  Extraction code: `417s`

- **Google Drive**: [datasets-YOLO-Based_Corner_Extraction_Methods_for_Document_Image_Rectification.zip](https://drive.google.com/file/d/1aSrWut3HGIgrTgvLb0EDoFmpwH2avpdI/view?usp=sharing)

**Contents:**

- `images/`: Document images
- `pose-labels/`: Corner labels
- `segment-labels/`: Polygon labels

---

## 🚀 Getting Started

### Requirements

- Python ≥ 3.8
- PyTorch ≥ 1.10
- OpenCV
- YOLO11

Install dependencies:

```bash
pip install -r requirements.txt
