# 🖼️ Image Classifier Project

[![Python 3.7+](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.19](https://img.shields.io/badge/TensorFlow-2.19-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11-green.svg)](https://opencv.org/)

A convolutional neural network (CNN) classifier trained on CIFAR-10 dataset to recognize 10 object categories:

✈️ Plane | 🚗 Car | 🐦 Bird | 🐱 Cat | 🦌 Deer  
🐶 Dog | 🐸 Frog | 🐴 Horse | 🚢 Ship | 🚚 Truck

## 🛠️ Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/image-classifier.git
cd image-classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Training the model
```python
# Uncomment training code in main.py then run:
python main.py
```

### Making predictions
1. Replace 'path/to/image.jpg' in main.py with your image path
2. Run:
```python
python main.py
```

## 🧠 Model Architecture
```mermaid
graph LR
A[Input 32x32x3] --> B[Conv2D 32]
B --> C[MaxPool 2x2]
C --> D[Conv2D 64]
D --> E[MaxPool 2x2]
E --> F[Conv2D 64]
F --> G[Flatten]
G --> H[Dense 64]
H --> I[Output 10]
```

## 📊 Dataset
Uses CIFAR-10 dataset:
- 50,000 training images
- 10,000 test images
- 10 classes
- 32x32 color images

## 📦 Dependencies
- Python 3.7+
- TensorFlow 2.19
- Keras 3.9.2
- OpenCV 4.11
- NumPy 2.1.3
- Matplotlib 3.10.1

Full list in [requirements.txt](requirements.txt)

## 🔧 TODO
- [ ] Create prediction script with CLI arguments
- [ ] Implement real-time webcam classification