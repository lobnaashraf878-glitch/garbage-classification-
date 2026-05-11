![Garbage Image](https://github.com/user-attachments/assets/e68bddbc-ca5c-43e1-a50b-7b9d7e591603)

# Garbage Classification using EfficientNet

This repository contains the implementation of a garbage classification system inspired by the research paper: [Deep Learning-based Waste Detection in Natural and Urban Environments](https://doi.org/10.1016/j.wasman.2021.12.001). The system achieves efficient and accurate classification of waste into predefined categories using a deep learning approach.

## Overview

Garbage pollution is a significant environmental challenge, and efficient waste classification is crucial for effective recycling and waste management. This project implements a model based on **EfficientNet** architecture, which achieves **83% classification accuracy**, making it a practical tool for real-world applications such as smart waste bins and urban waste monitoring.

### Features

- **Deep Learning Backbone**: Utilizes EfficientNet for image classification.
- **Dataset Integration**: Prepares and trains on the Kaggle [Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data).
- **Accuracy**: Achieves high accuracy with robust performance in diverse environmental conditions.
- **Customizable**: Easy-to-use training and testing pipelines.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- TensorFlow / PyTorch
- NumPy
- OpenCV (for image preprocessing)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/garbage-classification
   cd garbage-classification
   ```

2. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:

   Download the **Garbage Classification Dataset** from Kaggle: [https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data). Extract the dataset and place it in the `data/` directory.

4. Preprocess the dataset:

   ```bash
   python preprocess_data.py --dataset-path data/Garbage\ Classification
   ```

### Training the Model

Train the EfficientNet model using the provided dataset:

```bash
python train.py --dataset-path data/processed --epochs 50 --batch-size 32
```

### Testing the Model

Evaluate the trained model on the test set:

```bash
python test.py --model-path models/efficientnet_best.pth --dataset-path data/processed
```

## Implementation Details

### Model Architecture

The project uses EfficientNet-B2, chosen for its balance of computational efficiency and accuracy. The architecture includes:

1. Pre-trained EfficientNet backbone for feature extraction.
2. Fully connected layers for waste category classification.

### Waste Categories

The Kaggle dataset includes the following waste categories:
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

These categories are used directly in the model.

### Results

- **Classification Accuracy**: 83% on the test set.
- **Dataset Size**: The Kaggle dataset contains over 2,500 labeled images of waste.

## Contributing

We welcome contributions! Please follow the steps below to contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

## Acknowledgments

- **Dataset**: This project uses the [Garbage Classification dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification/data) from Kaggle.
- **Research Paper**: Inspired by [Deep Learning-based Waste Detection in Natural and Urban Environments](https://doi.org/10.1016/j.wasman.2021.12.001).
