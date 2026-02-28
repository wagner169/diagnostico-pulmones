# Sistema de Diagnóstico Médico por Imágenes (Pulmones)
# Proyecto UEES - Inteligencia Artificial
# Integrantes:
Wagner Moreno Alvarado
Elizabeth Amada Martínez Reyes

Overview

This project provides a set of high-performance Artificial Intelligence models designed and optimized to analyze chest X-ray images. These trained models are not just experimental prototypes — they are production-ready digital assets that can be integrated into applications, systems, or platforms to support clinical decision-making with high accuracy.

Trained Models

We developed and optimized deep learning models specialized in medical image classification.

4-Class Classification

The models classify chest X-ray images into the following categories:

COVID-19

Viral Pneumonia

Lung Opacity

Normal

High-Performance Architecture

The system uses Convolutional Neural Network (CNN) architectures inspired by VGG-style designs, known for robust feature extraction in medical imaging tasks.

Trained architectures include:

EfficientNet-B0

DenseNet121

ResNet50

Models are delivered in standard .pth format for seamless integration across environments such as APIs, backend systems, desktop, or mobile applications.

Value Proposition
Faster Innovation

Avoid months of data collection and training. Models are ready for deployment.

Reliable Results

High-accuracy classification reduces diagnostic workload.

Scalability

Optimized for efficient deployment and high-volume inference.

How Predictions Work
Usage

Send a chest X-ray image to the integrated API or system.

Output

The model returns a predicted class with confidence score (e.g., "Viral Pneumonia – 95% confidence").

The file compare_models.py evaluates the three trained models and selects the most confident prediction using the following logic:

# Select the model with the highest prediction confidence
best_model = max(results, key=lambda m: max(results[m].values()))
best_class = max(results[best_model], key=results[best_model].get)
return best_model, best_class, results[best_model]

This ensures the final diagnosis is selected from the model with the strongest confidence score while preserving visibility of all model predictions.

Interpretability
Preprocessing Demonstration (Masking)

The system applies a preprocessing step that removes ribs and non-lung noise, focusing exclusively on lung tissue before classification.

Input Image
<img width="467" height="467" alt="image" src="https://github.com/user-attachments/assets/95313148-2639-4bd3-9eb9-e26f71cde9c3" />
Lung Mask
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/041ec458-5879-48ec-b57d-080761aa3226" />
Processed Lung Image
<img width="467" height="467" alt="image" src="https://github.com/user-attachments/assets/345dcb9c-9e65-4a6b-93ee-43b1aea13c66" />

This preprocessing ensures decisions are based on relevant pulmonary structures only.

Model Performance Metrics

Each X-ray is analyzed by all trained models. Performance comparison allows the system to justify the final selected result.

Overall Performance Comparison
Model	Accuracy	Weighted F1 Score
EfficientNet-B0	95.8%	96.1%
DenseNet121	95.1%	95.5%
ResNet50	94.6%	94.8%
<img width="572" height="247" alt="image" src="https://github.com/user-attachments/assets/2a5e33d1-fb8f-4410-9630-7f40c284723e" />

The system selects the most reliable prediction based on confidence level.

Example:

ResNet50: Pneumonia (92%)

DenseNet121: Pneumonia (93%)

EfficientNet-B0: Pneumonia (98%)

Final result presented: Pneumonia – 98% confidence

Technical Description

4-class image classification (COVID, Normal, Viral Pneumonia, Lung Opacity)

CNN-based architectures

Evaluation with classification metrics and confusion matrix

Inference on new images

Model export in .pth format

API-ready deployment

Project Structure
/models

compare_models.py

process_masks.py

train_densenet121.py

train_efficientnet_b0.py

train_resnet50.py

prueba.py

predict_with_lime.py

/api

app.py

Other files:

README.md

requirements.txt

.gitignore

Streamlit Dashboard (Analytical Visualization Only)

Dashboard (for analysis demonstration):

https://1ddd735f0741.ngrok-free.app/

Google Colab Development:

https://colab.research.google.com/drive/1VYUeRJ_vqoeMRraKuUhkjQmA1iCIE-Jn#scrollTo=tOFEeddbwjIP

Production Deployment

The trained models are exposed through a deployed API and consumed by a live web application.

Live Application:

https://uees-lung-x-ray.vercel.app/

API Endpoint:

https://ensemble-api-qzpf.onrender.com

Web App Repository:

https://github.com/ElizaMarti/UEES-LungX-Ray

API Repository:

https://github.com/wagner169/ensemble-api

Trained Models (Download Links)

EfficientNet:
https://drive.google.com/file/d/1KpLhlrcAgSaIUIIABNpu6LzFGGG-a2TS/view?usp=drive_link

DenseNet:
https://drive.google.com/file/d/1FuliZMAQdaiOWYlsahiYJqzc8fy_2oUQ/view?usp=drive_link

ResNet:
https://drive.google.com/file/d/1yhdFMGyaw-gW4yU5pPl6CFJRZVvS44np/view?usp=drive_link

Installation & Usage
Install dependencies
pip install -r requirements.txt
Train models
python train_densenet121.py
python train_efficientnet_b0.py
python train_resnet50.py
Evaluate models
python compare_models.py
Run prediction
python prueba.py
python predict_with_lime.py
Requirements

Python 3.10+

TensorFlow 2.15+

NumPy 1.26+

Matplotlib 3.8+

Seaborn 0.12+

Scikit-learn 1.3+

OpenCV 4.9+

Pandas 2.1+

Pillow 10.0+

tqdm 4.66+

lime

scikit-image

