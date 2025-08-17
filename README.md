# Sistema de Diagnóstico Médico por Imágenes (Pulmones)

Este proyecto utiliza modelos de deep learning para diagnosticar automáticamente enfermedades pulmonares a partir de radiografías de tórax.

## Descripcion Tecnica del proyecto

- Clasificación de imágenes en 4 clases: **COVID, Normal, Neumonía Viral, Opacidad Pulmonar**
- Arquitectura CNN tipo VGG
- Evaluación con métricas de clasificación y matriz de confusión
- Predicción de nuevas imágenes
- Guardado del modelo en formatos `.keras` y `.h5`
- Preparado para despliegue en una API

## Estructura del proyecto

Carpeta Pulmones contiene: 

Carpeta modelos: 

- modelo_diagnostico.h5
- modelo_pulmones_mejor.keras

Carpeta scripts:

- cargar_datos.py
- entrenar_modelo.py
- evaluar_modelo.py
- predictor.py

Carpeta api

- app.py

README.md
requirements.txt
.gitignore

## Instrucciones de uso

1.- Instalar dependencias: 

 ```bash
   pip install -r requirements.txt

2.- Entrena el modelo:

python scripts/entrenar_modelo.py

3.- Evalúa el modelo: 

python scripts/evaluar_modelo.py

4.- Predice nuevas imágenes:

python scripts/predictor.py


##Requisitos

- Python 3.10+

- TensorFlow

- NumPy

- Matplotlib

- Seaborn

- Scikit-learn

- OpenCV

 
=======
# diagnostico-pulmones
>>>>>>> 9ea0b24f81d248786e722c226c1e0c80077a8068
