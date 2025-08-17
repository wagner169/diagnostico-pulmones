# Sistema de Diagnóstico Médico por Imágenes (Pulmones)
# Proyecto UEES - Inteligencia Artificial
# Integrantes:
Wagner Moreno Alvarado
Jean Paul Amay Cruz
Elizabeth Amada Martínez Reyes
Iván Vera Torres


## ¡Accede a la vanguardia de la IA en diagnóstico médico!
## Resumen

Ponemos a tu disposición un conjunto de modelos de Inteligencia Artificial de alto rendimiento, diseñados y optimizados para analizar radiografías de tórax. Nuestros modelos entrenados no son solo software; son activos digitales listos para ser integrados en tus aplicaciones, sistemas o plataformas existentes, brindando la capacidad de diagnosticar con una precisión superior.

## Nuestros Modelos Entrenados:

Hemos desarrollado y optimizado modelos de aprendizaje profundo que sobresalen en la clasificación de imágenes médicas.

Precisión en 4 Clases: Nuestros modelos están entrenados para clasificar imágenes de radiografías en categorías críticas:

COVID-19

Neumonía Viral

Opacidad Pulmonar

Normal

Arquitectura de Alto Rendimiento: Utilizan una arquitectura de red neuronal convolucional (CNN) tipo VGG, reconocida por su robustez y capacidad para extraer características complejas de las imágenes.

Formatos de Fácil Integración: Entregamos los modelos en formatos estándar de la industria (.h5 y .keras), lo que garantiza una integración sencilla y sin fricciones en una amplia gama de entornos de desarrollo, desde APIs hasta aplicaciones de escritorio o móviles.

#El Valor de Integrar Nuestros Modelos

Acelera tu Innovación: Ahorra meses de investigación, recolección de datos y entrenamiento de modelos. Nuestros modelos están listos para usar, permitiéndote llevar al mercado soluciones de diagnóstico más rápido.

Resultados Confiables: Reduce la carga de diagnóstico en el personal médico con modelos que ofrecen un alto grado de precisión.

Escalabilidad Garantizada: Nuestros modelos han sido optimizados para un despliegue eficiente, asegurando que tu sistema pueda manejar un alto volumen de predicciones de manera ágil.

## Cómo Usar e Interpretar los Resultados

Uso: Simplemente pasa una radiografía de tórax a la API o sistema donde integres el modelo.

Resultados: El modelo retornará una predicción con un nivel de confianza (por ejemplo, "Neumonía Viral" con un 95% de confianza). Puedes utilizar este resultado para apoyar la toma de decisiones clínicas.




## Nuestro Equipo

Este proyecto es el resultado de la dedicación y el conocimiento de un equipo de expertos en Inteligencia Artificial y salud de la UEES:

Wagner Moreno Alvarado

Jean Paul Amay Cruz

Elizabeth Amada Martínez Reyes

Iván Vera Torres


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
