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

## Precisión en 4 Clases: 
Nuestros modelos están entrenados para clasificar imágenes de radiografías en categorías críticas:

COVID-19

Neumonía Viral

Opacidad Pulmonar

Normal

## Arquitectura de Alto Rendimiento: 
Utilizan una arquitectura de red neuronal convolucional (CNN) tipo VGG, reconocida por su robustez y capacidad para extraer características complejas de las imágenes.

## Formatos de Fácil Integración: 
Entregamos los modelos en formatos estándar de la industria (.pth), lo que garantiza una integración sencilla y sin fricciones en una amplia gama de entornos de desarrollo, desde APIs hasta aplicaciones de escritorio o móviles.

## El Valor de Integrar Nuestros Modelos

## Acelera tu Innovación: 
Ahorra meses de investigación, recolección de datos y entrenamiento de modelos. Nuestros modelos están listos para usar, permitiéndote llevar al mercado soluciones de diagnóstico más rápido.

## Resultados Confiables: 
Reduce la carga de diagnóstico en el personal médico con modelos que ofrecen un alto grado de precisión.

## Escalabilidad Garantizada: 
Nuestros modelos han sido optimizados para un despliegue eficiente, asegurando que tu sistema pueda manejar un alto volumen de predicciones de manera ágil.

## Cómo Usar e Interpretar los Resultados

## Uso: 
Simplemente pasa una radiografía de tórax a la API o sistema donde integres el modelo.

## Resultados: 
El modelo retornará una predicción con un nivel de confianza (por ejemplo, "Neumonía Viral" con un 95% de confianza). Puedes utilizar este resultado para apoyar la toma de decisiones clínicas. Usando el compare_models.py utiliza los 3 modelos y entrega la top class explicado en esta linea de codigo:

    # elegir el modelo más seguro en su predicción
    best_model = max(results, key=lambda m: max(results[m].values()))
    best_class = max(results[best_model], key=results[best_model].get)
    return best_model, best_class, results[best_model]

de esta manera escoje el mejor model para la prediccion y tambien los resultados de los demas pronosticos.

## Interpretabilidad
## Demostración del Pre-procesamiento (la Máscara)
## Imagen de entrada

<img width="467" height="467" alt="image" src="https://github.com/user-attachments/assets/95313148-2639-4bd3-9eb9-e26f71cde9c3" />

Nuestros modelos no ven la imagen completa. Su primer paso es aplicar un filtro inteligente que elimina las costillas y otros ruidos para enfocarse únicamente en el tejido pulmonar. Esto asegura que nuestros modelos tomen sus decisiones basándose en la evidencia más relevante para un diagnóstico.

## Mascara
 
<img width="400" height="400" alt="image" src="https://github.com/user-attachments/assets/041ec458-5879-48ec-b57d-080761aa3226" />

## Imagen con filtro aplicado, lista para procesar exactamente los pulmones
 
<img width="467" height="467" alt="image" src="https://github.com/user-attachments/assets/345dcb9c-9e65-4a6b-93ee-43b1aea13c66" />

## Demostración de las Métricas 
Cada radiografía es analizada simultáneamente por cada modelo entrenado (deep learning) de Inteligencia Artificial. Al comparar sus métricas de rendimiento, podemos justificar por qué el sistema confía más en uno que en otro para dar la respuesta final.
## Cuadro comparativo de rendimiento general
<img width="572" height="247" alt="image" src="https://github.com/user-attachments/assets/2a5e33d1-fb8f-4410-9630-7f40c284723e" />

## Este cuadro muestra los resultados de cada modelo, demostrando su fiabilidad probada en nuestras evaluaciones.
Modelo	Precisión (Accuracy)	Puntaje F1 (Weighted)
EfficientNet-B0	95.8%	96.1%
DenseNet121	95.1%	95.5%
ResNet50	94.6%	94.8%

 <img width="975" height="605" alt="image" src="https://github.com/user-attachments/assets/4a12d75f-24a5-4280-bf50-05f46e792859" />

(Nota: El bloque representa el porcentaje del puntaje F1. Cuanto más alta la barra, mejor el rendimiento del modelo.)
Estas métricas demuestran que la solución no solo da una respuesta, sino que la da después de que los modelos se ponen en marcha, seleccionan la opinión de mayor calidad para ofrecer el diagnóstico más fiable.


## Demostración de la Lógica de la API (la Selección del Resultado)

<img width="975" height="975" alt="image" src="https://github.com/user-attachments/assets/14a08441-0b01-420b-8ae8-fbcb289e577b" />

 
Una vez que cada modelo ha dado su diagnóstico, toma la decisión final con una lógica simple pero poderosa: seleccionar el diagnóstico que tiene el mayor nivel de confianza. Por ejemplo, si los tres modelos analizan la misma radiografía y sus resultados son:
o	ResNet50: Neumonía (92% de confianza)
o	DenseNet121: Neumonía (93% de confianza)
o	EfficientNet-B0: Neumonía (98% de confianza)
La API presenta al usuario el resultado de Neumonía con un 98% de confianza, porque fue el diagnóstico más sólido. Esta es nuestra forma de garantizar que el resultado final sea siempre el más confiable y seguro posible para el profesional."


## Descripcion Tecnica del proyecto

- Clasificación de imágenes en 4 clases: **COVID, Normal, Neumonía Viral, Opacidad Pulmonar**
- Arquitectura CNN tipo VGG
- Evaluación con métricas de clasificación y matriz de confusión
- Predicción de nuevas imágenes
- Guardado del modelo en formatos `.pth`
- Preparado para despliegue en una API

## Estructura del proyecto

Carpeta Pulmones contiene: 

Carpeta modelos: 

- compare_models
- process_masks.py
- train_densenet121.py
- train_efficientnet_b0.py
- train_resnet50.py
- prueba.py
- predict_with_lime.py

Carpeta api

- app.py

README.md
requirements.txt
.gitignore

## Dashboard: Streamlit para visualización de resultados

link del dashboard solo para efectos analiticos , no es la API de resultado

https://1ddd735f0741.ngrok-free.app/

<img width="1402" height="1142" alt="image" src="https://github.com/user-attachments/assets/4fde1f63-c7a5-4bf1-819e-77dad144552f" />


<img width="1238" height="841" alt="image" src="https://github.com/user-attachments/assets/83f2004f-4a14-4778-8549-a6215ed0ebc9" />

De esta manera se demuestra como internamente trabajan los modelos, dependiendo de la certeza que tenga mas porcentaje que haya alcanzado en cada ejecusion

link del desarrollo en google collab con streamlit

https://colab.research.google.com/drive/1VYUeRJ_vqoeMRraKuUhkjQmA1iCIE-Jn#scrollTo=tOFEeddbwjIP

# Pagina Web en produccion
Usamos la API que generamos con los modelos entrenados , de esta manera la consumimos en el sitio web para que sea transparente al usuario, el sitio esta en linea listo para consumir.

link de la pagina web en produccion:

https://uees-lung-x-ray.vercel.app/

<img width="886" height="464" alt="image" src="https://github.com/user-attachments/assets/80df44fe-2356-4bfc-969b-a74c8cb32acd" />

<img width="886" height="502" alt="image" src="https://github.com/user-attachments/assets/8d191452-f641-4153-bfc8-b1598f133ca8" />

<img width="886" height="500" alt="image" src="https://github.com/user-attachments/assets/42ced6f6-6246-4b66-a065-7cbe6dbb0ec2" />

## GitHub del código de programación de la App web expuesta:

https://github.com/ElizaMarti/UEES-LungX-Ray

## Uso de API con modelos entrenados:
<img width="884" height="507" alt="image" src="https://github.com/user-attachments/assets/e3516735-f782-43b0-9ed2-d48847188502" />


## Nuestro Equipo

Este proyecto es el resultado de la dedicación y el conocimiento de un equipo de estudiantes de la Mestria en Ciencia de Datos e Inteligencia de Negocios - UEES:

Wagner Moreno Alvarado

Jean Paul Amay Cruz

Elizabeth Amada Martínez Reyes

Iván Vera Torres

## Enlaces de interes
Modelos entrenados

Efficientnet

https://drive.google.com/file/d/1KpLhlrcAgSaIUIIABNpu6LzFGGG-a2TS/view?usp=drive_link 

Densenet

https://drive.google.com/file/d/1FuliZMAQdaiOWYlsahiYJqzc8fy_2oUQ/view?usp=drive_link 

resnet

https://drive.google.com/file/d/1yhdFMGyaw-gW4yU5pPl6CFJRZVvS44np/view?usp=drive_link



Api para usar el servicio y los modelos entrenados

https://ensemble-api-qzpf.onrender.com



Aplicación funcional con las métricas para poder usar el api

https://uees-lung-x-ray.vercel.app/



GitHub del modelo entrenado para render

https://github.com/wagner169/ensemble-api



GitHub de la aplicación 

https://github.com/ElizaMarti/UEES-LungX-Ray/tree/main

Manual de uso de APP Web
Dentro de la carpeta Documentos


## Instrucciones de uso

1.- Instalar dependencias: 

 ```bash
   pip install -r requirements.txt

2.- Entrena el modelo:

python scripts/

- train_densenet121.py
- train_efficientnet_b0.py
- train_resnet50.py

3.- Evalúa el modelo: 

python scripts/compare_models

4.- Predice de problemas pulmonares:

python scripts/

- prueba.py
- predict_with_lime.py

## Requisitos

- Python 3.10+

- TensorFlow 2.15+

- NumPy 1.26+

- Matplotlib 3.8+

- Seaborn 0.12+

- Scikit-learn 1.3+

- OpenCV 4.9+

- Pandas 2.1+

- Pillow 10.0+

- tqdm 4.66+

- lime scikit-image
=======
# diagnostico-pulmones
>>>>>>> 9ea0b24f81d248786e722c226c1e0c80077a8068


