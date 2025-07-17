# Implementación de un Modelo de Difusión (DDPM) en TensorFlow

Este proyecto es una implementación desde cero de un **Modelo de Difusión Probabilística Denoising (DDPM)** utilizando TensorFlow y Keras. El objetivo es generar imágenes de dígitos manuscritos (MNIST) aprendiendo a revertir un proceso de adición de ruido.

![Imagen de ejemplo generada](reports/mnist_generated.png)

## Introducción

Los modelos generativos han revolucionado el aprendizaje profundo, permitiéndonos crear datos nuevos y realistas. En los últimos años ha surgido una nueva clase de modelos con una potencia y estabilidad impresionantes: los **Modelos de Difusión Probabilística Denoising (DDPMs)**.

La idea central es:
1.  **Proceso Directo:** Se toma una imagen real y se le añade ruido gradualmente hasta que se convierte en ruido puro.
2.  **Proceso Inverso:** Se entrena una red neuronal para revertir este proceso, partiendo de ruido y eliminándolo iterativamente hasta generar una imagen nueva.

## Trabajo Relacionado

Este trabajo se basa principalmente en los conceptos presentados en:

* **"Denoising Diffusion Probabilistic Models" (Ho, Jain, and Abbeel, 2020):** El trabajo seminal que popularizó los DDPMs.
* **"U-Net: Convolutional Networks for Biomedical Image Segmentation" (Ronneberger, et al., 2015):** La arquitectura U-Net es la columna vertebral de la mayoría de los modelos de difusión.

## Métodos

### Proceso de Difusión Directo (Forward Process)
Este es un proceso fijo que añade ruido a una imagen $x_0$ en $T$ pasos. Podemos muestrear el estado $x_t$ en cualquier paso $t$ con la fórmula:
$$ x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon $$

### Proceso Inverso y Arquitectura del Modelo (U-Net)
El objetivo es aprender a predecir el ruido ($\epsilon$) que fue añadido a una imagen. Para esto, usamos una arquitectura **U-Net** que recibe la imagen ruidosa $x_t$ y el paso de tiempo $t$ (codificado con embeddings posicionales) y predice el ruido $\epsilon_{\theta}(x_t, t)$.

## Estructura del Repositorio
```
ddpm-tensorflow/
├── models/               # Guarda los pesos del modelo entrenado
├── reports/              # Guarda las imágenes generadas
├── src/                  # Código fuente modular
│   ├── data_processing.py  # Carga y preprocesa los datos
│   ├── diffusion.py      # Lógica del proceso de difusión
│   └── model.py          # Arquitectura de la U-Net
├── .gitignore            # Archivos a ignorar por Git
├── generate.py           # Script para generar nuevas imágenes
├── README.md             # Este archivo
├── requirements.txt      # Dependencias del proyecto
└── train.py              # Script principal para entrenar el modelo
```

## Uso

1.  **Clona el repositorio:**
    ```bash
    git clone <URL-DE-TU-REPO>
    cd ddpm-tensorflow
    ```

2.  **Crea un entorno virtual e instala las dependencias:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Entrena el modelo:**
    ```bash
    python train.py
    ```
    Los pesos del modelo se guardarán en la carpeta `models/`.

4.  **Genera nuevas imágenes:**
    ```bash
    python generate.py
    ```
    La imagen con la cuadrícula de dígitos se guardará en `reports/mnist_generated.png`.
