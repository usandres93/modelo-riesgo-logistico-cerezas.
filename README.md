# MCRL: Modelo de Calificación de Riesgo Logístico para la Exportación de Cerezas

Este repositorio contiene la implementación técnica del **MCRL**, desarrollado como núcleo del Proyecto de Título para optar al grado de **Ingeniero Civil Industrial**. El software integra simulación estocástica, cinética de deterioro de alimentos y analítica predictiva para optimizar la cadena de suministro de cerezas chilenas hacia mercados internacionales.

---

## 📋 Descripción General
El MCRL evalúa el impacto de la variabilidad logística (tiempos de tránsito y quiebres térmicos) sobre la condición final de la fruta. A diferencia de un análisis estático, este sistema permite predecir el **Nivel de Riesgo Logístico** de cada lote antes de su llegada a destino, permitiendo una toma de decisiones proactiva.

## ⚙️ Arquitectura del Proyecto
El sistema se divide en tres capas fundamentales:

1.  **Capa de Simulación (`generador_datos.py`):**
    - Implementación de la **Ecuación de Arrhenius** para modelar el deterioro de firmeza y pedicelo.
    - Simulación de perfiles térmicos y eventos de deshidratación basados en tecnología de empaque.
    - Generación de un dataset sintético de 500 lotes con variables agronómicas y logísticas.

2.  **Capa de Inteligencia de Negocios (`analisis_datos.py`):**
    - Análisis Exploratorio de Datos (EDA) automatizado.
    - Visualización de correlaciones críticas y semaforización de riesgos (Verde, Amarillo, Naranja, Rojo).
    - Identificación de "Daño Oculto" por pérdida de peso.

3.  **Capa Predictiva (`modelo_ia.py`):**
    - Comparación de modelos de ensamble: **Random Forest** vs. **XGBoost**.
    - Clasificación multiclase de riesgo logístico con balanceo de pesos.
    - Evaluación de métricas de precisión (F1-Score, AUC-ROC) y análisis de falsos negativos.

## 🚀 Instalación y Uso
Para ejecutar el modelo en un entorno local:

1.  Clonar el repositorio:
    ```bash
    git clone [https://github.com/usandres93/modelo-riesgo-logistico-cerezas.git](https://github.com/usandres93/modelo-riesgo-logistico-cerezas.git)
    ```
2.  Instalar dependencias:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost openpyxl
    ```
3.  Ejecutar el flujo de trabajo:
    - Primero, generar los datos: `python generador_datos.py`
    - Segundo, realizar el análisis: `python analisis_datos.py`
    - Tercero, entrenar la IA: `python modelo_ia.py`

## 📊 Resultados Esperados
El sistema genera:
- Un dataset maestro en Excel (`Dataset_MCRL_Completo.xlsx`).
- Reportes visuales en formato PNG con la distribución de riesgos y matrices de confusión.
- Un modelo entrenado capaz de clasificar nuevos contenedores según su probabilidad de rechazo.

---

## 🎓 Autoría y Créditos
- **Autor:** Andres Uribe
- **Carrera:** Ingeniería Civil Industrial
- **Propósito:** Proyecto de Título 2026

## ⚖️ Licencia
Este proyecto se distribuye bajo la licencia **GNU General Public License v3.0**. Consulta el archivo `LICENSE` para más detalles sobre el uso y atribución.

## 📖 Cómo citar este trabajo
Si utilizas este modelo o los datos sintéticos para fines académicos, por favor utiliza la siguiente referencia:

> **Uribe, A. (2026).** *Modelo de Calificación de Riesgo Logístico (MCRL) para la Exportación de Cerezas* [Código de computación]. GitHub. https://github.com/usandres93/modelo-riesgo-logistico-cerezas
