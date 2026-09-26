# Comparación con la literatura

## Resultados de este trabajo

ResNet1D (variante `estandar`, 8.7 M parámetros), ECG de 12 derivaciones a 100 Hz, PTB-XL 1.0.3.
Tarea binaria: IAM (superclase diagnóstica `MI`) frente a cualquier otro ECG.
Partición oficial por paciente: folds 1–8 entrenamiento, 9 validación, 10 prueba.
Umbral elegido en validación (mayor especificidad con sensibilidad ≥ 0.85) y aplicado sin
reajustar a prueba.

| Conjunto | Sensibilidad | Especificidad | AUC-ROC |
|----------|--------------|---------------|---------|
| Validación (fold 9, n = 2 183) | 0.852 | 0.846 | 0.929 |
| Prueba (fold 10, n = 2 198) | 0.840 (IC 95 %: 0.808–0.871) | 0.843 (0.825–0.861) | 0.925 (0.913–0.937) |

Especificidad y AUC-ROC cumplen los objetivos del anteproyecto. La sensibilidad en prueba
queda 1 punto por debajo de 0.85, aunque 0.85 está dentro del intervalo de confianza.

## Ribeiro et al. (2020)

Ribeiro, A. H. et al. *Automatic diagnosis of the 12-lead ECG using a deep neural network.*
Nature Communications 11, 1760 (2020). https://doi.org/10.1038/s41467-020-15432-4

- **Datos:** más de 2 millones de ECG de la Red de Telesalud de Minas Gerais (estudio CODE), Brasil.
- **Tarea:** seis anomalías: bloqueo AV de primer grado, bloqueo de rama derecha, bloqueo de rama
  izquierda, bradicardia sinusal, fibrilación auricular y taquicardia sinusal.
- **Resultados en prueba (827 ECG anotados por cardiólogos):** F1 entre 0.870 y 1.000 y
  especificidad superior a 0.99 en las seis clases.
- **Relación con este proyecto:** es la referencia de arquitectura (red residual convolucional
  1D de extremo a extremo sobre las 12 derivaciones). **No evalúa infarto agudo al miocardio ni
  usa PTB-XL**, por lo que sus métricas no son comparables numéricamente con las nuestras.

## Strodthoff et al. (2021), benchmark de PTB-XL

Strodthoff, N. et al. *Deep Learning for ECG Analysis: Benchmarks and Insights from PTB-XL.*
IEEE Journal of Biomedical and Health Informatics 25(5), 1519–1528 (2021).
https://arxiv.org/abs/2004.13701

- **Datos y partición:** los mismos que en este trabajo (PTB-XL, folds 9 y 10 para validación y prueba).
- **Tarea "superclases diagnósticas":** clasificación multietiqueta de 5 clases (NORM, MI, STTC,
  CD, HYP). Reportan la AUC macro, es decir, el promedio de las cinco clases.
- **Resultados (AUC macro en prueba):** resnet1d_wang 0.930, xresnet1d101 0.928, inception1d 0.921.

| Trabajo | Tarea | Métrica | Valor |
|---------|-------|---------|-------|
| Este trabajo (ResNet1D) | IAM vs. resto, binaria | AUC-ROC de IAM | 0.925 |
| Strodthoff et al., resnet1d_wang | 5 superclases, multietiqueta | AUC macro | 0.930 |
| Strodthoff et al., xresnet1d101 | 5 superclases, multietiqueta | AUC macro | 0.928 |

La comparación es orientativa. Nuestra AUC es de una sola clase (IAM) y la suya es el
promedio de cinco, aunque ambas usan la misma partición de prueba. Nuestro resultado cae en
el mismo rango que las mejores arquitecturas convolucionales del benchmark.

## Limitaciones

- La sensibilidad en prueba (0.840) no alcanza el objetivo de 0.85 como estimación puntual.
- El reparto de Grad-CAM entre zonas (onda Q, ST, onda T) es cualitativo. La capa objetivo
  tiene baja resolución temporal y cada tramo resaltado abarca varios latidos, así que el
  reparto por zona tiende a ser parejo.
- Las etiquetas de PTB-XL no distinguen infarto agudo de infarto antiguo. La clase `MI`
  incluye ambos.
