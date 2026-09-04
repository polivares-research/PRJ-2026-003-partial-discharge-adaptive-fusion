# Hypotheses

## H1 — Fusión adaptativa

La media ponderada por confianza calibrada de un experto temporal y uno
espectral obtiene un MCC mayor que cada experto por separado y que la media
de probabilidades con pesos fijos.

## H0 — Sin beneficio de adaptación

La fusión adaptativa no mejora de forma relevante al mejor baseline, definido
para este PoC como una diferencia de MCC inferior a `0.005` en `Te0`.

## Racional

La representación temporal conserva la localización y forma de los pulsos,
mientras que la representación espectral resume su contenido frecuencial. La
entropía de la probabilidad calibrada se usa como proxy de fiabilidad por
muestra, permitiendo reducir el peso del experto menos seguro.
