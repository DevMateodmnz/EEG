# Métodos

Adaptación del documento [canónico en inglés](methods.md). El flujo completado inspecciona el EEG antes de transformarlo, identifica el tramo válido de la grabación, aplica la referencia promedio y filtros FIR congelados, crea épocas alineadas a eventos y registra calidad de ensayo. C3/C4 son sensores centrales convencionales para imaginación de mano; las afirmaciones siguen siendo a nivel de sensor.

Los decodificadores espectrales y CSP+LDA usan *runs* o participantes retenidos según el estudio. CSP, escalado y LDA se ajustan sólo en los datos de entrenamiento y los archivos de auditoría registran identidades de entrenamiento/prueba. La validación Lee2019 no usa EMG ni calibración de sesión objetivo y aún no se evalúa.
