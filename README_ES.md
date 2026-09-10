# Análisis y decodificación reproducible de EEG de imaginación motora

[English (canonical)](README.md) · [Citación](CITATION.cff) · [Resumen](docs/project_overview_es.md)

Proyecto independiente de investigación y software sobre EEG público de imaginación motora, procesamiento de señales, neurociencia computacional y métodos BCI. El trabajo completado usa PhysioNet EEGMMIDB para replicar efectos espectrales a nivel de sensores y evaluar decodificadores clásicos con *runs* retenidos y entre participantes. No presenta afirmaciones clínicas, causales ni de BCI en línea.

La validación externa Lee2019/OpenBMI entre sesiones está **EN PROGRESO**. El protocolo está congelado, pero los datos no están completos y no existen resultados externos.

| Hallazgo completado | Diseño | Resultado |
| --- | --- | --- |
| Reducción task-versus-rest de 12–13 Hz en C4 durante imaginación de puños | Participantes 2–109 | 87/105 estimaciones negativas |
| Misma tendencia en C3 | Participantes 2–109 | 91/105 estimaciones negativas |
| CSP+LDA específico por participante | 86 participantes; *runs* retenidos | mediana BA 0.658 |
| CSP estrictamente zero-shot | Mismo grupo; personas retenidas | mediana BA 0.570 |

Los valores provienen del resumen congelado [`docs/assets/final_results_summary.csv`](docs/assets/final_results_summary.csv). Son asociaciones a nivel de sensor y resultados predictivos *offline*, no evidencia de neuronas individuales ni de control BCI práctico.

Los artefactos públicos congelados usan procedencia portátil relativa a la raíz de datos; los resultados completados de PhysioNet no cambiaron durante esta migración de procedencia. El estudio externo continúa en progreso.

## Reproducción mínima

```bash
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -r requirements.txt
python -m unittest discover -s tests
python scripts/inspect_raw_data.py
```

La guía científica canónica está en inglés: [reproducibilidad](docs/reproducibility.md). Los datos crudos se descargan a `data/` y no se incluyen en Git. La procedencia pública portátil y el límite de historia se describen en inglés en [public provenance](docs/public_provenance.md). Consulte [resultados](docs/results_es.md), [validación externa](docs/external_validation_es.md), [limitaciones](docs/limitations.md) y [colaboración](docs/collaboration_es.md).
