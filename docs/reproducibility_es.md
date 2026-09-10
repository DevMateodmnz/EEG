# Reproducibilidad

Adaptación de la guía [canónica en inglés](reproducibility.md). El entorno verificado usa Python 3.14.7 y las dependencias directas fijadas en `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate.fish
python -m pip install -r requirements.txt
python check_environment.py
python -m unittest discover -s tests
```

Los datos EDF de PhysioNet se descargan a `data/`, que está excluido de Git. La adquisición Lee2019 con NEMAR sólo debe descargar y verificar fuentes; no debe ejecutar evaluación hasta que las 108 fuentes estén validadas. La guía inglesa contiene comandos de cohorte, artefactos esperados y la nota de auditoría pública.
