.PHONY: serve train export install health evaluate

VENV := ./venv/bin
PORT ?= 8002

# Levantar el microservicio (equivalente a php artisan serve / ng s)
serve:
	$(VENV)/uvicorn app.main:app --reload --port $(PORT)

# Pipeline ML completo: exportar BD → Prophet → Ridge
train:
	$(VENV)/python -m scripts.train_all

# Solo exportar CSVs desde PostgreSQL
export:
	$(VENV)/python -m scripts.export_training_data

# Verificar que el servidor responde y cuántos modelos cargó
health:
	@curl -s http://127.0.0.1:$(PORT)/health | python3 -m json.tool

# Evaluar calidad de modelos (backtest + métricas out-of-sample)
evaluate:
	$(VENV)/python -m scripts.evaluate_models

# Instalar dependencias en el venv
install:
	$(VENV)/pip install -r requirements.txt
