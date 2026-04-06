.PHONY: install data features train api test run_all

install:
	pip install -r requirements.txt

data:
	python -m src.data.generate_data

features:
	python -m src.features.build_features

train:
	python -m src.models.train

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	PYTHONPATH=. pytest tests/ -v

run_all: data features train
	@echo "Pipeline executed successfully! Run 'make api' to start the server."

monitor:
	python -m src.utils.monitor	