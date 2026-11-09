DATA ?= data/case_data.xlsx

.PHONY: install run test clean
install:
	pip install -e ".[xlsb]"

run:
	python scripts/run_pipeline.py --data $(DATA) --out results

test:
	python -m unittest discover -s tests -v

clean:
	rm -rf results/figures/*.png results/*.xlsx results/*.csv results/*.json data/*.pkl
