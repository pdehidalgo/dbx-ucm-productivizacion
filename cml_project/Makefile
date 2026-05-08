.PHONY: install data validate smoke train report clean run

UV ?= uv

install:
	$(UV) sync --locked

data:
	$(UV) run python src/get_data.py

validate:
	$(UV) run python src/validate_data.py

smoke:
	$(UV) run python src/smoke_train.py

train:
	$(UV) run python src/train.py

report:
	$(UV) run python src/build_report.py

run: data validate smoke train report

clean:
	rm -rf data/processed artifacts report.md
