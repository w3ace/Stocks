.PHONY: setup lint test run-app

setup:
	 pip install -r requirements.txt

lint:
	 flake8 stocks cli tests

test:
	 pytest -q

run-app:
	 streamlit run app/Home.py
