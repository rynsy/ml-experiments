run:
	python3 main.py

check:
	python3 skl.py

init:
	pip install -r requirements.txt
	unzip hw1_data.zip
	mkdir graphs

test:
	py.test tests

clean:
	rm *.pyc
	rm */*.pyc
	rm -rf __pycache__

.PHONY: init test
