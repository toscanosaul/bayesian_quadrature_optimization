.PHONY: clean

env:
	virtualenv --setuptools $@
	$@/bin/pip install -U "setuptools>=19,<20"
	$@/bin/pip install -U "pip>=7,<8"
	$@/bin/pip install -U "pip-tools==1.6.4"

env/.requirements: requirements.txt requirements-test.txt | env
	$|/bin/pip-sync $^
	touch $@

# To run a test
# make test ARG='-k test_location'
test: env/.requirements
	env/bin/py.test tests $(ARG)

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -f .coverage
	rm -fr htmlcov/

lint: env/.requirements
	env/bin/flake8 --max-line-length=100 stratified_bayesian_optimization tests

install: clean
	python setup.py install

cover_tests: env/.requirements
	env/bin/py.test -s --tb short --cov-config .coveragerc --cov stratified_bayesian_optimization --cov-report term-missing --cov-report xml \
	--cov-report html \
	--junitxml junit.xml \
	--no-cov-on-fail \
	--cov-fail-under=100 \
	tests

coverage:
	open htmlcov/index.html

cover:
	make lint
	make cover_tests
	mkdir -p coverage
	cp coverage.xml coverage/cobertura-coverage.xml


env/.requirements-docs: docs/requirements-docs.txt | env
	$|/bin/pip-sync $^
	touch $@

docs: env/.requirements
	rm -f docs/stratified_bayesian_optimization.rst
	rm -f docs/modules.rst
	env/bin/sphinx-apidoc -o docs/ stratified_bayesian_optimization
	PATH=$(PATH):$(CURDIR)/env/bin $(MAKE) -C docs clean
	PATH=$(PATH):$(CURDIR)/env/bin $(MAKE) -C docs html

deps:
	@touch requirements.in requirements-test.in
	$(MAKE) requirements.txt requirements-test.txt

requirements.txt requirements-test.txt: %.txt: %.in | env
	$|/bin/pip-compile --no-index $^





