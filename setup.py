# setup.py file

from setuptools import setup, find_packages


setup(
	name='automl_prs',
	version='0.0.1',
	author='Ross DeVito',
    author_email='rdevito@ucsd.edu',
	packages=find_packages(),
	install_requires=[
		'flaml',
		'lightgbm',
		'matplotlib',
		'numpy',
		'optuna',
		'pandas',
		'polars',
		'scikit-learn',
		'tqdm'
	],
)
