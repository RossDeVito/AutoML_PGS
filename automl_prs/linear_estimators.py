"""Linear models for use with two-step AutoML-PRS."""

import time
import logging
from pprint import pprint

import numpy as np
import pandas as pd
import polars as pl
from flaml.automl.model import SKLearnEstimator
from flaml import tune
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import clone
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm


logger = logging.getLogger(__name__)


def subset_data(data, start, end, axis=0):
	if isinstance(data, (pd.DataFrame, pd.Series)):
		return data.iloc[start:end] if axis == 0 else data.iloc[:, start:end]	# type: ignore
	elif isinstance(data, pl.DataFrame):
		return data.filter(pl.col("*").slice(start, end))
	else:
		return data[start:end] if axis == 0 else data[:, start:end]
	

class PartitionedEnsembleRows(BaseEstimator, RegressorMixin):
	def __init__(self, estimator, n_partitions=3, verbose=0, **kwargs):
		self.estimator = estimator
		self.n_partitions = n_partitions
		self.verbose = verbose
		self.kwargs = kwargs
		self.models = []

	def fit(self, X, y):
		indices = np.arange(len(y))
		np.random.shuffle(indices)

		partition_size = len(y) // self.n_partitions

		for i in tqdm(
			range(self.n_partitions),
			desc='Fitting on sample partitions',
			total=self.n_partitions,
			ncols=100,
			disable=(self.verbose == 0)
		):
			start_idx = i * partition_size
			end_idx = (i + 1) * partition_size if i != self.n_partitions - 1 else len(y)
			X_subset = subset_data(X, start_idx, end_idx)
			y_subset = subset_data(y, start_idx, end_idx)

			model = clone(self.estimator)
			model.set_params(**self.kwargs)
			model.fit(X_subset, y_subset)
			self.models.append(model)

		return self

	def predict(self, X):
		predictions = np.zeros(len(X))

		for model in tqdm(
			self.models,
			desc='Predicting with all models',
			total=len(self.models),
			ncols=100,
			disable=(self.verbose == 0)
		):
			predictions += model.predict(X)

		return predictions / self.n_partitions


class ElasticNetEstimatorPRS(SKLearnEstimator):
	"""Elastic net estimator using n_partitions paritions."""

	@classmethod
	def search_space(cls, data_size, task):
		space = {
			"alpha": {
				"domain": tune.loguniform(lower=1e-10, upper=2.0),
				"init_value": 1e-4,
			},
			"l1_ratio": {
				"domain": tune.uniform(0.0, 1),
				"init_value": 1.0,
			},
			"max_iter": {
				"domain": tune.lograndint(lower=800, upper=10000),
				"init_value": 1000,
				"low_cost_init_value": 800,
			},
			"tol": {
				"domain": tune.loguniform(lower=1e-8, upper=5e-3),
				"init_value": 1e-4,
			},
			"selection": {
				"domain": tune.choice(['cyclic', 'random']),
				"init_value": 'cyclic',
			}
		}
		return space

	def __init__(
			self,
			task="regression",
			n_partitions=3,
			n_jobs=None,
			scale=True,
			**config
		):
		super().__init__(
			task,
			n_partitions=n_partitions,
			estimator=linear_model.ElasticNet(),
			**config
		)
		
		if task != "regression":
			raise ValueError(
				"ElasticNetEstimatorPRS only supports regression tasks."
			)
		
		if scale:
			self.scaler = MinMaxScaler()
			self.scaler_fit = False
		else:
			self.scaler = None
			self.scaler_fit = True
		
		self.estimator_class = PartitionedEnsembleRows

	def _preprocess(self, X):
		"""Preprocess data, including scaling."""
		if self.scaler is not None and not self.scaler_fit:
			X = self.scaler.fit_transform(X)
			self.scaler_fit = True
		elif self.scaler is not None and self.scaler_fit:
			X = self.scaler.transform(X)

		return X
	
	def _fit(
		self,
		X_train,
		y_train,
		print_params=False,
		**kwargs
	):
		"""Fit the model with early stopping."""
		if print_params:
			pprint(self.params)
		
		super()._fit(X_train, y_train, **kwargs)