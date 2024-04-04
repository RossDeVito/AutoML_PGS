"""Linear models for use with two-step AutoML-PRS."""

import time
import logging
from pprint import pprint

from flaml.automl.model import SKLearnEstimator
from flaml import tune
from sklearn import linear_model
from sklearn import pipeline
from sklearn.preprocessing import MinMaxScaler


logger = logging.getLogger(__name__)


class ElasticNetEstimatorPRS(SKLearnEstimator):
	"""Elastic net estimator. """

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
			n_jobs=None,
			scale=True,
			**config
		):
		super().__init__(task, **config)
		
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
		
		self.estimator_class = linear_model.ElasticNet

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