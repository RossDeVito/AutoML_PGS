"""LightBGM estimators for AutoML-PRS.

Main addition is the filtering of variants by p-value and window size
using a dictionary of variant sets for each p-value and window size
before fitting/predicting.
"""

import logging
import time

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor, early_stopping
from flaml.automl.model import LGBMEstimator
from flaml.automl.data import group_counts
from flaml import tune


logger = logging.getLogger(__name__)


class LGBMEstimatorPRS(LGBMEstimator):
	"""LightGBM estimator for AutoML-PRS.
	
	Adds two new properties to estimator class:

	- var_sets_map: Dict which maps from p-value + window size thresholds
		to a subset of variant IDs.

	- covar_cols: List of column names for covariates. These are always
		included for fitting and predicting.

	Adds hyperparameter 'filter_threshold' which is a string key of
	var_sets_map. This is used to filter the variants before fitting
	and prediction.

	When running fit with this estimator, you will have to manually add the
	space of variant threshold tuples using the 'custom_hp' argument. 
	var_sets_map and covar_cols are required to be passed in using
	'fit_kwargs_by_estimator'.
	"""

	@classmethod
	def search_space(cls, data_size, **params):
		upper = max(5, min(32768, int(data_size[0])))  # upper must be larger than lower
		return {
			"n_estimators": {
				"domain": tune.lograndint(lower=4, upper=upper),
				"init_value": 4,
				"low_cost_init_value": 4,
			},
			"num_leaves": {
				"domain": tune.lograndint(lower=4, upper=upper),
				"init_value": 4,
				"low_cost_init_value": 4,
			},
			"min_child_samples": {
				"domain": tune.lograndint(lower=2, upper=2**7 + 1),
				"init_value": 20,
			},
			"learning_rate": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1.0),
				"init_value": 0.1,
			},
			"log_max_bin": {  # log transformed with base 2
				"domain": tune.lograndint(lower=3, upper=11),
				"init_value": 8,
			},
			"colsample_bytree": {
				"domain": tune.uniform(lower=0.01, upper=1.0),
				"init_value": 1.0,
			},
			"reg_alpha": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1024),
				"init_value": 1 / 1024,
			},
			"reg_lambda": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1024),
				"init_value": 1.0,
			},
		}

	def __init__(
		self,
		task,
		**kwargs
	):
		super().__init__(task, **kwargs)
		
		if self._task.is_classification():
			self.estimator_class = LGBMClassifier
		elif task == "rank":
			self.estimator_class = LGBMRanker
		else:
			self.estimator_class = LGBMRegressor

		self.var_sets_map = None
		self.covar_cols = None

	def _preprocess(self, X):
		"""Filter variants by p-value and window size.
		
		Will always include covariates in the output dataset.
		"""
		# Get variant subset
		var_subset = self.var_sets_map[					# type: ignore
			self.params['filter_threshold']
		]

		# Subset X to include only the variant subset and covariates
		return X[self.covar_cols + var_subset]
	
	def _fit(self, X_train, y_train, print_params=False, **kwargs):
		"""Fit the model.
		
		Sets var_sets_map and covar_cols so they are used for this
		fitting and future predictions.

		Args:
			print_params (bool): If True, print the parameters before fitting.
		"""
		current_time = time.time()

		if "groups" in kwargs:
			kwargs = kwargs.copy()
			groups = kwargs.pop("groups")
			if self._task == "rank":
				kwargs["group"] = group_counts(groups)

		# Update var_sets_map and covar_cols
		self.var_sets_map = kwargs.pop('var_sets_map')
		self.covar_cols = kwargs.pop('covar_cols')
		
		X_train = self._preprocess(X_train)

		model = self.estimator_class(
			**{k:v for k,v in self.params.items() if k != 'filter_threshold'}
		)
		if logger.level == logging.DEBUG or print_params:
			logger.debug(f"flaml.automl.model - {model} fit started with params {self.params}")
		model.fit(X_train, y_train, **kwargs)
		if logger.level == logging.DEBUG or print_params:
			logger.debug(f"flaml.automl.model - {model} fit finished")
		train_time = time.time() - current_time
		self._model = model
		return train_time

class LGBMEstimatorPRSv1(LGBMEstimatorPRS):
	"""LightGBM estimator with early stopping hyperparameter and
	hyperparameter space adjusted for larger dataset.
	"""

	@classmethod
	def search_space(cls, data_size, **params):
		upper = max(5, min(32768, int(data_size[0])))  # upper must be larger than lower
		return {
			"n_estimators": {
				"domain": tune.lograndint(lower=250, upper=upper),
				"init_value": 1000,
				"low_cost_init_value": 250,
			},
			"num_leaves": {
				"domain": tune.lograndint(lower=4, upper=upper),
				"init_value": 7,
				"low_cost_init_value": 4,
			},
			"min_child_samples": {
				"domain": tune.lograndint(lower=250, upper=8000),
				"init_value": 2000,
			},
			"learning_rate": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1.0),
				"init_value": 0.1,
			},
			"log_max_bin": {  # log transformed with base 2
				"domain": tune.lograndint(lower=3, upper=11),
				"init_value": 8,
			},
			"colsample_bytree": {
				"domain": tune.uniform(lower=0.25, upper=1.0),
				"init_value": 1.0,
			},
			"reg_alpha": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1024),
				"init_value": 1 / 1024,
			},
			"reg_lambda": {
				"domain": tune.loguniform(lower=1 / 1024, upper=1024),
				"init_value": 1.0,
			},
			"early_stopping_rounds": {
				"domain": tune.randint(lower=10, upper=100),
				"init_value": 10,
			},
		}
	
	def _fit(self, X_train, y_train, print_params=False, **kwargs):
		"""Fit the model.

		Creates validation split for early stopping.
		
		Sets var_sets_map and covar_cols so they are used for this
		fitting and future predictions.

		To work with early stopping, X_val and y_val must be created from
		10% of X_train and y_train.

		Args:
			X_train (pd.DataFrame or pl.DataFrame): Training data.
			y_train (pd.Series or np.ndarray): Training labels.
		"""
		current_time = time.time()

		if "groups" in kwargs:
			kwargs = kwargs.copy()
			groups = kwargs.pop("groups")
			if self._task == "rank":
				kwargs["group"] = group_counts(groups)

		# Update var_sets_map and covar_cols
		self.var_sets_map = kwargs.pop('var_sets_map')
		self.covar_cols = kwargs.pop('covar_cols')
		
		X_train = self._preprocess(X_train)

		# Create validation set by first creating a binary mask
		n_samples = X_train.shape[0]
		val_mask = np.random.choice([True, False], n_samples, p=[0.1, 0.9])

		if isinstance(X_train, pl.DataFrame):
			X_val = X_train.filter(val_mask)
			X_train = X_train.filter(~val_mask)
		else:
			X_val = X_train[val_mask]
			X_train = X_train[~val_mask]
		y_val = y_train[val_mask]
		y_train = y_train[~val_mask]

		# Create model
		non_lgbm_params = ['filter_threshold', 'early_stopping_rounds']
		model = self.estimator_class(
			**{k:v for k,v in self.params.items() if k not in non_lgbm_params}
		)

		early_stopping_callback = early_stopping(
			stopping_rounds=self.params['early_stopping_rounds']
		)

		if 'callbacks' in kwargs:
			kwargs['callbacks'].append(early_stopping_callback)
		else:
			kwargs['callbacks'] = [early_stopping_callback]

		if logger.level == logging.DEBUG or print_params:		
			logger.debug(
				f"flaml.automl.model - {model} fit started with params {self.params}"
			)
		
		model.fit(
			X_train, 
			y_train,
			eval_set=[(X_val, y_val)],
			**kwargs
		)
		
		if logger.level == logging.DEBUG or print_params:
			logger.debug(f"flaml.automl.model - {model} fit finished")

		train_time = time.time() - current_time
		self._model = model
		return train_time
	

class LGBMEstimatorPRSv2(LGBMEstimatorPRSv1):
	"""LightGBM estimator with early stopping hyperparameter and
	hyperparameter space adjusted for larger dataset.

	Param suggestions from: https://github.com/Microsoft/LightGBM/issues/695#issuecomment-315591634
	"""

	@classmethod
	def search_space(cls, data_size, **params):
		return {
			"num_leaves": {
				"domain": tune.lograndint(lower=7, upper=4095),
				"init_value": 7,
				"low_cost_init_value": 7,
			},
			"max_depth": {
				"domain": tune.randint(lower=2, upper=64),
			},
			"min_child_samples": {
				"domain": tune.lograndint(lower=250, upper=8000),
				"init_value": 2000,
			},
			"colsample_bytree": {
				"domain": tune.uniform(lower=0.4, upper=1.0),
				"init_value": 1.0,
			},
			"subsample": {
				"domain": tune.uniform(lower=0.4, upper=1.0),
				"init_value": 1.0,
			},
			"reg_alpha": {
				"domain": tune.loguniform(lower=1e-12, upper=1),
				"init_value": 1e-9,
			},
			"reg_lambda": {
				"domain": tune.loguniform(lower=1e-12, upper=1000),
				"init_value": 1e-10,
			},
			"early_stopping_rounds": {
				"domain": tune.randint(lower=10, upper=250),
				"init_value": 50,
				"low_cost_init_value": 10,
			},
		}
	
	@classmethod
	def size(cls, config):
		num_leaves = int(
			round(config.get("num_leaves"))
		)
		n_estimators = 1000
		return (num_leaves * 3 + (num_leaves - 1) * 4 + 1.0) * n_estimators * 8
	
	def __init__(
		self,
		task,
		max_n_estimators=50000,
		max_bin=192,
		**kwargs
	):
		super().__init__(task, **kwargs)
		
		if self._task.is_classification():
			self.estimator_class = LGBMClassifier
		else:
			self.estimator_class = LGBMRegressor

		# Set n_estimators and max_bin in params
		self.max_n_estimators = max_n_estimators
		self.max_bin = max_bin

		self.var_sets_map = None
		self.covar_cols = None
	