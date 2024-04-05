"""LightBGM estimators for AutoML-PRS."""

import logging
import time
from pprint import pprint

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from flaml.automl.model import LGBMEstimator
from flaml.automl.data import group_counts
from flaml import tune


logger = logging.getLogger(__name__)


class LGBMEstimatorPRS(LGBMEstimator):
	"""LightGBM estimator for AutoML-PRS.
	
	Uses early stopping.
	
	Param suggestions from: 
	https://github.com/Microsoft/LightGBM/issues/695#issuecomment-315591634
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
		max_bin=32,
		**kwargs
	):
		super().__init__(task, **kwargs)
		
		if self._task.is_classification():
			self.estimator_class = LGBMClassifier
		else:
			self.estimator_class = LGBMRegressor

		# Set n_estimators and max_bin in params
		self.params['n_estimators'] = max_n_estimators
		self.params['max_bin'] = max_bin

	def _preprocess(self, X):
		"""Return X."""
		return X
	
	def _fit(
		self,
		X_train,
		y_train,
		val_frac=0.1,
		print_params=False,
		**kwargs
	):
		"""Fit the model with early stopping.

		Creates validation split for early stopping.
		
		Sets var_sets_map and covar_cols so they are used for this
		fitting and future predictions.

		To work with early stopping, X_val and y_val must be created from
		10% of X_train and y_train.

		Args:
			X_train (pd.DataFrame or pl.DataFrame): Training data.
			y_train (pd.Series or np.ndarray): Training labels.
			val_frac (float): Fraction of data to use for validation.
				Default is 0.1.
			print_params (bool): If True, log the parameters before fitting.
		"""
		if print_params:		
			logger.debug(
				f"flaml.automl.model - params: {self.params}"
			)
			pprint(self.params)
		
		current_time = time.time()

		if "groups" in kwargs:
			kwargs = kwargs.copy()
			groups = kwargs.pop("groups")
			if self._task == "rank":
				kwargs["group"] = group_counts(groups)
		
		X_train = self._preprocess(X_train)

		# Create validation set by first creating a binary mask
		train_frac = 1 - val_frac
		n_samples = X_train.shape[0]
		val_mask = np.random.choice(
			[True, False],
			n_samples,
			p=[val_frac, train_frac]
		)

		if isinstance(X_train, pl.DataFrame):
			X_val = X_train.filter(val_mask)
			X_train = X_train.filter(~val_mask)
		else:
			X_val = X_train[val_mask]
			X_train = X_train[~val_mask]
		y_val = y_train[val_mask]
		y_train = y_train[~val_mask]

		# Create model
		non_lgbm_params = ['early_stopping_rounds']
		model = self.estimator_class(
			verbose=1,
			**{k:v for k,v in self.params.items() if k not in non_lgbm_params}
		)

		if logger.level == logging.DEBUG:
			logger.debug(f"flaml.automl.model - {model} fit started - params: {self.params}")

		early_stopping_callback = early_stopping(
			stopping_rounds=self.params['early_stopping_rounds']
		)

		if 'callbacks' in kwargs:
			kwargs['callbacks'].append(early_stopping_callback)
		else:
			kwargs['callbacks'] = [early_stopping_callback]
		
		model.fit(
			X_train, 
			y_train,
			eval_set=[(X_val, y_val)],
			**kwargs
		)
		
		if logger.level == logging.DEBUG:
			logger.debug(f"flaml.automl.model - {model} fit finished")

		train_time = time.time() - current_time
		self._model = model
		return train_time
