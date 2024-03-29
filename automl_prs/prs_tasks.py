"""Custom FLAML tasks for PRS with this library.

Key differences:

	- Takes polars DataFrames as input
"""

import logging

import numpy as np
import pandas as pd
import polars as pl
from sklearn.utils import shuffle

from flaml.automl.task.generic_task import GenericTask
from flaml.config import RANDOM_SEED

from automl_prs import (
	LGBMEstimatorPRS, LGBMEstimatorPRSv1, LGBMEstimatorPRSv2
)


logger = logging.getLogger(__name__)


class PRSTask(GenericTask):

	def __init__(
		self,
		task_name: str,
	):
		"""Constructor.

		Args:
			task_name: String name for this type of task. Used when the Task
				can be generic and implement a number of types of sub-task.
		"""
		super().__init__(task_name)

	@property
	def estimators(self):
		if self._estimators is None:
			self._estimators = {
				"lgbm": LGBMEstimatorPRS,
				"lgbm_v1": LGBMEstimatorPRSv1,
				"lgbm_v2": LGBMEstimatorPRSv2,
			}

		return self._estimators

	def validate_data(
		self,
		automl,
		state,
		X_train_all,
		y_train_all,
		dataframe,
		label,
		X_val=None,
		y_val=None,
		groups_val=None,
		groups=None,
	):
		# Check valid task
		if self.is_nlp():
			raise ValueError(
				"AutoML-PRS does not support NLP tasks."
			)
		if self.is_ts_forecast():
			raise ValueError(
				"AutoML-PRS does not support time series forecasting tasks."
			)

		# Check training data
		if X_train_all is not None and y_train_all is not None:
			assert isinstance(X_train_all, (pd.DataFrame, pl.DataFrame)), (
				"X_train_all must a pandas of polars dataframe."
			)
			assert isinstance(
				y_train_all, (np.ndarray, pd.Series)
			), "y_train_all must be a numpy array or a pandas series."

			assert np.all(np.array(X_train_all.shape) > 0), (
				"Input data must not be empty."
			)

			if isinstance(y_train_all, np.ndarray):
				y_train_all = y_train_all.flatten()
			assert X_train_all.shape[0] == y_train_all.shape[0], "# rows in X_train must match length of y_train."

			automl._df = isinstance(X_train_all, (pd.DataFrame, pl.DataFrame))
			automl._nrow, automl._ndim = X_train_all.shape
			
			if self.is_ts_forecast():
				raise ValueError(
					"AutoML-PRS does not support time series forecasting tasks."
				)
			
			X, y = X_train_all, y_train_all
		else:
			raise ValueError(
				"X_train+y_train required with AutoML-PRS. Cannot use dataframe+label."
			)

		# Skip FLAML data transformation for PRS tasks, we'll do our own
		automl._skip_transform = True
		automl._transformer = automl._label_transformer = False
		automl._X_train_all, automl._y_train_all = X, y

		automl._sample_weight_full = state.fit_kwargs.get(
			"sample_weight"
		)  # NOTE: _validate_data is before kwargs is updated to fit_kwargs_by_estimator

		# Check validation data
		if X_val is not None and y_val is not None:
			assert isinstance(X_val, (pd.DataFrame, pl.DataFrame)), (
				"X_val must be a pandas or polars dataframe."
			)
			assert isinstance(y_val, (np.ndarray, pd.Series)), (
				"y_val must be a numpy array or a pandas series."
			)
			assert np.all(np.array(X_val.shape) > 0), (
				"Input data must not be empty."
			)
			if isinstance(y_val, np.ndarray):
				y_val = y_val.flatten()

			assert X_val.shape[0] == y_val.shape[0], (
				"# rows in X_val must match length of y_val."
			)
			
			state.X_val = X_val
			state.y_val = y_val
		else:
			raise ValueError(
				"X_val and y_val required with AutoML-PRS."
			)

		if groups is not None and len(groups) != automl._nrow:
			# groups is given as group counts
			state.groups = np.concatenate([[i] * c for i, c in enumerate(groups)])
			assert len(state.groups) == automl._nrow, "the sum of group counts must match the number of examples"
			state.groups_val = (
				np.concatenate([[i] * c for i, c in enumerate(groups_val)]) if groups_val is not None else None
			)
		else:
			state.groups_val = groups_val
			state.groups = groups

		automl.data_size_full = len(automl._y_train_all)


	def prepare_data(
		self,
		state,
		X_train_all,
		y_train_all,
		auto_augment,
		eval_method,
		split_type,
		split_ratio,
		n_splits,
		data_is_df,
		sample_weight_full,
	):
		X_val, y_val = state.X_val, state.y_val

		is_spark_dataframe = False
		self.is_spark_dataframe = is_spark_dataframe
		if (
			self.is_classification()
			and auto_augment
			and state.fit_kwargs.get("sample_weight")
			is None  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
			and split_type in ["stratified", "uniform"]
			and not self.is_token_classification()
		):
			raise ValueError(
				"AutoML-PRS does not support auto_augment with classification tasks."
			)

		if split_type in ["uniform", "stratified"]:
			if sample_weight_full is not None:
				X_train_all, y_train_all, state.sample_weight_all = shuffle(
					X_train_all,
					y_train_all,
					sample_weight_full,
					random_state=RANDOM_SEED,
				)
				state.fit_kwargs[
					"sample_weight"
				] = (
					state.sample_weight_all
				)  # NOTE: _prepare_data is before kwargs is updated to fit_kwargs_by_estimator
				if isinstance(state.sample_weight_all, pd.Series):
					state.sample_weight_all.reset_index(drop=True, inplace=True)
			else:
				X_train_all, y_train_all = shuffle(X_train_all, y_train_all, random_state=RANDOM_SEED)
			if isinstance(X_train_all, pd.DataFrame):
				X_train_all.reset_index(drop=True, inplace=True)
			if isinstance(y_train_all, pd.Series):
				y_train_all.reset_index(drop=True, inplace=True)

		X_train, y_train = X_train_all, y_train_all
		state.groups_all = state.groups

		if X_val is None:
			raise ValueError("X_val is required with AutoML-PRS.")
		
		state.data_size = X_train.shape					# type: ignore
		state.data_size_full = len(y_train_all)			# type: ignore
		state.X_train, state.y_train = X_train, y_train
		state.X_val, state.y_val = X_val, y_val
		state.X_train_all = X_train_all
		state.y_train_all = y_train_all

		state.kf = None
		
		return