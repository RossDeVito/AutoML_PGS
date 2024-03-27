"""Fit AutoML PRS model.

Underlying model is either a (possibly regularized) linear regression
or a LightGBM model. Specifics of the model are defined in the model
configuration JSON file.

Requires:

	- A parquet file containing the genotype data.
	- A JSON file containing the variant subsets for different p-value
		and window size combinations.
	- A tab-separated file containing the phenotype data.
	- A tab-separated file containing the covariate data.
	- One ID per line text files containing the IDs of the samples to use
		for the training, validation, and test sets.
	- A model configuration JSON file.

Model configuration JSON file has the keys:

	- model_type (str): One of 'lgbm'. TODO add linear options.
	- metric (str): Valid metric with FLAML see https://microsoft.github.io/FLAML/docs/Use-Cases/Task-Oriented-AutoML#optimization-metric
		Valid options include 'r2', 'mae', 'mse', 'rmse', 'mape'.
	- task (str): One of 'regression', 'classification'.
	- time_budget (int): Time budget in seconds for the model fitting.

Outputs:
	- Best model configuration as a JSON file ('best_model_config.json').
	- Predictions for the validation and test sets as CSV files named
		'val_preds.csv' and 'test_preds.csv'.
	- A JSON file 'runtime.json' with the runtime of the model fitting in
		seconds under the the key 'runtime_seconds'.
	- The log file from FLAML AutoML.fit() ('fit.log').

Args:

	* -g, --geno-parquet: Path to the parquet file containing the genotype
		data.
	* -s, --var-subsets: Path to the JSON file containing the variant
		subsets for different p-value and window size combinations.
	* -p, --pheno: Path to the tab-separated file containing the phenotype
		data.
	* -c, --covars: Path to the tab-separated file containing the covariate
		data.
	* -t, --train-ids: Path to the text file containing the IDs of the
		samples to use for the training set.
	* -v, --val-ids: Path to the text file containing the IDs of the
		samples to use for the validation set.
	* -e, --test-ids: Path to the text file containing the IDs of the
		samples to use for the test set.
	* -m, --model-config: Path to the model configuration JSON file.
	* -o, --out-dir: Directory in which to save the output files.
		Default: '.'.
	* -i, --id-col: Name of the column in the genotype data that contains
		the sample IDs. Default: 'IID'.


python fit_automl_prs.py  -g ../dev_data/geno.parquet -v ../dev_data/var_subsets.json  -p ../dev_data/sim_pheno.tsv  -c ../dev_data/covars.tsv  --train-ids ../dev_data/train_ids.txt  --val-ids ../dev_data/val_ids.txt  --test-ids ../dev_data/test_ids.txt -m ../dev_data/basic_config.json -o ../dev_data/output
"""

import argparse
import json
import pickle
import os
from pprint import pprint

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from flaml import AutoML
from flaml import tune
from flaml.automl.data import get_output_from_log

from automl_prs import PRSTask


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Fit AutoML PRS model.'
	)
	parser.add_argument(
		'-g', '--geno-parquet',
		required=True,
		help='Path to the parquet file containing the genotype data.'
	)
	parser.add_argument(
		'-v', '--var-subsets',
		required=True,
		help='Path to the JSON file containing the variant subsets for '
			'different p-value and window size combinations.'
	)
	parser.add_argument(
		'-p', '--pheno',
		required=True,
		help='Path to the tab-separated file containing the phenotype data.'
	)
	parser.add_argument(
		'-c', '--covars',
		required=True,
		help='Path to the tab-separated file containing the covariate data.'
	)
	parser.add_argument(
		'--train-ids',
		required=True,
		help='Path to the text file containing the IDs of the samples to '
			'use for the training set.'
	)
	parser.add_argument(
		'--val-ids',
		required=True,
		help='Path to the text file containing the IDs of the samples to '
			'use for the validation set.'
	)
	parser.add_argument(
		'--test-ids',
		required=True,
		help='Path to the text file containing the IDs of the samples to '
			'use for the test set.'
	)
	parser.add_argument(
		'-m', '--model-config',
		required=True,
		help='Path to the model configuration JSON file.'
	)
	parser.add_argument(
		'-o', '--out-dir',
		default='.',
		help='Directory in which to save the output files.'
	)
	parser.add_argument(
		'-i', '--id-col',
		default='IID',
		help='Name of the column in the genotype data that contains the '
			'sample IDs.'
	)
	return parser.parse_args()


def main():
	args = parse_args()
	print('Args:')
	pprint(vars(args))

	# Load model configuration
	print('Loading model configuration...')
	with open(args.model_config, 'r') as f:
		model_config = json.load(f)
	# Save config to out_dir
	with open(os.path.join(args.out_dir, 'input_model_config.json'), 'w') as f:
		json.dump(model_config, f, indent=4)

	# Load sample ID sets as lists
	print('Loading sample ID sets...')
	with open(args.train_ids, 'r') as f:
		train_ids = f.read().splitlines()
	with open(args.val_ids, 'r') as f:
		val_ids = f.read().splitlines()
	with open(args.test_ids, 'r') as f:
		test_ids = f.read().splitlines()

	# Load variant subsets and create hyperparameter space
	print('Loading variant subsets...')
	with open(args.var_subsets, 'r') as f:
		var_subsets = json.load(f)

	# Flatten var_subsets by combining p-value and window size keys
	new_var_subsets = dict()
	for p_val in var_subsets:
		for window in var_subsets[p_val]:
			new_var_subsets[
				f"p-val:{p_val}, window:{window}"
			] = var_subsets[p_val][window]
	var_subsets = new_var_subsets
	
	cutoff_keys = list(var_subsets.keys())
	cutoff_counts = [len(var_subsets[k]) for k in cutoff_keys]

	cutoff_hp_space = {
		'filter_threshold': {
			'domain': tune.choice(cutoff_keys),
			'low_cost_init_value': cutoff_keys[np.argmin(cutoff_counts)],
			'cat_hp_cost': np.log(cutoff_counts).tolist(),
		}
	}

	# Load phenotype and covariate data
	print('Loading phenotype data...')
	pheno_df = pl.read_csv(
		args.pheno,
		separator='\t',
	)
	pheno_df_cols = [c for c in pheno_df.columns if c != args.id_col]
	assert len(pheno_df_cols) == 1, 'Phenotype file must have exactly one phenotype column.'
	pheno_name = pheno_df_cols[0]

	print('Loading covariate data...')
	covars_df = pl.read_csv(
		args.covars,
		separator='\t',
	)
	covar_names = [col for col in covars_df.columns if col != args.id_col]

	# Load genotype data for samples in the training and validation sets
	train_ids_expr = pl.lit(train_ids)
	val_ids_expr = pl.lit(val_ids)

	geno_lazy = pl.scan_parquet(args.geno_parquet)

	print('Loading training genotype data...')
	train_geno_df = geno_lazy.filter(
		pl.col(args.id_col).is_in(train_ids)
	).collect(streaming=True)

	print('Loading validation genotype data...')
	val_geno_df = geno_lazy.filter(
		pl.col(args.id_col).is_in(val_ids)
	).collect(streaming=True)

	# Join genotype data with phenotype and covariate data, then pop labels
	# and sample IDs
	print('Joining training genotype data with phenotype and covariate data...')
	train_df = train_geno_df.join(
		pheno_df,
		args.id_col
	).join(
		covars_df,
		args.id_col
	)
	train_ids = train_df[args.id_col].to_numpy()
	train_labels = train_df[pheno_name].to_numpy()

	train_df = train_df.drop(args.id_col, pheno_name)

	print('Joining validation genotype data with phenotype and covariate data...')
	val_df = val_geno_df.join(
		pheno_df,
		args.id_col
	).join(
		covars_df,
		args.id_col
	)
	val_ids = val_df[args.id_col].to_numpy()
	val_labels = val_df[pheno_name].to_numpy()

	val_df = val_df.drop(args.id_col, pheno_name)

	# Fit AutoML PRS model

	# Create AutoML object and set default parameters
	automl = AutoML(
		auto_augment=False,
		hpo_method='bs',	# BlendSearch
		log_type='all',
		log_training_metric=True,
		skip_transform=True,
		retrain_full='budget',	# do best effort to retrain without violating the time budget.
		sample=True,
		min_sample_size=80_000,
		early_stop=True,
		starting_points='static',
		verbose=3,
		ensemble=False,
	)

	automl.fit(
		X_train=train_df,
		y_train=train_labels,
		X_val=val_df,
		y_val=val_labels,
		task=PRSTask(model_config['task']),
		estimator_list=[model_config['model_type']],
		custom_hp={
			model_config['model_type']: cutoff_hp_space
		},
		fit_kwargs_by_estimator={
			model_config['model_type']: {
				'var_sets_map': var_subsets,
				'covar_cols': covar_names,
			}
		},
		time_budget=model_config['time_budget'],
		metric=model_config['metric'],
		log_file_name=os.path.join(args.out_dir, 'fit.log')
	)

	# Print best model
	print(f'\nBest model: {automl.best_estimator}', flush=True)
	pprint(automl.best_config)
	
	# Save best model config as JSON
	with open(os.path.join(args.out_dir, 'best_model_config.json'), 'w') as f:
		json.dump({
				'best_estimator': automl.best_estimator,
				'best_config': automl.best_config,
			},
			f,
			indent=4
		)

	# Save best model
	with open(os.path.join(args.out_dir, 'best_model.pkl'), 'wb') as f:
		pickle.dump(automl.model, f, protocol=4)

	# Plot learning curve
	(
		time_history, best_valid_loss_history, valid_loss_history,
		config_history, metric_history
	) = get_output_from_log(
		os.path.join(args.out_dir, 'fit.log'),
		time_budget=model_config['time_budget'],
	)

	if model_config['metric'] in {'r2'}:
		best_valid_loss_history = 1 - np.array(best_valid_loss_history)
		valid_loss_history = 1 - np.array(valid_loss_history)

	# Plot best curve and all results curve on same plot
	plt.step(
		time_history,
		best_valid_loss_history,
		where='post',
		label='Best Model'
	)
	plt.plot(
		time_history, 
		valid_loss_history,
		'--',
		label='All Models'
	)
	plt.ylim(
		bottom=0,
		top=max(max(best_valid_loss_history), max(valid_loss_history)) * 1.05
	)
	plt.legend()
	plt.title('Learning Curve')
	plt.xlabel('Wall Clock Time (s)')
	plt.ylabel(model_config['metric'].title())
	plt.savefig(
		os.path.join(args.out_dir, 'learning_curve.png'),
		dpi=200
	)
	plt.close()

	# Load test data and evaluate best model
	del train_df, train_labels

	print('Loading test genotype data...')
	test_geno_df = geno_lazy.filter(
		pl.col(args.id_col).is_in(test_ids)
	).collect(streaming=True)

	# Join genotype data with phenotype and covariate data, then pop labels
	# and sample IDs
	test_df = test_geno_df.join(
		pheno_df,
		args.id_col
	).join(
		covars_df,
		args.id_col
	)
	test_ids = test_df[args.id_col].to_numpy()
	test_labels = test_df[pheno_name].to_numpy()

	test_df = test_df.drop(args.id_col, pheno_name)

	# Make predictions for validation and test sets
	print('\nPredicting for validation and test sets with best model')
	val_preds = automl.predict(val_df)
	test_preds = automl.predict(test_df)

	# Save predictions as CSV files with sample IDs
	val_preds_df = pd.DataFrame({
		args.id_col: val_ids,
		'pred': val_preds,
	})
	val_preds_df.to_csv(
		os.path.join(args.out_dir, 'val_preds.csv'),
		index=False
	)

	test_preds_df = pd.DataFrame({
		args.id_col: test_ids,
		'pred': test_preds,
	})
	test_preds_df.to_csv(
		os.path.join(args.out_dir, 'test_preds.csv'),
		index=False
	)


if __name__ == '__main__':
	main()