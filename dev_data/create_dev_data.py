"""Simulate data to use as input for the AutoML PRS model.

Creates files for the following arguments of fit_automl_prs.py:

	* -g, --geno-parquet: Path to the parquet file containing the genotype
		data.
	* -v, --var-subsets: Path to the JSON file containing the variant
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
"""

import os
import json

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':

	# Set random seed for reproducibility
	np.random.seed(0)

	# Options
	num_samples = 5000
	num_variants = 1000
	num_covars = 5
	covar_min_max = (0, 100)
	noise_to_signal_ratio = 1.0 # Noise added by randomly selecting
								# |causal_vars| * noise_to_signal_ratio
								# causal variants, shuffling their values,
								# and treating them as inputs to a linear
								# function to generate the phenotype

	# Create sample IDs with format '{x}_{x}' where x is a random 6
	# digit integer
	sample_ids = set()

	while len(sample_ids) < num_samples:
		id_int = np.random.randint(100000, 999999)
		sample_ids.add(f'{id_int}_{id_int}')

	# Split sample IDs into training, validation, and test sets roughly 8:1:1
	sample_ids = list(sample_ids)
	np.random.shuffle(sample_ids)
	train_ids = sample_ids[:int(0.8 * num_samples)]
	val_ids = sample_ids[int(0.8 * num_samples):int(0.9 * num_samples)]
	test_ids = sample_ids[int(0.9 * num_samples):]

	# Save sample IDs to 'train_ids.txt', 'val_ids.txt', and 'test_ids.txt'
	with open('train_ids.txt', 'w') as f:
		f.write('\n'.join(train_ids))
	with open('val_ids.txt', 'w') as f:
		f.write('\n'.join(val_ids))
	with open('test_ids.txt', 'w') as f:
		f.write('\n'.join(test_ids))

	# Create genotype data with float values ranging from 0 to 2
	geno_df = pd.DataFrame({'IID': sample_ids})

	geno_col_names = [f'var_{i}' for i in range(num_variants)]
	geno_col_vals = np.random.uniform(0, 2, (num_samples, num_variants))
	geno_df = pd.concat(
		[geno_df, pd.DataFrame(geno_col_vals, columns=geno_col_names)],
		axis=1,
	)

	for i in range(num_variants):
		geno_df[f'var_{i}'] = np.random.uniform(0, 2, num_samples)

	# Convert genotype data to polars where the IID column is a string and
	# the variant columns are float32
	geno_dtypes = {'IID': pl.String}
	for col in geno_df.columns[1:]:
		geno_dtypes[col] = pl.Float32	# type: ignore

	geno_df = pl.from_pandas(geno_df)

	# Cast genotype data to the types output by raw_to_input_parquet.py
	geno_df = geno_df.with_columns(
		**{
			col_name: pl.col(col_name).cast(col_dtype)
	 		for col_name, col_dtype in geno_dtypes.items()				
		}
	)

	# Save genotype data to 'geno.parquet'
	geno_df.write_parquet('geno.parquet', compression='snappy')

	# Create variant subsets for made up p-value and window sizes
	var_names = [c for c in geno_df.columns if c != 'IID']
	var_names = list(np.random.permutation(var_names))

	var_sets = {
		'1e-8': {
			'0': var_names[:250],
			'5000': var_names[:500],
		},
		'1e-12': {
			'0': var_names[0:400],
			'5000': var_names[0:1000],
		},
	}

	# Save variant subsets to 'filtered_vars.json'
	with open('var_subsets.json', 'w') as f:
		json.dump(var_sets, f)

	# Randomly select half of first 1000 variants to be true causal variants
	causal_vars = np.random.choice(
		var_names[:1000],
		int(0.5 * 1000),
		False
	).astype(str).tolist()

	# Create covariate data with one col that is 0 or 1 and for the rest
	# float values ranging from covar_min to covar_max
	covar_df = pd.DataFrame({'IID': sample_ids})

	covar_col_names = [f'covar_{i}' for i in range(num_covars)]
	covar_col_vals = np.random.uniform(
		*covar_min_max, (num_samples, num_covars - 1)
	)
	covar_col_vals = np.hstack(
		[np.random.randint(0, 2, (num_samples, 1)), covar_col_vals]
	)
	covar_df = pd.concat(
		[covar_df, pd.DataFrame(covar_col_vals, columns=covar_col_names)],
		axis=1,
	)
	covar_df['covar_0'] = covar_df['covar_0'].astype(int)

	# Save covariate data to 'covars.tsv'
	covar_df.to_csv('covars.tsv', sep='\t', index=False)

	# Simulate phenotype
	# Join covars and causal genotype columns on 'IID'
	causal_df = covar_df.merge(
		geno_df.select(*(['IID'] + causal_vars)).to_pandas(),
		on='IID',
	)

	causal_cols = causal_df.columns[1:]

	# Add shuffled causal columns to covar_df to simulate noise
	noise_cols = []

	for i, col in enumerate(np.random.choice(
		causal_cols,
		int(len(causal_cols) * noise_to_signal_ratio),
		replace=True
	)):
		noise_cols.append(
			np.random.permutation(causal_df[col].values)
		)

	noise_df = pd.DataFrame(
		np.vstack(noise_cols).T,
		columns=[f'noise_{i}' for i in range(len(noise_cols))]
	)
	causal_df = pd.concat([causal_df, noise_df], axis=1)

	# Draw random beta for non-IID columns and add phenotype column
	# as a linear combination of all columns

	betas = np.random.uniform(-1, 1, len(causal_df.columns) - 1)
	causal_df['phenotype'] = np.dot(
		causal_df.values[:, 1:],
		betas
	)

	# Plot the distribution of the phenotype
	sns.displot(causal_df['phenotype'])		# type: ignore
	plt.show()

	# Save phenotype data to 'sim_pheno.pheno'
	causal_df[['IID', 'phenotype']].to_csv(
		'sim_pheno.tsv',
		sep='\t',
		index=False
	)

	# Create model configuration JSON files
	basic_config = {
		'model_type': 'lgbm',
		'metric': 'r2',
		'task': 'regression',
		'time_budget': 60,
	}
	with open('basic_config.json', 'w') as f:
		json.dump(basic_config, f)



	

	

	





