"""Create parquet file which will be used as input to auto-ML models.

Each p-value and window size combination will use a subset of the features
from the genotype data. This script saves a JSON file 'filtered_vars.json'
that contains the variant sets for each p-value and window size combination
with names that will match the parquet files.

IID is used as the sample ID column.

Args:

* -f, --filtered-vars-raw: Path to the 'filtered_vars_raw.json' file output by
	'filter_vars_by_pval.py'. This file contains the variant sets for each
	p-value and window size combination with names that are based on the
	original PGEN file. The .raw file output by plink2 modifies the variant
	names to include the allele the dosage is for, which this script corrects
	for in the output JSON file.
* -r, --raw-geno: Path to the raw genotype data TSV file. This will be used
	to map column names and then create the parquet file.
* -o, --out-dir: Directory in which to save the parquet files. Default: '.'.
* --out-parquet-fname: Name of output parquet file. Default:
	'filtered_vars.parquet'.
* --out-json-fname: Name of output JSON file. Contains updated mapping
	of which variants are part of which sets (to match changes by plink2
	export). Default: 'filtered_vars.json'.
"""

import argparse
import os
import json
from collections import defaultdict

import pandas as pd
import polars as pl
from tqdm.autonotebook import tqdm


def parse_args():
	"""Parse command line arguments."""
	parser = argparse.ArgumentParser(
		description='Create parquet file which will be used as input to '
			'auto-ML models.'
	)
	parser.add_argument(
		'-f', '--filtered-vars-raw',
		required=True,
		help='Path to the \'filtered_vars_raw.json\' file output by '
			'\'filter_vars_by_pval.py\'. This file contains the variant sets '
			'for each p-value and window size combination with names that are '
			'based on the original PGEN file. The .raw file output by plink2 '
			'modifies the variant names to include the allele the dosage is '
			'for, which this script corrects for in the output JSON file.'
	)
	parser.add_argument(
		'-r', '--raw-geno',
		required=True,
		help='Path to the raw genotype data TSV file. This will be used to '
			'map column names and then create the parquet file.'
	)
	parser.add_argument(
		'-o', '--out-dir',
		default='.',
		help='Directory in which to save the parquet files. Default: \'.\'.'
	)
	parser.add_argument(
		'--out-parquet-fname',
		default='filtered_vars.parquet',
		help='Name of output parquet file. Default: \'filtered_vars.parquet\'.'
	)
	parser.add_argument(
		'--out-json-fname',
		default='filtered_vars.json',
		help='Name of output JSON file. Contains updated mapping of which '
			'variants are part of which sets (to match changes by plink2 '
			'export). Default: \'filtered_vars.json\'.'
	)
	return parser.parse_args()


def get_var_name_mapping(col_name):
	"""Given a variant column name from the plink2 raw export file, returns
	the original variant name and the raw variant name as a tuple.

	Gets the original variant name by removing the '_{allele_dosage_is_for}'
	part of the column name.
	"""
	return (
		'_'.join(col_name.split('_')[:-1]),
		col_name
	)


if __name__ == '__main__':

	args = parse_args()

	# Load variant sets
	with open(os.path.join(args.filtered_vars_raw), 'r') as f:
		raw_var_sets = json.load(f)

	# Load raw genotype data header with pandas
	print('Loading raw genotype header...')
	geno_header = pd.read_csv(
		args.raw_geno,
		sep='\t',
		nrows=0,
	)

	# Map variant column names
	print('Mapping variant column names...')
	nongeno_cols = set(['FID', 'IID', 'PAT', 'MAT', 'SEX', 'PHENOTYPE'])
	geno_col_mapping = dict(
		[get_var_name_mapping(c) for c in geno_header.columns if c not in nongeno_cols]
	)

	# Create updated variant sets
	print('Updating variant sets...')
	updated_var_sets = defaultdict(dict)
	for p_val in raw_var_sets.keys():
		for window in raw_var_sets[p_val].keys():
			updated_var_sets[p_val][window] = [
				geno_col_mapping[var] for var in raw_var_sets[p_val][window]
			]

	# Save updated variant sets
	print('Saving updated variant sets...')
	with open(os.path.join(args.out_dir, args.out_json_fname), 'w') as f:
		json.dump(updated_var_sets, f)

	# Scan and sink with polars
	raw_dtypes = {'IID': pl.String}
	for col in geno_col_mapping.values():
		raw_dtypes[col] = pl.Float32	# type: ignore
		
	lazy_df = pl.scan_csv(
		args.raw_geno,
		separator='\t',
		n_rows=1000,
		low_memory=False,
		dtypes=raw_dtypes,
	).select(
		*list(raw_dtypes.keys())
	)

	# Check that there are no missing values
	print('Checking for missing values...')
	null_counts = lazy_df.null_count().collect(streaming=True)
	n_missing = null_counts.sum_horizontal().item()
	if n_missing > 0:
		raise ValueError(
			f'Found {n_missing} missing values in the genotype data.'
		)
	
	# Sink to parquet
	print('Saving parquet file...')
	lazy_df.sink_parquet(
		os.path.join(args.out_dir, args.out_parquet_fname),
		compression="snappy"
	)
