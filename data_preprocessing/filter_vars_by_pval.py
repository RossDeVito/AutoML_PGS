"""Filter variants for inclusion as PRS model input by Bonferonni corrected
p-value & window size. 

Will only include thresholds with a number of variants less than or equal
to --max-variants.

Saves superset of all variant IDs that pass any filter to a text file, which
can be used as input to plink --extract to filter the genotype data to just
these variants.

Has three outputs:

1. Text file with variant IDs one-per-line that pass all the the p-value
and window size filters (the superset of all pairs of p-value and window 
size combinations). This can be used as input to plink --extract to filter
the genotype data to just these variants. Will be named 'filtered_vars_all.txt'.

2. A JSON file that stores which variants are included for each p-value and
window size combination. Will be named 'filtered_vars_raw.json'. First level
keys are p-value thresholds, second level keys are window sizes, and values are
lists of variant IDs.

3. A meta data JSON file, named 'fitered_vars_meta.json'. The meta data
section includes info on:
	- the number of variants that pass each threshold
	- the parameters of the filtering process.

Args:

* -s, --sum-stats-file: Path to whitespace delimited summary statistics
	file that includes the var_id_col, p_val_col, chrom_col, and pos_col
	columns.
* -m, --max-variants: Maximum number of variants to include in the PRS model.
	Default: 80,000.
* -p, --pval-thresh: P-value threshold(s) to use for filtering.
* -w, --window-bp: Window size(s) in base pairs around each variant below the 
	p-value threshold. Default: 0.
* -o, --out-dir: Directory in which to save the output files.
	Default: '.'.
* --var-id-col: Name of the column in the summary statistics file that
	contains the variant IDs. Default: 'ID'.
* --p-val-col: Name of the column in the summary statistics file that contains
	the p-values. Default: 'P'.
* --chrom-col: Name of the column in the summary statistics file that contains
	the chromosome numbers. Default: '#CHROM'.
* --pos-col: Name of the column in the summary statistics file that contains
	the base pair positions. Default: 'POS'.

Example usage:

```bash
python filter_vars_by_pval.py \
	-s /path/to/sum_stats.tsv \
	-p 5e-8 1e-5 0.05 \
	-w 0 10000 100000 \
	-o /path/to/output_dir \
	--var-id-col ID \
	--p-val-col P \
	--chrom-col '#CHROM' \
	--pos-col POS
```
"""

import argparse
import os
import json
from itertools import product
from collections import defaultdict

import pandas as pd
from tqdm import tqdm


def parse_args():
	parser = argparse.ArgumentParser(
		description=__doc__,
		formatter_class=argparse.RawDescriptionHelpFormatter
	)
	parser.add_argument(
		'-s', '--sum-stats-file',
		required=True,
		help='Path to summary statistics file that includes the var_id_col, '
			'p_val_col, chrom_col, and pos_col columns.'
	)
	parser.add_argument(
		'-m', '--max-variants',
		type=int,
		default=80_000,
		help='Maximum number of variants to include in the PRS model.'
	)
	parser.add_argument(
		'-p', '--pval-thresh',
		nargs='+',
		required=True,
		help='P-value threshold(s) to use for filtering.'
	)
	parser.add_argument(
		'-w', '--window-bp',
		nargs='+',
		type=int,
		default=[0],
		help='Window size(s) in base pairs around each variant below the '
			'p-value threshold. Default: 0.'
	)
	parser.add_argument(
		'-o', '--out-dir',
		default='.',
		help='Directory in which to save the filtered variant files. Default: '
			'\'\'.'
	)
	parser.add_argument(
		'--var-id-col',
		default='ID',
		help='Name of the column in the summary statistics file that contains '
			'the variant IDs. Default: \'ID\'.'
	)
	parser.add_argument(
		'--p-val-col',
		default='P',
		help='Name of the column in the summary statistics file that contains '
			'the p-values. Default: \'P\'.'
	)
	parser.add_argument(
		'--chrom-col',
		default='#CHROM',
		help='Name of the column in the summary statistics file that contains '
			'the chromosome numbers. Default: \'#CHROM\'.'
	)
	parser.add_argument(
		'--pos-col',
		default='POS',
		help='Name of the column in the summary statistics file that contains '
			'the base pair positions. Default: \'POS\'.'
	)
	
	return parser.parse_args()


if __name__ == '__main__':

	# Parse command line arguments
	args = parse_args()

	# Load summary statistics
	ss_df = pd.read_csv(args.sum_stats_file, sep='\t')

	# Bonferonni correct p-values
	ss_df[args.p_val_col] = ss_df[args.p_val_col] * len(ss_df)

	# Save all variant IDs that pass any filter
	all_sig_variants = set()

	# Create output JSON dicts
	var_ids_sets = defaultdict(dict)	# First level keys are p-value
										# thresholds, second level keys are 
										# window sizes, and values are lists of
										# variant IDs
	
	meta_dict = {
		"filtering": {
			"p_val_col": args.p_val_col,
			"chrom_col": args.chrom_col,
			"pos_col": args.pos_col,
			"pval_thresh": args.pval_thresh,
			"window_bp": args.window_bp,
			"max_variants": args.max_variants,
		},
	}

	# Loop through each p-value threshold and window size and filter variants
	for p_val_thresh_str, window_bp in tqdm(
		product(args.pval_thresh, args.window_bp),
		desc='Filtering variants',
		total=len(args.pval_thresh) * len(args.window_bp)
	):
		print(f"Filtering variants for p-value threshold: {p_val_thresh_str} window: {window_bp}")

		# Filter by p-value
		pval_thresh = float(p_val_thresh_str)

		sig_variants = ss_df[ss_df[args.p_val_col] < pval_thresh]

		# Check if number of significant variants is less than max_variants
		if len(sig_variants) > args.max_variants:
			print(f"\tMax number of variants exceeded: {len(sig_variants)} > {args.max_variants}")
			continue

		if window_bp > 0:
			variant_ids = set(sig_variants[args.var_id_col])
			
			# Loop through each significant variant and extract nearby variants
			for _, variant in sig_variants.iterrows():
				nearby_variants = ss_df[
					(ss_df[args.chrom_col] == variant[args.chrom_col]) & 
					(ss_df[args.pos_col] >= variant[args.pos_col] - window_bp) & 
					(ss_df[args.pos_col] <= variant[args.pos_col] + window_bp)
				]

				variant_ids.update(nearby_variants[args.var_id_col])

			sig_variants = ss_df[ss_df[args.var_id_col].isin(variant_ids)]

			if len(sig_variants) > args.max_variants:
				print(f"\tMax number of variants exceeded: {len(sig_variants)} > {args.max_variants}")
				continue

		# Add selected variants to output JSON dict
		var_ids_sets[p_val_thresh_str][window_bp] = sig_variants[
			args.var_id_col
		].to_list()

		all_sig_variants.update(sig_variants[args.var_id_col])

		print(f"Added {len(sig_variants)} variants for p-val: {p_val_thresh_str} window: {window_bp}")
		print(f"\tTotal included variants: {len(all_sig_variants)}")

	# Save all variant IDs that pass any filter
	all_sig_variants_file = os.path.join(args.out_dir, 'filtered_vars_all.txt')
	with open(all_sig_variants_file, 'w') as f:
		f.write('\n'.join(all_sig_variants))

	# Add counts to meta data section
	meta_dict["filtering"]["n_var_total"] = len(all_sig_variants)
	meta_dict["filtering"]["n_var_threshold"] = defaultdict(dict)

	for p_val_thresh_str in var_ids_sets.keys():
		for window_bp in var_ids_sets[p_val_thresh_str].keys():
			meta_dict["filtering"]["n_var_threshold"][p_val_thresh_str][window_bp] = len(
				var_ids_sets[p_val_thresh_str][window_bp]
			)

	# Save JSON output
	var_ids_sets_file = os.path.join(args.out_dir, 'filtered_vars_raw.json')
	with open(var_ids_sets_file, 'w') as f:
		json.dump(var_ids_sets, f, indent=4)
	
	meta_dict_file = os.path.join(args.out_dir, 'filtered_vars_meta.json')
	with open(meta_dict_file, 'w') as f:
		json.dump(meta_dict, f, indent=4)
