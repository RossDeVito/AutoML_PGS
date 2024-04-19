"""Filter variants for inclusion as PRS model using just SNPs used by BASIL.

Saves variant IDs to a text file, which can be used as input to plink 
--extract to filter the genotype data to just these variants.

Has three outputs:

1. Text file with variant IDs one-per-line are clump lead SNPs. This can be
used as input to plink --extract to filter the genotype data to just these
variants. Will be named 'filtered_vars_all.txt'.

2. A JSON file in the format of those that store which variants are included
for each p-value and window size combination. Will be named 
'filtered_vars_raw.json'. First level key is 'clump', second level key is
'lead', and values are lists of variant IDs.

3. A meta data JSON file, named 'fitered_vars_meta.json'. The meta data
section includes info on:
	- the number of variants that pass each threshold
	- the parameters of the filtering process.

Args:

* -b, --basil-included-file: Path to the CSV (or one ID per line text file)
	containing the variant IDs that were included in the BASIL model.
* -o, --out-dir: Directory in which to save the output files.
	Default: '.'.

Example usage:

```bash
python filter_vars_by_pval.py \
	-b /path/to/basil_included_file.csv \
	-o /path/to/output_dir
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
		'-b', '--basil-included-file',
		required=True,
		help='Path to the CSV (or one ID per line text file) containing the '
			'variant IDs that were included in the BASIL model.'
	)
	parser.add_argument(
		'-o', '--out-dir',
		default='.',
		help='Directory in which to save the output files. Default: \'.\'.'
	)
	
	return parser.parse_args()


if __name__ == '__main__':

	# Parse command line arguments
	args = parse_args()

	# Load BASIL included variants
	incl_id_series = pd.read_csv(
		args.basil_included_file,
		header=None,
	).iloc[:, 0]

	# Drop IDs that do not contain [numbers]:[numbers]_\
	incl_id_series = incl_id_series[incl_id_series.str.contains(r'\d+:\d+_')]

	# Drop everything after the last underscore
	incl_id_series = incl_id_series.apply(lambda x: '_'.join(x.split('_')[:-1]))

	# Get
	incl_variants = incl_id_series.to_list()

	# Create output JSON dicts
	var_ids_sets = {
		"basil": {
			"included": incl_variants
		}
	}
	
	meta_dict = {
		"filtering": {
			"basil": True,
		},
	}

	# Save all variant IDs that pass any filter
	all_sig_variants_file = os.path.join(args.out_dir, 'filtered_vars_all.txt')
	with open(all_sig_variants_file, 'w') as f:
		f.write('\n'.join(incl_variants))

	# Add counts to meta data section
	meta_dict["filtering"]["n_var_total"] = len(incl_variants) # type: ignore
	meta_dict["filtering"]["n_var_threshold"] = { # type: ignore
		"basil": {"included": len(incl_variants)}
	}

	# Save JSON output
	var_ids_sets_file = os.path.join(args.out_dir, 'filtered_vars_raw.json')
	with open(var_ids_sets_file, 'w') as f:
		json.dump(var_ids_sets, f, indent=4)
	
	meta_dict_file = os.path.join(args.out_dir, 'filtered_vars_meta.json')
	with open(meta_dict_file, 'w') as f:
		json.dump(meta_dict, f, indent=4)
