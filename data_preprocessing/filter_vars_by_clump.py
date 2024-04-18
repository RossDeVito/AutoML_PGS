"""Filter variants for inclusion as PRS model using just lead SNPs from each
clump from ".clumps" file output by plink2.

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

* -c, --clumps-file: Path to the '.clumps' file output by plink2.
* -o, --out-dir: Directory in which to save the output files.
	Default: '.'.
* --var-id-col: Name of the column in the clumps file that contains the
	variant IDs. Default: 'ID'.
* --chrom-col: Name of the column in the clumps file that contains the
	chromosome numbers. Default: '#CHROM'.
* --pos-col: Name of the column in the clumps file that contains the base
	pair positions. Default: 'POS'.

Example usage:

```bash
python filter_vars_by_pval.py \
	-c /path/to/clumps.clumps \
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
		'-c', '--clumps-file',
		required=True,
		help='Path to the \'.clumps\' file output by plink2.'
	)
	parser.add_argument(
		'-o', '--out-dir',
		default='.',
		help='Directory in which to save the output files. Default: \'.\'.'
	)
	parser.add_argument(
		'--var-id-col',
		default='ID',
		help='Name of the column in the clumps file that contains the variant IDs.'
	)
	parser.add_argument(
		'--chrom-col',
		default='#CHROM',
		help='Name of the column in the clumps file that contains the chromosome numbers.'
	)
	parser.add_argument(
		'--pos-col',
		default='POS',
		help='Name of the column in the clumps file that contains the base pair positions.'
	)
	
	return parser.parse_args()


if __name__ == '__main__':

	# Parse command line arguments
	args = parse_args()

	# Load clumps file
	ss_df = pd.read_csv(args.clumps_file, sep='\s+')

	# Clump lead SNPS are values in the variant ID column
	lead_variants = ss_df[args.var_id_col].to_list()

	# Create output JSON dicts
	var_ids_sets = {
		"clump": {
			"lead": lead_variants
		}
	}
	
	meta_dict = {
		"filtering": {
			"clumps_leads": True,
		},
	}

	# Save all variant IDs that pass any filter
	all_sig_variants_file = os.path.join(args.out_dir, 'filtered_vars_all.txt')
	with open(all_sig_variants_file, 'w') as f:
		f.write('\n'.join(lead_variants))

	# Add counts to meta data section
	meta_dict["filtering"]["n_var_total"] = len(lead_variants) # type: ignore
	meta_dict["filtering"]["n_var_threshold"] = { # type: ignore
		"clump": {"lead": len(lead_variants)}
	}

	# Save JSON output
	var_ids_sets_file = os.path.join(args.out_dir, 'filtered_vars_raw.json')
	with open(var_ids_sets_file, 'w') as f:
		json.dump(var_ids_sets, f, indent=4)
	
	meta_dict_file = os.path.join(args.out_dir, 'filtered_vars_meta.json')
	with open(meta_dict_file, 'w') as f:
		json.dump(meta_dict, f, indent=4)
