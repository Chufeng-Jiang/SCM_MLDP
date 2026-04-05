import pandas as pd
import numpy as np

# Read CSV file
df = pd.read_csv('./inference_output/merged.csv')

# Prepare output lines
output_lines = ['module goodRules.\n\nindex(+,-)\n']

# Column name to operator mapping
col_to_op = {
    'prob_SPLUS':  'splus',
    'prob_SMINUS': 'sminus',
    'prob_MINUSS': 'minuss'
}

# Process each row
for _, row in df.iterrows():
    constant = int(row['Constant'])

    probs = {
        'prob_SPLUS':  row['prob_SPLUS'],
        'prob_SMINUS': row['prob_SMINUS'],
        'prob_MINUSS': row['prob_MINUSS']
    }

    # Sort by probability descending
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    max_col,    max_val    = sorted_probs[0]
    second_col, second_val = sorted_probs[1]

    diff = max_val - second_val

    if diff > 0.2:
        # Only output the highest probability operator
        output_lines.append(f'op({constant}, {col_to_op[max_col]}).\n')
    else:
        # Output the top two operators
        output_lines.append(f'op({constant}, {col_to_op[max_col]}).\n')
        output_lines.append(f'op({constant}, {col_to_op[second_col]}).\n')

# Write output file
with open('goodRules.pi', 'w') as f:
    f.writelines(output_lines)

print(f"Done! Processed {len(df)} rows")
print(f"Output file: goodRules.pi")