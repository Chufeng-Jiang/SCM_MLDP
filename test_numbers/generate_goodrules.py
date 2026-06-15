import pandas as pd
import numpy as np

"""
This script reads model inference results from a CSV file and generates
a Picat rule file named goodRules.pi.

For each constant, the script checks the predicted probabilities of three
possible operators:

    - SPLUS   -> splus
    - SMINUS  -> sminus
    - MINUSS  -> minuss

If the highest probability is clearly better than the second highest
probability by more than 0.2, only the highest-probability operator is kept.

Otherwise, the top two operators are both kept as possible rules.

The generated output has the following Picat-style format:

    module goodRules.

    index(+,-)

    op(Constant, Operator).

Example:

    op(12, splus).
    op(35, sminus).
"""

# ============================================================
# 1. Read the inference result CSV file
# ============================================================

# The CSV file is expected to contain at least the following columns:
#   - Constant
#   - prob_SPLUS
#   - prob_SMINUS
#   - prob_MINUSS
df = pd.read_csv('./inference_output/merged.csv')


# ============================================================
# 2. Initialize the output Picat file content
# ============================================================

# These lines define the Picat module and index declaration.
# The generated rules will be appended after this header.
output_lines = [
    'module goodRules.\n\n',
    'index(+,-)\n'
]


# ============================================================
# 3. Define the mapping from CSV probability columns to operators
# ============================================================

# The model outputs probabilities using column names such as prob_SPLUS.
# The Picat rule file should use lowercase operator names instead.
col_to_op = {
    'prob_SPLUS':  'splus',
    'prob_SMINUS': 'sminus',
    'prob_MINUSS': 'minuss'
}


# ============================================================
# 4. Process each row and generate operator rules
# ============================================================

for _, row in df.iterrows():

    # Convert the constant ID/value to an integer.
    # This assumes the Constant column contains numeric values.
    constant = int(row['Constant'])

    # Collect the predicted probabilities for the three operators.
    probs = {
        'prob_SPLUS':  row['prob_SPLUS'],
        'prob_SMINUS': row['prob_SMINUS'],
        'prob_MINUSS': row['prob_MINUSS']
    }

    # Sort operators by predicted probability in descending order.
    # The first item is the most likely operator.
    sorted_probs = sorted(
        probs.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Extract the most likely and second most likely operators.
    max_col, max_val = sorted_probs[0]
    second_col, second_val = sorted_probs[1]

    # Compute the confidence gap between the top two predictions.
    diff = max_val - second_val

    # ------------------------------------------------------------
    # Rule selection logic:
    #
    # If the top prediction is much stronger than the second one,
    # keep only the top operator.
    #
    # Otherwise, keep both top candidates because the model is not
    # confident enough to choose only one.
    # ------------------------------------------------------------
    if diff > 0.2:
        output_lines.append(
            f'op({constant}, {col_to_op[max_col]}).\n'
        )
    else:
        output_lines.append(
            f'op({constant}, {col_to_op[max_col]}).\n'
        )
        output_lines.append(
            f'op({constant}, {col_to_op[second_col]}).\n'
        )


# ============================================================
# 5. Write the generated rules to a Picat file
# ============================================================

output_file = 'goodRules.pi'

with open(output_file, 'w') as f:
    f.writelines(output_lines)


# ============================================================
# 6. Print summary information
# ============================================================

print(f"Done! Processed {len(df)} rows")
print(f"Output file: {output_file}")