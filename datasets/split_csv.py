import argparse
import pandas as pd

# Create an argument parser
parser = argparse.ArgumentParser(description='Extract and rename columns from a CSV file')

# Add the arguments
parser.add_argument('--input_file', type=str, help='Path to the input CSV file')
parser.add_argument('--output_file', type=str, help='Path to the output CSV file')
parser.add_argument('--text_column', type=str, help='Name of the text column to extract')
parser.add_argument('--label_column', type=str, help='Name of the label column to extract')

# Parse the arguments
args = parser.parse_args()

# Read the input CSV file
df = pd.read_csv(args.input_file)

# Extract the desired columns
df_new = df[[args.text_column, args.label_column]]

# Rename the columns
df_new.columns = ['text', 'label']

# Save the result as a new CSV file
df_new.to_csv(args.output_file, index=False)
