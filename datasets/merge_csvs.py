import argparse
import pandas as pd

# Create argument parser
parser = argparse.ArgumentParser(description='CSV Appender')

# Add the arguments
parser.add_argument('--csv1_path', type=str, help='Path to the first CSV file')
parser.add_argument('--csv2_path', type=str, help='Path to the second CSV file')
parser.add_argument('--output_csv_path', type=str, help='Path to the output CSV file')
# parser.add_argument('append_columns', nargs='+', help='Column names to append')

# Parse the arguments
args = parser.parse_args()

# Load CSV files
csv1_data = pd.read_csv(args.csv1_path)
csv2_data = pd.read_csv(args.csv2_path)

# Select the columns to append
# append_data = csv2_data[args.append_columns]

# Append append_data to csv1_data
combined_data = pd.concat([csv1_data, csv2_data], axis=0)

# Save combined_data to a new CSV file
combined_data.to_csv(args.output_csv_path, index=False)

print("Data appended and saved successfully!")
