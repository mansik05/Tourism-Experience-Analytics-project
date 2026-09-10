import pandas as pd
import os

# Path to raw data folder
DATA_PATH = "C:/Users/mansi/Desktop/ML/Tourism Experience Analytics project/data/raw"

# Get all Excel files
files = [file for file in os.listdir(DATA_PATH) if file.endswith(".xlsx")]

for file in files:

    print("\n")
    print("=" * 70)
    print(f"FILE: {file}")
    print("=" * 70)

    # Read Excel file
    file_path = os.path.join(DATA_PATH, file)

    df = pd.read_excel(file_path)

    print("\nSHAPE:")
    print(df.shape)

    print("\nCOLUMN NAMES:")
    print(df.columns.tolist())

    print("\nDATA TYPES:")
    print(df.dtypes)

    print("\nMISSING VALUES:")
    print(df.isnull().sum())

    print("\nDUPLICATE ROWS:")
    print(df.duplicated().sum())

    print("\nFIRST 5 ROWS:")
    print(df.head())