import pandas as pd
from pathlib import Path


# ==================================================
# PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed"

# Create processed folder if it doesn't exist
PROCESSED_DATA_PATH.mkdir(exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

print("Loading datasets...")

city = pd.read_excel(RAW_DATA_PATH / "City.xlsx")

continent = pd.read_excel(RAW_DATA_PATH / "Continent.xlsx")

country = pd.read_excel(RAW_DATA_PATH / "Country.xlsx")

item = pd.read_excel(RAW_DATA_PATH / "Item.xlsx")

mode = pd.read_excel(RAW_DATA_PATH / "Mode.xlsx")

region = pd.read_excel(RAW_DATA_PATH / "Region.xlsx")

transaction = pd.read_excel(RAW_DATA_PATH / "Transaction.xlsx")

attraction_type = pd.read_excel(RAW_DATA_PATH / "Type.xlsx")

user = pd.read_excel(RAW_DATA_PATH / "User.xlsx")


# ==================================================
# CLEANING
# ==================================================

print("Cleaning datasets...")


# Rename transaction VisitMode because it contains IDs
transaction = transaction.rename(
    columns={"VisitMode": "VisitModeId"}
)


# Handle missing User City IDs
user["CityId"] = user["CityId"].astype("Int64")


# ==================================================
# MERGE TRANSACTION + USER
# ==================================================

print("Merging transaction and user data...")

master_df = transaction.merge(
    user,
    on="UserId",
    how="left"
)


# ==================================================
# MERGE VISIT MODE
# ==================================================

print("Adding visit mode information...")

master_df = master_df.merge(
    mode,
    on="VisitModeId",
    how="left"
)


# ==================================================
# MERGE ATTRACTION INFORMATION
# ==================================================

print("Adding attraction information...")

master_df = master_df.merge(
    item,
    on="AttractionId",
    how="left"
)


# ==================================================
# MERGE ATTRACTION TYPE
# ==================================================

print("Adding attraction type information...")

master_df = master_df.merge(
    attraction_type,
    on="AttractionTypeId",
    how="left"
)


# ==================================================
# ADD USER CITY
# ==================================================

print("Adding user city information...")

user_city = city.rename(
    columns={
        "CityId": "UserCityId",
        "CityName": "UserCity"
    }
)[["UserCityId", "UserCity"]]

master_df = master_df.merge(
    user_city,
    left_on="CityId",
    right_on="UserCityId",
    how="left"
)

master_df.drop(
    columns=["UserCityId"],
    inplace=True
)


# ==================================================
# ADD ATTRACTION CITY
# ==================================================

print("Adding attraction city information...")

attraction_city = city.rename(
    columns={
        "CityId": "AttractionCityId",
        "CityName": "AttractionCity"
    }
)[["AttractionCityId", "AttractionCity"]]

master_df = master_df.merge(
    attraction_city,
    on="AttractionCityId",
    how="left"
)


# ==================================================
# ADD COUNTRY
# ==================================================

print("Adding country information...")

master_df = master_df.merge(
    country[["CountryId", "Country"]],
    on="CountryId",
    how="left"
)


# ==================================================
# ADD REGION
# ==================================================

print("Adding region information...")

master_df = master_df.merge(
    region[["RegionId", "Region"]],
    on="RegionId",
    how="left"
)


# ==================================================
# ADD CONTINENT
# ==================================================

print("Adding continent information...")

master_df = master_df.merge(
    continent[["ContinentId", "Continent"]],
    on="ContinentId",
    how="left"
)


# ==================================================
# DATA CLEANING AFTER MERGING
# ==================================================

print("Performing final cleaning...")


# Replace missing user cities
master_df["UserCity"] = master_df["UserCity"].fillna("Unknown")


# Replace missing attraction city if any
master_df["AttractionCity"] = master_df["AttractionCity"].fillna("Unknown")


# ==================================================
# REMOVE UNNECESSARY PLACEHOLDER VALUES
# ==================================================

# Convert "-" placeholder values to "Unknown"

columns_to_clean = [
    "VisitMode",
    "UserCity",
    "Country",
    "Region",
    "Continent"
]

for column in columns_to_clean:

    master_df[column] = master_df[column].replace(
        "-",
        "Unknown"
    )


# ==================================================
# REMOVE DUPLICATES
# ==================================================

master_df.drop_duplicates(inplace=True)


# ==================================================
# SAVE CLEANED DATA
# ==================================================

output_path = (
    PROCESSED_DATA_PATH /
    "master_tourism_data.csv"
)

master_df.to_csv(
    output_path,
    index=False
)


# ==================================================
# FINAL REPORT
# ==================================================

print("\n" + "=" * 70)

print("MASTER DATASET CREATED SUCCESSFULLY")

print("=" * 70)

print(f"\nShape: {master_df.shape}")

print("\nColumns:")

print(master_df.columns.tolist())


print("\nMissing Values:")

print(master_df.isnull().sum())


print("\nDuplicate Rows:")

print(master_df.duplicated().sum())


print("\nDataset saved at:")

print(output_path)