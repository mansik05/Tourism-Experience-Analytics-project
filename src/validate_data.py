import pandas as pd
from pathlib import Path


# --------------------------------------------------
# PROJECT PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw"


# --------------------------------------------------
# LOAD DATASETS
# --------------------------------------------------

city = pd.read_excel(DATA_PATH / "City.xlsx")
continent = pd.read_excel(DATA_PATH / "Continent.xlsx")
country = pd.read_excel(DATA_PATH / "Country.xlsx")
item = pd.read_excel(DATA_PATH / "Item.xlsx")
mode = pd.read_excel(DATA_PATH / "Mode.xlsx")
region = pd.read_excel(DATA_PATH / "Region.xlsx")
transaction = pd.read_excel(DATA_PATH / "Transaction.xlsx")
attraction_type = pd.read_excel(DATA_PATH / "Type.xlsx")
user = pd.read_excel(DATA_PATH / "User.xlsx")


# --------------------------------------------------
# FUNCTION TO CHECK REFERENTIAL INTEGRITY
# --------------------------------------------------

def check_relationship(left_df, left_column, right_df, right_column, relationship_name):

    missing_ids = set(left_df[left_column].dropna()) - set(right_df[right_column].dropna())

    print("\n" + "=" * 70)
    print(relationship_name)
    print("=" * 70)

    print(f"Total unique IDs in source: {left_df[left_column].nunique()}")
    print(f"Missing IDs in reference table: {len(missing_ids)}")

    if len(missing_ids) == 0:
        print("Status: VALID ✓")

    else:
        print("Status: INVALID ✗")
        print("Sample missing IDs:")
        print(list(missing_ids)[:10])


# --------------------------------------------------
# TRANSACTION RELATIONSHIPS
# --------------------------------------------------

check_relationship(
    transaction,
    "UserId",
    user,
    "UserId",
    "Transaction.UserId → User.UserId"
)


check_relationship(
    transaction,
    "AttractionId",
    item,
    "AttractionId",
    "Transaction.AttractionId → Item.AttractionId"
)


check_relationship(
    transaction,
    "VisitMode",
    mode,
    "VisitModeId",
    "Transaction.VisitMode → Mode.VisitModeId"
)


# --------------------------------------------------
# USER GEOGRAPHIC RELATIONSHIPS
# --------------------------------------------------

check_relationship(
    user,
    "CityId",
    city,
    "CityId",
    "User.CityId → City.CityId"
)


check_relationship(
    user,
    "CountryId",
    country,
    "CountryId",
    "User.CountryId → Country.CountryId"
)


check_relationship(
    user,
    "RegionId",
    region,
    "RegionId",
    "User.RegionId → Region.RegionId"
)


check_relationship(
    user,
    "ContinentId",
    continent,
    "ContinentId",
    "User.ContinentId → Continent.ContinentId"
)


# --------------------------------------------------
# ATTRACTION RELATIONSHIPS
# --------------------------------------------------

check_relationship(
    item,
    "AttractionCityId",
    city,
    "CityId",
    "Item.AttractionCityId → City.CityId"
)


check_relationship(
    item,
    "AttractionTypeId",
    attraction_type,
    "AttractionTypeId",
    "Item.AttractionTypeId → Type.AttractionTypeId"
)


# --------------------------------------------------
# GEOGRAPHIC HIERARCHY
# --------------------------------------------------

check_relationship(
    city,
    "CountryId",
    country,
    "CountryId",
    "City.CountryId → Country.CountryId"
)


check_relationship(
    country,
    "RegionId",
    region,
    "RegionId",
    "Country.RegionId → Region.RegionId"
)


check_relationship(
    region,
    "ContinentId",
    continent,
    "ContinentId",
    "Region.ContinentId → Continent.ContinentId"
)