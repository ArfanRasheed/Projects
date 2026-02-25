import pandas as pd

df = pd.read_csv("311_requests.csv", low_memory=False)

print("Loading complete, selecting columns...")

# Use exact column names from your file
cols = [
    "Unique Key", "Created Date", "Closed Date", "Agency",
    "Agency Name", "Problem (formerly Complaint Type)",
    "Problem Detail (formerly Descriptor)", "Borough", "Status",
    "Latitude", "Longitude"
]
df = df[cols]

print("Columns selected, renaming...")

# Rename to clean names for Power BI
df = df.rename(columns={
    "Problem (formerly Complaint Type)": "Complaint Type",
    "Problem Detail (formerly Descriptor)": "Descriptor"
})

print("Parsing dates...")

# Parse dates with explicit format to avoid warning
df["Created Date"] = pd.to_datetime(df["Created Date"], format="mixed", errors="coerce")
df["Closed Date"] = pd.to_datetime(df["Closed Date"], format="mixed", errors="coerce")

# Calculate resolution time
df["Resolution Days"] = (df["Closed Date"] - df["Created Date"]).dt.days

print("Cleaning data...")

# Keep open requests (no closed date) but filter bad resolution times
df = df[df["Resolution Days"].between(0, 365) | df["Resolution Days"].isna()]

# Standardize borough names
df["Borough"] = df["Borough"].str.title().str.strip()
df = df[df["Borough"] != "Unspecified"]

# Drop rows missing critical fields
df = df.dropna(subset=["Complaint Type", "Borough", "Agency"])

# Add time columns
df["Month"] = df["Created Date"].dt.month_name()
df["Day of Week"] = df["Created Date"].dt.day_name()
df["Hour"] = df["Created Date"].dt.hour
df["Year"] = df["Created Date"].dt.year

print("Exporting...")

df.to_csv("311_clean.csv", index=False)
print(f"Done. Clean dataset shape: {df.shape}")
print(df.head())