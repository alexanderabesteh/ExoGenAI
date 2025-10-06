import pandas as pd
import numpy as np

# Load exoTest.csv
df = pd.read_csv('data/exoTest.csv')

# Get first row (first light curve)
first_row = df.iloc[0]
label = first_row.iloc[0]
flux_values = first_row.iloc[1:].values

# Create a proper CSV with time and flux columns
time = np.arange(len(flux_values))
sample_df = pd.DataFrame({
    'time': time,
    'flux': flux_values
})

sample_df.to_csv('data/sample_upload.csv', index=False)
print(f"Created sample_upload.csv")
print(f"Original label: {label} (2=exoplanet, 1=not exoplanet)")
print(f"Data points: {len(flux_values)}")