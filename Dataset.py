import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducibility
np.random.seed(0)

# Number of samples
num_samples = 1000

# Generate random data for elevator system
data = pd.DataFrame({
    'Date': pd.date_range(start='2024-01-01', periods=num_samples, freq='h'),
    'Temperature': np.random.normal(70, 10, num_samples),
    'Vibration': np.random.normal(0.5, 0.1, num_samples),
    'CurrentUsage': np.random.normal(50, 5, num_samples),
})

# Add breakdown flag for next 12 hours
breakdown_window = 12
data['Breakdown'] = np.random.choice([0, 1], size=num_samples, p=[0.9, 0.1])
data['BreakdownWithin12Hrs'] = data['Breakdown'].rolling(window=breakdown_window).max()

# Save to CSV
data.to_csv('elevator_data.csv', index=False)

# Display the first few rows of the generated dataset
print(data.head())


