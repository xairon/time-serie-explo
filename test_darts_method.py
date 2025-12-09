from darts import TimeSeries
import pandas as pd
import numpy as np

# Create dummy series
df = pd.DataFrame({'a': np.random.rand(10)}, index=pd.date_range('2020-01-01', periods=10))
ts = TimeSeries.from_dataframe(df)

try:
    print("Testing to_dataframe()...")
    df_out = ts.to_dataframe()
    print("✅ to_dataframe() works!")
    print(df_out.head(2))
except AttributeError:
    print("❌ to_dataframe() failed.")

try:
    print("Testing pd_dataframe()...")
    df_out = ts.pd_dataframe()
    print("✅ pd_dataframe() works!")
except AttributeError:
    print("❌ pd_dataframe() failed.")
