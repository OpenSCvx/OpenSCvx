import os

# Set Equinox error handling to return NaN instead of crashing
os.environ["EQX_ON_ERROR"] = "nan"
