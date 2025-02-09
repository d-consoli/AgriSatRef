"""
additional script to read and plot crated reference dataset
"""

import xarray as xr
import matplotlib.pyplot as plt
import pathlib


# Automatically set module root based on the installed package location
module_root = pathlib.Path(__file__).parent.resolve().parent

zarr_file = r"{}\tests\ref_dataset\UAVmask_mean_32632_FRIEN_WRg.zarr".format(module_root)


ds = xr.open_zarr(zarr_file)

print(ds)

# Check if 'bare_soil' exists in the dataset
if "bare_soil" not in ds:
    raise ValueError("Variable 'bare_soil' not found in the dataset!")

# Select the first time slice (if time exists)
if "time" in ds.dims:
    da = ds["bare_soil"].isel(time=0)  # Select the first time step
else:
    da = ds["bare_soil"]

#da = ds['QC_RMSE_per_time_step']

# Plot the variable
plt.figure(figsize=(10, 6))
da.plot(cmap="viridis")
#plt.title("Bare Soil Fraction")
plt.show()