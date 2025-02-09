# AgriSatRef
A module for downscaling high-resolution spatial information and organizing it into an xarray dataset.


## Database Integration with dbflow (SQLite)

For SQLite database functionality, the dbflow package can be installed to manage database connections and execute SQL queries efficiently.
### Custom SQL Query Management

    Custom Folder: If your database contains geoinformation or requires additional organization, you can create a custom folder within the module to store your SQL queries.
    Main Script: SQL queries can alternatively be defined directly in the main script if a separate folder is not needed.


### Configuration via INI File

To streamline setup and customization, an INI configuration file can be created. 
This file allows you to define custom paths, including the location of the custom SQL query folder and other relevant settings.

Paths should be defined relative to the location of this config.ini file.
Use forward slashes (`/`) for cross-platform compatibility.

```ini
[paths]
custom_sql_dir = ./custom/sql
custom_db_structure = ./custom/db_structure.py
executed_sql_dir = ../_sql_executed
```
The script `main.py` demonstrates the case example for the RCM Project, covering all steps presented in **Figure 1**.

As an input, it requires **high-resolution classification** and a **coarse-resolution data grid** to be downscaled to.

### **Figure 1: Reference Dataset Creation Workflow**
![Reference dataset creation workflow](https://github.com/Aranil/AgriSatRef/blob/main/_images/Figure_Workflow_Steps.png)

[View Full-Sized Image](https://github.com/Aranil/AgriSatRef/blob/main/_images/Figure_Workflow_Steps.png)


Step 4 is currently under development.

STAC Conversion & Zenodo Bulk Upload Status

Certain parts of the scripts for converting data to STAC format and uploading bulk data to Zenodo are still under development. These functionalities are in progress, and some features may not be fully operational. Users should expect potential modifications and improvements in upcoming updates.
- io_handler/stac_generator.py
- io_handler/zenodo_helper.py


## Usage and Example Data
Examples of how to use the module **AgriSatRef**, along with sample input data, are located in the **`examples/`** folder. The example files are numbered for easy reference.

---

## Folder Structure

### Input Folders (Contains Input Data & Intermediate Results)
| Folder | Description |
|--------|-------------|
| `data/Classification/` | Input for **Step 1a** (Classification of high-resolution data) |
| `data/Fboundary/` | Input for **Step 1a** (Field boundary data) |
| `data/S1_grid/` | Input for **Step 1b** (Sentinel-1 Grid data) |
| `data/output_zarr/` | Input for **Step 1b** (Processed Sentinel-1 Zarr dataset) |

###  Output Folders
| Folder | Description |
|--------|-------------|
| `ref_dataset/` | Stores the **results of Step 2 and 3** |

---

## Notes
- Ensure all input data is correctly placed in the respective folders before running the module.
- Intermediate results will be generated inside the `data/` directory as processing steps progress.
- Final outputs are stored in the **`ref_dataset/`** folder.


Contributors: [Panos Koutsikos](https://github.com/PanosKoutsikos), 
              [Christoph Liedel](https://github.com/Flunsiana)