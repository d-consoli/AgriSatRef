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


```ini
[paths]
custom_sql_dir = ./custom/sql
custom_db_structure = ./custom/db_structure.py
executed_sql_dir = ../_sql_executed
```