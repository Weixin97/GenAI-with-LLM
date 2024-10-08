# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic init, schemas and catalogs
spark.sql(f"USE CATALOG {DA.catalog_name}")
spark.sql(f"USE SCHEMA {DA.schema_name}")

## define host and toke
DA.scope_name = f"genai_training_{DA.unique_name(sep='_')}"
w = WorkspaceClient()
try:
    w.secrets.delete_scope(DA.scope_name)
except:
    pass

try:
    w.secrets.create_scope(DA.scope_name)
    w.secrets.put_secret(DA.scope_name, key="depl_demo_host", string_value='https://' + dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().get('browserHostName').getOrElse(None))
    w.secrets.put_secret(DA.scope_name, key="depl_demo_token", string_value=dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None))
except:
    print(f'unable to create secret scope {DA.scope_name}')


DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe examples and models presented in this c   ourse are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")