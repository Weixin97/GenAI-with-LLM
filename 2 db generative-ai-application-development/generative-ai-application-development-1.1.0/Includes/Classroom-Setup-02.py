# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# MAGIC %run ./_helper_functions

# COMMAND ----------

# Create a delta table for diet features
def create_dataset(self):

    from pyspark.sql import functions as F
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    
    source_table_fullname = f"{DA.catalog_name}.{DA.schema_name}.dais_text"
    
    df = spark.read.load(f"{DA.paths.datasets}/dais.delta")

    df = df.withColumn("id", F.monotonically_increasing_id())
    df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(source_table_fullname)
    spark.sql(f"ALTER TABLE {source_table_fullname} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

    print("Dataset is created successfully.")
        
DBAcademyHelper.monkey_patch(create_dataset)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs

DA.create_dataset()                                # Load dataset and set a unique id

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")