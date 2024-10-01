# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

# MAGIC %run ./_helper_functions

# COMMAND ----------

def create_production_text_table(self):
    """
    Load a dataset from Hugging Face, process it, and save it as a Spark DataFrame table.
    """
    from datasets import load_dataset
    from pyspark.sql import SparkSession
    from datasets.utils.logging import disable_progress_bar
    # Define a persistent cache directory
    cache_dir = "/dbfs/cache/"

    # Disable progress bars
    disable_progress_bar()

    # Load dataset from Hugging Face, limit to 50%
    dataset = load_dataset("xiyuez/red-dot-design-award-product-description", split='train[:50%]', cache_dir=cache_dir)
    
    # Extract product, category, and text columns
    products = dataset['product']
    categories = dataset['category']
    texts = dataset['text']
    
    # Initialize SparkSession
    spark = SparkSession.builder \
        .appName("Save Dataset to Table") \
        .getOrCreate()

    # Create DataFrame
    spark_df = spark.createDataFrame(zip(products, categories, texts), ["product", "category", "text"])
    
    # Save DataFrame as table
    production_table = "production_text"
    spark_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(production_table)
    
    return production_table

# Monkey patch the function
DBAcademyHelper.monkey_patch(create_production_text_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")

# COMMAND ----------

DA.create_production_text_table()

# COMMAND ----------

