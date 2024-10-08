# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB: Batch Inference Workflow Using SLM
# MAGIC
# MAGIC In this lab, you will learn how to implement a batch inference pipeline using a Small Language Model (SLM) in a production environment. The objective is to follow a structured approach to develop, test, and deploy a language model-based pipeline using tools such as MLflow, and Unity Catalog. This process focuses on effective model management and operational strategies, facilitating batch inference using Spark DataFrames, and managing model life cycles via model registration and querying.
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will need to complete the following tasks:*
# MAGIC
# MAGIC 1. **Task 1:** Create a Hugging Face question-answering pipeline and test it.
# MAGIC 2. **Task 2:** Track and register the model using MLflow and Unity Catalog.
# MAGIC 3. **Task 3:** Manage the registered model's state.
# MAGIC 4. **Task 4:** Perform single-node and multi-node batch inference.
# MAGIC 5. **Task 5:** Perform batch inference using SQL `ai_query`.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install --quiet datasets mlflow==2.12.1 transformers==4.41.2
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the Lab, run the provided classroom setup script. This script will define configuration variables necessary for the lab. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this lab, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dataset Overview
# MAGIC
# MAGIC In this Lab, you will be using the SQuAD dataset hosted on HuggingFace. This is a reading comprehension dataset which consists of questions and answers based on the provided context. Let's load and inspect the structure of the SQuAD dataset.

# COMMAND ----------

from datasets import load_dataset
from delta.tables import DeltaTable

prod_data_table_name = f"{DA.catalog_name}.{DA.schema_name}.m4_1_prod_data"
squad_dataset = load_dataset("squad")
test_spark_df = spark.createDataFrame(squad_dataset["validation"].to_pandas())
test_spark_df.write.mode("overwrite").saveAsTable(prod_data_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task 1: Develop a LLM Pipeline
# MAGIC
# MAGIC Create a language model pipeline that efficiently answers questions by leveraging pre-trained model.

# COMMAND ----------

# MAGIC %md
# MAGIC ###1.1: Create a Hugging Face Q&A Pipeline
# MAGIC Initialize a QA pipeline using a specified model tailored for question answering. This step involves selecting a model that has been optimized for the "`question-answering`" task.

# COMMAND ----------

# Import the pipeline function from the transformers library
from transformers import pipeline  
# Define variables for the model name, device mapping, and cache directory
hf_model_name = "distilbert-base-cased-distilled-squad"  
device_map = "auto"  # Automatically use the best available device (CPU or GPU)
cache_dir = DA.paths.datasets.replace("dbfs:/", "/dbfs")  # Path for caching data, adjusted for Databricks file system

# Initialize a question-answering pipeline with the specified model
qa_pipeline = pipeline(
    task=<FILL_IN>,  # Specify the task type as 'question-answering'
    model=<FILL_IN>,  # Model to be loaded
    model_kwargs={"cache_dir": cache_dir},  
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ###1.2: Test Question-Answering Pipeline
# MAGIC Validate the pipeline's functionality by running a predefined question and context to observe how the model interprets and responds.

# COMMAND ----------

# Define the context string where the model will search for answers
context = """Marie Curie was a Polish and naturalized-French physicist and chemist who conducted pioneering research on radioactivity. She was the first woman to win a Nobel Prize and the first person and only woman to win the Nobel prize twice in different scientific fields."""

# Define the question to be answered based on the given context
question = "Why is Marie Curie famous?"

# Use the question-answering pipeline to find an answer to the question from the context
answer = qa_pipeline(<FILL_IN>)

# Print the question and answer
print(f"Question: <FILL_IN>")

print(f"Answer: <FILL_IN>")
print("===============================================")

# Print the context to show the content the model used to find the answer
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Model Development and Registering
# MAGIC Track the developed model using MLflow and register it in the Unity Catalog for lifecycle management.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1: Track LLM Development with MLflow
# MAGIC
# MAGIC Log the model's parameters, configuration, and outputs to MLflow for tracking experiments, versioning, and reproducibility.

# COMMAND ----------

# TODO 
# Import necessary MLflow and related library modules for model tracking

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output

# Generate a model output using the QA pipeline for a given input to use in the model signature
output = generate_signature_output(<FILL_IN>)

# Infer a model signature that defines the input and output schema of the model
signature = infer_signature(<FILL_IN>)

# Set the name of the experiment in MLflow
experiment_name = f"/Users/{DA.username}/GenAI-As-04-Batch-Demo"
mlflow.set_experiment(<FILL_IN>)

# Define a path within the MLflow Artifacts repository to store the model
model_artifact_path = "qa_pipeline"

# Start an MLflow run to log parameters, artifacts, and models
with mlflow.start_run():
    # Log parameters used in the model; here, the model name
    <FILL_IN>,
    })

    # Define inference configuration for logging purposes, could include other configurations
    <FILL_IN>,
    }

    # Log the model along with its configuration, signature, and an example for use
    model_info = mlflow.transformers.log_model(
        transformers_model=<FILL_IN>,
        artifact_path=<FILL_IN>h,
        task=<FILL_IN>,  # Type of task for the model
        inference_config=<FILL_IN>,  # Configuration used for inference
        signature=<FILL_IN>,  # Signature that defines model input and output
        input_example={"question": "Why is Marie Curie famous?", "context": context}  # Example of input
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2: Query the MLflow Tracking Server
# MAGIC Retrieve information about the model's performance and other metrics from the MLflow tracking server.

# COMMAND ----------

# Retrieve the experiment ID using the experiment name
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
# Search for all runs in the experiment using the experiment ID
runs = mlflow.search_runs([experiment_id])
# Sort the runs by their start time in descending order and get the run ID of the latest run
last_run_id = runs.sort_values('start_time', ascending=False).iloc[0].run_id
# Construct the model URI using the last run ID and the specified artifact path
model_uri = f"runs:/{last_run_id}/{model_artifact_path}"

# COMMAND ----------

# MAGIC %md
# MAGIC ###2.3: Load Model Back as a Pipeline
# MAGIC Load the registered model from MLflow to verify its performance and integration capabilities post-registration.
# MAGIC

# COMMAND ----------

loaded_qa_pipeline = <FILL_IN>
loaded_qa_pipeline.predict(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4: Register the Model to Unity Catalog
# MAGIC Register the model in the Unity Catalog for better version control and to facilitate the deployment process.

# COMMAND ----------

from mlflow import MlflowClient
# Define the model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.qa_pipeline"
# Set the MLflow registry URI
mlflow.set_registry_uri("databricks-uc")
# Register the model in the MLflow model registry under the specified name and model URI
mlflow.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: LLM Model State Management
# MAGIC In this task, you'll manage your model's lifecycle across different stages using MLflow and Unity Catalog. By leveraging MLflow's Model Registry, you will update and maintain the model's state to enhance tracking, version control, and deployment efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ###3.1: Search and Inspect Registered Model
# MAGIC Identify and inspect the latest version of your registered model to ensure you are managing the most current and relevant iteration. This step is crucial as it determines the baseline for setting model stages or aliases.
# MAGIC
# MAGIC - Retrieve the Latest Model Version
# MAGIC - Set Model Alias

# COMMAND ----------

def get_latest_model_version(model_name_in):
    # Initialize the MLflow Client to interact with the MLflow server
    client = MlflowClient()
    
    # Search for all versions of the specified model in the Model Registry
    model_version_infos = <FILL_IN>
    
    # Extract the version numbers and return the highest (latest) version
    return max([model_version_info.version for model_version_info in model_version_infos])

# Initialize the MLflow Client for further operations
client = mlflow.tracking.MlflowClient()

# Get the latest version number of the specified model
current_model_version = get_latest_model_version(model_name)

# Set an alias 'champion' for the latest version of the model
client.<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Batch Inference
# MAGIC Perform inference using the registered model on new data, both in single-node and multi-node environments.Perform inference using the registered model on new data, both in single-node and multi-node environments.

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.1: Load the Model for Batch Inference
# MAGIC Prepare the environment and load the model from Unity Catalog for batch processing.

# COMMAND ----------

prod_data_table = f"{DA.catalog_name}.{DA.schema_name}.m4_1_prod_data"
# Read data from the specified Spark table and limit the results to the first 100 rows
prod_data_df = spark.read.table(prod_data_table).limit(100)
# Display the DataFrame to visualize the top 100 rows of the dataset
display(prod_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.2: Single-node Batch Inference
# MAGIC Conduct inference tests on a limited dataset to validate the model's response accuracy and speed in a single-node setup.
# MAGIC

# COMMAND ----------

# Load the latest version of the model from MLflow using the provided model URI
latest_model = mlflow.pyfunc.<FILL_IN>

# Convert the first two rows of the DataFrame to a Pandas DataFrame for easier manipulation
prod_data_sample_pdf = prod_data_df.limit(2).toPandas()

# Define a list of questions to be answered by the model
questions = [<FILL_IN>]

# Generate answers for each question by applying the loaded model on the context provided in the DataFrame
qa_results = [latest_model.predict({"question": q, "context": doc}) for q, doc in zip(questions, prod_data_sample_pdf["context"])]

# Import the pprint function for formatted display of objects
from pprint import pprint

# Print each result in a formatted manner using pprint for better readability
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.3: Multi-node Batch Inference
# MAGIC Scale the inference process using Spark to simulate real-world, large-scale data handling scenarios.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col

# Ensure that the input DataFrame contains 'question' and 'context' columns
prod_data_df = <FILL_IN>
prod_data_df = <FILL_IN>

prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}@champion",
    env_manager="local",
    result_type="string",
)
batch_inference_results_df = <FILL_IN>
# Display the DataFrame containing the results of the batch inference with generated answers
<FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ###4.4: Write Inference Results to Delta Table
# MAGIC Store the inference results in a Delta table to ensure data integrity and enable further analysis.
# MAGIC
# MAGIC

# COMMAND ----------

prod_data_summaries_table_name = f"{DA.catalog_name}.{DA.schema_name}.m4_1_batch_inference"
batch_inference_results_df.write.mode("append").saveAsTable(prod_data_summaries_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 5: Batch Inference Using `ai_query()`
# MAGIC
# MAGIC Utilize SQL capabilities to perform batch inference directly using SQL queries, integrating AI functions for broader accessibility and efficiency.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1: Run SQL Batch Inference
# MAGIC
# MAGIC Create a SQL query that executes an AI model inference directly within the SQL. This approach utilizes the `ai_query()` function in SQL to process batch queries against the dataset.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ###Step 1: Run SQL Batch Inference

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ai_query_inference AS (
# MAGIC   SELECT
# MAGIC     id,
# MAGIC     <FILL_IN> as generated_answer
# MAGIC   FROM m4_1_prod_data LIMIT 100
# MAGIC );

# COMMAND ----------

# MAGIC %md
# MAGIC ###5.2: Query Inference Results
# MAGIC Query the generated table to view the inference results.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Retrieve all records from the 'ai_query_inference' table to view the results
# MAGIC <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ##Cleanup Classroom
# MAGIC Run the following cell to remove lesson-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this lab, you successfully implemented a batch inference workflow using a small language model. You created a question-answering pipeline, tracked and registered the model using MLflow, managed model versions and stages with Unity Catalog, and performed both single-node and multinode batch inference. Finally, you explored an alternative method for batch inference using the `ai_query` SQL function.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>