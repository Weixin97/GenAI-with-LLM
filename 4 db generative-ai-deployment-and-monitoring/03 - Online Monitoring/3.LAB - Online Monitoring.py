# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # LAB - Online Monitoring
# MAGIC
# MAGIC In this lab, you will create an online monitor for a sample inference table using Databricks Lakehouse Monitoring. A sample inference table, extracted from a deployed Model Serving Endpoint, has been imported for you to use for monitoring.
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC *In this lab, you will need to complete the following tasks:*
# MAGIC
# MAGIC * **Task 1:** Define Evaluation Metrics
# MAGIC * **Task 2:** Unpack the Request Payload
# MAGIC * **Task 3:** Compute Metrics
# MAGIC * **Task 4:** Save the Processed Inference Table
# MAGIC * **Task 5:** Create a Monitor on the Inference Table
# MAGIC * **Task 6:** Review the Monitor Details
# MAGIC * **Task 7:** View the Monitor Dashboard

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries and load classroom configuration.

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk mlflow==2.12.1 textstat tiktoken evaluate
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference Table
# MAGIC
# MAGIC You are going to use the same inference table that we used for the demo. The inference table is pre-loaded and ready to be used.

# COMMAND ----------

inference_table_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_realtime_payload"
display(spark.sql(f"SELECT * FROM {inference_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Define Evaluation Metrics
# MAGIC In this task, you will define evaluation metrics such as toxicity, perplexity, and readability, which will be used to analyze the inference table data.
# MAGIC
# MAGIC -  Define the evaluation metrics functions using `pandas_udf`.

# COMMAND ----------

# Import necessary libraries
import tiktoken, textstat, evaluate
import pandas as pd
from pyspark.sql.functions import pandas_udf

# Define a pandas UDF to compute the number of tokens in the text
@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(<FILL_IN>)

# Define a pandas UDF to compute the toxicity of the text
@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = <FILL_IN>
  return pd.Series(toxicity.compute(<FILL_IN>)["toxicity"]).where(<FILL_IN>)

# Define a pandas UDF to compute the perplexity of the text
@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = <FILL_IN>
  return pd.Series(perplexity.compute(<FILL_IN>)["perplexities"]).where(<FILL_IN>)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2: Unpack the Request Payload
# MAGIC In this task, you will unpack the request payload from the inference table and prepare it for processing.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC - Unpack the requests as a stream.
# MAGIC - Drop unnecessary columns for monitoring jobs.

# COMMAND ----------

import os

# Reset checkpoint [for demo purposes ONLY]
checkpoint_location = os.path.join(DA.paths.working_dir, "checkpoint")
dbutils.fs.rm(checkpoint_location, True)

# Define the JSON path and type for the input requests
INPUT_REQUEST_JSON_PATH = <FILL_IN>
INPUT_JSON_PATH_TYPE = <FILL_IN>
KEEP_LAST_QUESTION_ONLY = False

# Define the JSON path and type for the output responses
OUTPUT_REQUEST_JSON_PATH = <FILL_IN>
OUPUT_JSON_PATH_TYPE = <FILL_IN>

# Unpack the requests as a stream
requests_raw_df = spark.readStream.table(inference_table_name)
requests_processed_df = unpack_requests(
    <FILL_IN>,
    <FILL_IN>,
    <FILL_IN>,
    <FILL_IN>,
    <FILL_IN>,
    <FILL_IN>
)

# Drop un-necessary columns for monitoring jobs
requests_processed_df = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Compute Metrics
# MAGIC
# MAGIC In this task, you will compute the defined evaluation metrics for the unpacked request payloads.
# MAGIC
# MAGIC - Compute the toxicity, perplexity, and token count for the input and output columns.

# COMMAND ----------

# Define the columns to measure
column_to_measure = <FILL_IN>

# Iterate over each column to measure
for column_name in column_to_measure:
    Compute the metrics and add them as new columns to the DataFrame
    requests_df_with_metrics = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Save the Processed Inference Table
# MAGIC
# MAGIC In this task, you will save the processed inference table with the computed metrics to a Delta table.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC - Create the processed inference table if it doesn't exist.
# MAGIC - Append the new unpacked payloads and metrics to the processed table.

# COMMAND ----------

from delta.tables import DeltaTable
# Define the name of the processed table
processed_table_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_processed_inferences_lab"

# Create the table if it does not exist
(DeltaTable.createOrReplace(spark)
        .tableName(<FILL_IN>) 
        .addColumns(<FILL_IN>.schema) 
        .property("delta.enableChangeDataFeed", "true") 
        .property("delta.columnMapping.mode", "name") 
        .execute()) # Execute the table creation

# Write the requests_df_with_metrics DataFrame to the processed table as a stream
(requests_df_with_metrics.writeStream
                      .trigger(availableNow=True) 
                      .format("delta") 
                      .outputMode("append") 
                      .option("checkpointLocation", <FILL_IN>)
                      .toTable(<FILL_IN>).awaitTermination())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Create a Monitor on the Inference Table
# MAGIC
# MAGIC In this task, you will create a monitor on the processed inference table using Databricks Lakehouse Monitoring.
# MAGIC
# MAGIC - Create a monitor using the `databricks-sdk`.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries
# Initialize the workspace client
w = WorkspaceClient()

try:
  # Create a monitor using the workspace client's quality_monitors service
  lhm_monitor = w.quality_monitors.create(
    table_name=<FILL_IN>,
    time_series = MonitorTimeSeries(
      timestamp_col = "timestamp",
      granularities = ["5 minutes"],
    ),
    assets_dir = <FILL_IN>,
    slicing_exprs = <FILL_IN>,
    output_schema_name=f"{DA.catalog_name}.{DA.schema_name}"
  )

# Handle any exceptions that occur during monitor creation
except Exception as lhm_exception:
  <FIll_IN>

# COMMAND ----------

from databricks.sdk.service.catalog import MonitorInfoStatus

# Get the monitor information for the processed table
monitor_info = <FILL_IN>
print(monitor_info.status)

# Check if the monitor status is pending
if monitor_info.status == MonitorInfoStatus.<FILL_IN>:
    print("Wait until monitor creation is completed...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 6: Review the Monitor Details
# MAGIC
# MAGIC In this task, you will review the details of the monitor created in the previous step. This will involve checking the **Quality** tab for the monitor details and reviewing the metrics tables generated by the monitor.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC
# MAGIC Complete following steps:
# MAGIC
# MAGIC
# MAGIC 1. **Review Monitor Details in Quality Tab**
# MAGIC    - Go to the **[Catalog](explore/data)** and find the table you monitored.
# MAGIC    - Click on the **Quality** tab to view the monitor details.
# MAGIC
# MAGIC 2. **Review Metrics Tables**
# MAGIC    - Examine the metrics tables (`*_processed_profile_metrics` and `*_processed_drift_metrics`).
# MAGIC
# MAGIC
# MAGIC **🚨Note:** Ensure that the refresh process is completed and the metrics tables are ready before reviewing the details.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 7: View the Monitor Dashboard
# MAGIC
# MAGIC In this task, you will view the Databricks SQL dashboard generated by Lakehouse Monitoring to review the data and metrics of your monitoring solution.
# MAGIC
# MAGIC **Steps:**
# MAGIC
# MAGIC Complete following steps:
# MAGIC
# MAGIC 1. **View the SQL Dashboard**
# MAGIC    - Click on **View Dashboard** to open the SQL dashboard from the **Quality** tab.
# MAGIC
# MAGIC 2. **Inspect Overall Summary Statistics**
# MAGIC    - Examine the overall summary statistics presented in the dashboard.
# MAGIC
# MAGIC 3. **Review the Created Metrics**
# MAGIC    - Review the metrics that were created in the first step of this lab to understand the data quality and model performance over time.
# MAGIC
# MAGIC
# MAGIC **🚨Note:** Make sure there is an accessible DBSQL cluster up and running to ensure dashboard creation.

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
# MAGIC In this lab, you created an online monitor using Databricks Lakehouse Monitoring. First, you defined evaluation metrics and computed these metrics for the inference table. Then, you created a monitor on the inference table. Lastly, you reviewed the monitor details and the auto-created Databricks SQL dashboard. After successfully completing this lab, you should be able to create online monitoring for an inference table that captures the inference requests of deployed AI models.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>