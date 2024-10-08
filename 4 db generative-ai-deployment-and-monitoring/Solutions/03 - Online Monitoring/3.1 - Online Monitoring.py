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
# MAGIC # Online Monitoring Demo
# MAGIC
# MAGIC
# MAGIC **In this demo, we will lay the foundation for monitoring our GenAI applications using Lakehouse Monitoring.** Lakehouse Monitoring is an automated data monitoring solution provided by Databricks. We are going to use it to monitor the input/output data of GenAI applications.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Unpack an inference table to structure model serving endpoints' requests/responses
# MAGIC * Describe the basics of using Lakehouse Monitoring
# MAGIC * Set up a monitor on an unpacked/processed inference table

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk mlflow textstat tiktoken evaluate
# MAGIC
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-03

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"Dataset Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demonstration, we will be introducing **Lakehouse Monitoring** for GenAI applications.
# MAGIC
# MAGIC To complete this demo, we'll follow the below steps:
# MAGIC
# MAGIC 1. Unpack the Inference Table for an existing Model Serving Endpoint.
# MAGIC 2. Compute some LLM metrics.
# MAGIC 2. Describe the basics of using Lakehouse Monitoring.
# MAGIC 3. Set up a more robust monitor using Lakehouse Monitoring

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step1: Create Inference Table
# MAGIC
# MAGIC To demonstrate monitoring, we will create a pre-populated sample inference table. **The inference table is already created for you in the course config notebook.**

# COMMAND ----------

from delta.tables import DeltaTable


inference_table_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_realtime_payload"

# Check whether the table exists before proceeding.
inference_table_exists = DeltaTable.forName(spark, inference_table_name)

if inference_table_exists:
    display(spark.sql(f"SELECT * FROM {inference_table_name} LIMIT 5"))

else:
    raise("Inference table does not exist, please re-run/verify classroom setup script")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Unpack the Inference Table and Compute LLM Metrics
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-1.png?raw=true" style="float: right" width="900px">

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1: Unpacking the table
# MAGIC
# MAGIC The request and response columns contains model prompts and output as a `string`.
# MAGIC
# MAGIC **Note :** the format depends on model definition but inputs are usually represented as JSON with TF format, and the output depends on model definition as well.
# MAGIC
# MAGIC
# MAGIC We will use Spark JSON Path annotation to directly access the prompt and completions as string, concatenate them together with an `array_zip` and finally `explode` the content to have single prompt/completions rows.
# MAGIC
# MAGIC *Note: This will be made easier within the product directly--we provide this notebook to simplify this task for now.*

# COMMAND ----------

# The format of the input payloads, following the TF "inputs" serving format with a "query" field.
# Single query input format: {"inputs": [{"query": "User question?"}]}
INPUT_REQUEST_JSON_PATH = "inputs[*].query"

# Matches the schema returned by the JSON selector (inputs[*].query is an array of string)
INPUT_JSON_PATH_TYPE = "array<string>"
KEEP_LAST_QUESTION_ONLY = False

# Answer format: {"predictions": ["answer"]}
OUTPUT_REQUEST_JSON_PATH = "predictions"

# Matches the schema returned by the JSON selector (predictions is an array of string)
OUPUT_JSON_PATH_TYPE = "array<string>"

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's **test** the unpacking logic **on a sample** in batch mode.

# COMMAND ----------

# Unpack using provided helper function
payloads_sample_df = spark.table(inference_table_name).where('status_code == 200').limit(10)
payloads_unpacked_sample_df = unpack_requests(
    payloads_sample_df,
    INPUT_REQUEST_JSON_PATH,
    INPUT_JSON_PATH_TYPE,
    OUTPUT_REQUEST_JSON_PATH,
    OUPUT_JSON_PATH_TYPE,
    KEEP_LAST_QUESTION_ONLY
)

display(payloads_unpacked_sample_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2: Compute [Prompt-Completion] Evaluation Metrics
# MAGIC
# MAGIC Let's compute some text evaluation metrics such as toxicity, perplexity and readability.
# MAGIC
# MAGIC These will be analyzed by Lakehouse Monitoring so that we can understand how these metrics change over time.
# MAGIC
# MAGIC *Note: This is a non-exhaustive list and these calculations will be automatically performed out-of-the-box within the product --we provide this notebook to simplify this task for now.*

# COMMAND ----------

import tiktoken, textstat, evaluate
import pandas as pd
from pyspark.sql.functions import pandas_udf


@pandas_udf("int")
def compute_num_tokens(texts: pd.Series) -> pd.Series:
  encoding = tiktoken.get_encoding("cl100k_base")
  return pd.Series(map(len, encoding.encode_batch(texts)))

@pandas_udf("double")
def flesch_kincaid_grade(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.flesch_kincaid_grade(text) for text in texts])
 
@pandas_udf("double")
def automated_readability_index(texts: pd.Series) -> pd.Series:
  return pd.Series([textstat.automated_readability_index(text) for text in texts])

@pandas_udf("double")
def compute_toxicity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  toxicity = evaluate.load("toxicity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(toxicity.compute(predictions=texts.fillna(""))["toxicity"]).where(texts.notna(), None)

@pandas_udf("double")
def compute_perplexity(texts: pd.Series) -> pd.Series:
  # Omit entries with null input from evaluation
  perplexity = evaluate.load("perplexity", module_type="measurement", cache_dir="/tmp/hf_cache/")
  return pd.Series(perplexity.compute(data=texts.fillna(""), model_id="gpt2")["perplexities"]).where(texts.notna(), None)

# COMMAND ----------

from pyspark.sql import DataFrame
from pyspark.sql.functions import col


def compute_metrics(requests_df: DataFrame, column_to_measure = ["input", "output"]) -> DataFrame:
  for column_name in column_to_measure:
    requests_df = (
      requests_df.withColumn(f"toxicity({column_name})", compute_toxicity(col(column_name)))
                 .withColumn(f"perplexity({column_name})", compute_perplexity(col(column_name)))
                 .withColumn(f"token_count({column_name})", compute_num_tokens(col(column_name)))
                 .withColumn(f"flesch_kincaid_grade({column_name})", flesch_kincaid_grade(col(column_name)))
                 .withColumn(f"automated_readability_index({column_name})", automated_readability_index(col(column_name)))
    )
  return requests_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3: Incrementally unpack & compute metrics from payloads and save to final `_processed` table
# MAGIC
# MAGIC 1. Read `inference_table_name` delta table as stream and unpack payloads
# MAGIC 2. Drop unnecessary columns from streaming dataframe
# MAGIC 3. Calculate (some) LLM-related evaluation metrics
# MAGIC 4. Initialize the `processed_table` (create table using schema from streaming dataframe)
# MAGIC     1. Enable Delta's [Change-Data-Feed](https://docs.delta.io/2.0.0/delta-change-data-feed.html) to ensure incremental processing of payloads
# MAGIC     2. Enable support for special characters in column names (by enabling [column mapping](https://docs.delta.io/latest/delta-column-mapping.html))
# MAGIC 5. Append/Write new processed payloads and metrics to `processed_table_name` delta table

# COMMAND ----------

import os

# Reset checkpoint [for demo purposes ONLY]
checkpoint_location = os.path.join(DA.paths.working_dir, "checkpoint")
dbutils.fs.rm(checkpoint_location, True)

# Unpack the requests as a stream.
requests_raw_df = spark.readStream.table(inference_table_name)
requests_processed_df = unpack_requests(
    requests_raw_df,
    INPUT_REQUEST_JSON_PATH,
    INPUT_JSON_PATH_TYPE,
    OUTPUT_REQUEST_JSON_PATH,
    OUPUT_JSON_PATH_TYPE,
    KEEP_LAST_QUESTION_ONLY
)

# Drop un-necessary columns for monitoring jobs
requests_processed_df = requests_processed_df.drop("date", "status_code", "sampling_fraction", "client_request_id", "databricks_request_id")

# Compute text evaluation metrics
requests_with_metrics_df = compute_metrics(requests_processed_df)

# COMMAND ----------

def create_processed_table_if_not_exists(table_name, requests_with_metrics):
    """
    Helper method to create processed table using schema
    """
    (
      DeltaTable.createOrReplace(spark) # to avoid dropping everytime .createIfNotExists(spark)
        .tableName(table_name)
        .addColumns(requests_with_metrics.schema)
        .property("delta.enableChangeDataFeed", "true")
        .property("delta.columnMapping.mode", "name")
        .execute()
    )

# COMMAND ----------

# Persist the requests stream, with a defined checkpoint path for this table
processed_table_name = f"{DA.catalog_name}.{DA.schema_name}.rag_app_processed_inferences"
create_processed_table_if_not_exists(processed_table_name, requests_with_metrics_df)

# Append new unpacked payloads & metrics
(requests_with_metrics_df.writeStream
                      .trigger(availableNow=True)
                      .format("delta")
                      .outputMode("append")
                      .option("checkpointLocation", checkpoint_location)
                      .toTable(processed_table_name).awaitTermination())

# Display the table (with requests and text evaluation metrics) that will be monitored.
display(spark.table(processed_table_name))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Describe the Basics of Lakehouse Monitoring
# MAGIC
# MAGIC While the built-in Model Serving inference tables are a simple and effective way to collect information on our application, we are able to do a lot more when we use Lakehouse Monitoring.
# MAGIC
# MAGIC Databricks Lakehouse Monitoring lets you monitor the *statistical properties* and *quality* of all data. This includes the data associated with classical ML and GenAI models and model-serving endpoints.
# MAGIC
# MAGIC Here are a few applications of Lakehouse Monitoring for GenAI:
# MAGIC
# MAGIC * Monitor the statistical properties of the data used in the table associated with a Vector Search index
# MAGIC * Monitor the relative performance of different entities over time (i.e how is model version A performing compared to version B)
# MAGIC * Monitor unstructured/text-related metrics of prompt/completions of Model Serving endpoints
# MAGIC
# MAGIC ### How Lakehouse Monitoring Works
# MAGIC
# MAGIC Lakehouse Monitoring is focused on the **data** associated with your application – in other words, Delta tables in Unity Catalog.
# MAGIC
# MAGIC To monitor a table, you create a **monitor** attached to the table. To monitor the performance of a machine learning model, you attach the **monitor** to an inference table that holds the model’s inputs and corresponding predictions.
# MAGIC
# MAGIC Visualize Lakehouse Monitoring for Machine Learning below:
# MAGIC
# MAGIC <img src="https://docs.databricks.com/en/_images/lakehouse-monitoring-overview.png" style="float: right" width="800px">
# MAGIC
# MAGIC In the above visual, the flow of data is broken down into a few steps:
# MAGIC
# MAGIC 1. Data starts in an **input table**
# MAGIC 2. Data is processed through an ML pipeline
# MAGIC 3. Data is written to an **inference table**
# MAGIC
# MAGIC Lakehouse Monitoring is designed to monitor the **input table** and the **inference table**
# MAGIC
# MAGIC **Note:** The above diagram is for classical machine learning, but similar principles apply for GenAI.
# MAGIC
# MAGIC ### Types of Monitors
# MAGIC
# MAGIC Lakehouse Monitoring has three distinct types of monitors, detailed below:
# MAGIC
# MAGIC | **Type** | **Description** |
# MAGIC |------| ------------|
# MAGIC | Time series | Use for tables that contain a time series dataset based on a timestamp column. Monitoring computes data quality metrics across time-based windows of the time series.|
# MAGIC | InferenceLog   | Use for tables that contain the request log for a model. Each row is a request, with columns for the timestamp, the model inputs, the corresponding prediction, and (optional) ground-truth label. Monitoring compares model performance and data quality metrics across time-based windows of the request log.|
# MAGIC | Snapshot    | Use for all other types of tables. Monitoring calculates data quality metrics over all data in the table. The complete table is processed with every refresh.|
# MAGIC
# MAGIC ### Lakehouse Monitoring Output
# MAGIC
# MAGIC When a monitor is set up, Lakehouse Monitoring will automatically generate:
# MAGIC
# MAGIC 1. Two **metrics table** which are delta tables containing profiling and drift measurements described above
# MAGIC 2. A **dashboard** to visualize the calculated metrics stored in the above tables
# MAGIC A series of **SQL alerts** [AWS]() | [Azure]() can be manually created by users to alert stakeholders (or **destinations** [AWS]() | [Azure]() such as slack/teams webhook, pagerduty, email notifications) of certain data characteristics
# MAGIC
# MAGIC Let's get started with an example.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Create a Monitor on the processed inference table
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-eval-online-2.png?raw=true" style="float: right" width="900px">
# MAGIC
# MAGIC We're going to use Databricks to create a monitor on the `_processed` inference table supporting a previously deployed application/endpoint during the course.
# MAGIC
# MAGIC See the Lakehouse Monitoring documentation ([AWS](https://docs.databricks.com/lakehouse-monitoring/index.html) | [Azure](https://learn.microsoft.com/azure/databricks/lakehouse-monitoring/index)) for more details on the parameters and the expected usage.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1: (Optional) Using the UI 
# MAGIC To set up this monitor, we'll follow the below steps:
# MAGIC
# MAGIC 1. Navigate to the **Catalog**
# MAGIC 2. Find the table that we want to monitor
# MAGIC 3. Click the **Quality** tab.
# MAGIC 4. Click the **Get started** button.
# MAGIC 5. In Create monitor, choose the options we want to set up the monitor, *shown below*.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-online-monitoring-config.png" align="center" width="80%" />
# MAGIC
# MAGIC **Note:** We are going to set up a **Timeseries** profile here.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2: Using the databricks-sdk
# MAGIC
# MAGIC See the reference material for the [Databricks Lakehouse Monitoring API](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/catalog/quality_monitors.html#).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries

# Create monitor using databricks-sdk's `quality_monitors` client
w = WorkspaceClient()

try:
  lhm_monitor = w.quality_monitors.create(
    table_name=processed_table_name, # Always use 3-level namespace
    time_series = MonitorTimeSeries(
      timestamp_col = "timestamp",
      granularities = ["5 minutes"],
    ),
    assets_dir = os.getcwd(),
    slicing_exprs = ["model_id"],
    output_schema_name=f"{DA.catalog_name}.{DA.schema_name}"
  )

except Exception as lhm_exception:
  print(lhm_exception)

# COMMAND ----------

from databricks.sdk.service.catalog import MonitorInfoStatus

monitor_info = w.quality_monitors.get(processed_table_name)
print(monitor_info.status)

if monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_PENDING:
    print("Wait until monitor creation is completed...")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The monitor assests will be created in this directory. 
# MAGIC
# MAGIC **⏰ Expected monitor creation & refresh time: ~7 mins**

# COMMAND ----------

monitor_info = w.quality_monitors.get(processed_table_name)
assert monitor_info.status == MonitorInfoStatus.MONITOR_STATUS_ACTIVE, "Monitoring is not ready yet. Check back in a few minutes or view the monitoring creation process for any errors."

# COMMAND ----------

# MAGIC %md
# MAGIC **🚨 Note 1: Refresh time will take around 5 minutes after the monitor is created.**
# MAGIC
# MAGIC **🚨 Note 2: Make sure there's an accessible DBSQL cluster up and running to ensure dashboard creation**

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3: Refresh Metrics Manually
# MAGIC
# MAGIC You can run "refresh metrics" to manually refresh the metrics and Dashboards.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3: Review the Monitor and Data in the Catalog Explorer
# MAGIC
# MAGIC Once the monitor is created, we can review the **Quality** tab in the original table's catalog view.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-online-monitoring-quality.png" width="100%">
# MAGIC
# MAGIC **Question:** What information do you see?
# MAGIC
# MAGIC We can also take a look at the tables generated by the monitor. For our **Timeseries** example, these include:
# MAGIC
# MAGIC * `*_processed_profile_metrics`
# MAGIC * `*_processed_drift_metrics`
# MAGIC
# MAGIC **Question:** What is the record level of this data?
# MAGIC
# MAGIC **🚨 Note: Make sure the refresh process is completed and metrics table are ready.**

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {monitor_info.profile_metrics_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC **Question:** What do you notice about our drift metrics table?

# COMMAND ----------

display(spark.sql(f"SELECT * FROM {monitor_info.drift_metrics_table_name}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Examine the Dashboard in Databricks SQL
# MAGIC
# MAGIC As mentioned earlier, Lakehouse Monitoring will generate Databricks SQL dashboards to review the data of a monitoring solution.
# MAGIC
# MAGIC We can see the link to the dashboard directly in the **Quality** tab of our primary table:
# MAGIC
# MAGIC This dashboard contains the following information for our monitor:
# MAGIC
# MAGIC * Primary name
# MAGIC * Overall summary statistics
# MAGIC * Time range filters
# MAGIC * Time-based metrics on:
# MAGIC   * Table size
# MAGIC   * Numeric/categorical profiles
# MAGIC   * Data integrity
# MAGIC   * Drift
# MAGIC
# MAGIC You can take a look at the dashboard below:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-online-monitoring-dashboard.png" width="100%">

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we demonstrated how to create an online monitor for a deployed AI model by observing updates to the inference table. First, we imported a sample inference table and performed data transformations to prepare the dataset for metric calculations. Next, we computed evaluation metrics. In the second part of the demo, we created a monitor on the processed inference table, showed how to refresh the metrics, query them, and view the metrics on the auto-created dashboard.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>