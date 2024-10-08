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
# MAGIC # Batch Inference Workflow
# MAGIC
# MAGIC In this example, we will walk through some key steps for implementing an LLM-based pipeline using a **Small Language Model (SLM)** for batch inference in a production environment.
# MAGIC
# MAGIC **Notes about this workflow:**
# MAGIC
# MAGIC **📌 This notebook vs. modular scripts**: Since this demo is contained within a single notebook, we will divide the workflow from development to production into notebook sections. In a more realistic LLM Ops setup, these sections would likely be split into separate notebooks or scripts.
# MAGIC
# MAGIC **📌 Promoting models vs. code**: We track the path from development to production via the Model Registry. That is, we are *promoting models* towards production, rather than promoting code.
# MAGIC
# MAGIC ## Learning Objectives
# MAGIC
# MAGIC By the end of this demo, you should be able to:
# MAGIC
# MAGIC 1. Load a model from the Model Registry for batch inference.
# MAGIC
# MAGIC 1. Manage model aliases and retrieve the latest version of the model.
# MAGIC
# MAGIC 1. Apply batch inference on a Spark DataFrame using single-node batch inference.
# MAGIC
# MAGIC 1. Apply multinode batch inference using `spark_udf`.
# MAGIC
# MAGIC 1. Explain other methods of batch inference.

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
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install --quiet datasets mlflow==2.12.1 transformers==4.41.2
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC Before starting the demo, run the provided classroom setup script.

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

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
# MAGIC 1. Prepare dataset.
# MAGIC 1. Develop a Huggingface/transformer LLM pipeline.
# MAGIC 1. Apply/test pipeline to data, and log results to MLflow Tracking.
# MAGIC 1. Log the pipeline to the MLflow Tracking server as an MLflow Model.
# MAGIC 1. Load LLM pipeline from registry and run batch inference
# MAGIC 1. Use SQL `ai_query()` for batch inference on existing/supported _Foundation Models API_ models

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data and Model Preparation 
# MAGIC
# MAGIC In this section, we will create a dataset and the model that we will be using for the rest of the demo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Dataset
# MAGIC
# MAGIC Prepare a Delta table containing texts to summarize from the [Extreme Summarization (XSum) Dataset](https://huggingface.co/datasets/xsum), which we will use to run batch inferences.

# COMMAND ----------

from datasets import load_dataset
from transformers import pipeline
from delta.tables import DeltaTable

prod_data_table_name = f"{DA.catalog_name}.{DA.schema_name}.m4_1_prod_data"

xsum_dataset = load_dataset(
    "xsum",
    version="1.2.0"
)

# Save test set to delta table
test_spark_df = spark.createDataFrame(xsum_dataset["test"].to_pandas())
test_spark_df.write.mode("overwrite").saveAsTable(prod_data_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a Hugging Face Pipeline
# MAGIC
# MAGIC For this notebook we'll use the <a href="https://huggingface.co/t5-small" target="_blank">T5 Text-To-Text Transfer Transformer</a> from Hugging Face.

# COMMAND ----------

from transformers import pipeline

# Define pipeline inference parameters - to be logged in mlflow as part of model _metadata
hf_model_name = "t5-small"
min_length = 20
max_length = 40
truncation = True
do_sample = True
device_map = "auto" # 'cuda', 'cpu'

cache_dir =DA.paths.datasets.replace("dbfs:/", "/dbfs")

summarizer = pipeline(
    task="summarization",
    model=hf_model_name,
    min_length=min_length,
    max_length=max_length,
    truncation=truncation,
    do_sample=do_sample,
    device_map=device_map,
    model_kwargs={"cache_dir": cache_dir},
)  # Note: We specify cache_dir to use pre-cached models.

# COMMAND ----------

# MAGIC %md
# MAGIC We can now examine the `summarizer` pipeline summarizing some text

# COMMAND ----------

text_to_summarize= """ Barrington DeVaughn Hendricks (born October 22, 1989), known professionally as JPEGMafia (stylized in all caps), is an American rapper, singer, and record producer born in New York City and based in Baltimore, Maryland. His 2018 album Veteran, released through Deathbomb Arc, received widespread critical acclaim and was featured on many year-end lists. It was followed by 2019's All My Heroes Are Cornballs and 2021's LP!, released to further critical acclaim. """

summarized_text = summarizer(text_to_summarize)[0]['summary_text']
print(f"Summary:\n {summarized_text}")
print("===============================================")
print(f"Original Document: {text_to_summarize}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Development and Registering

# COMMAND ----------

# MAGIC %md
# MAGIC ### Track LLM Development with MLflow
# MAGIC
# MAGIC Before we start the model development, here is a quick refresher for MLflow tracking.
# MAGIC
# MAGIC [MLflow](https://mlflow.org/) Tracking helps track model or pipeline production during development. Even without fitting a model, you can use it to track example queries and responses to the LLM pipeline and store the model as an [MLflow Model flavor](https://mlflow.org/docs/latest/models.html#built-in-model-flavors) for simpler deployment. 
# MAGIC
# MAGIC MLflow Tracking is organized hierarchically: an [experiment](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments) corresponds to creating a primary model or pipeline, containing multiple [runs](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments). Each run logs parameters, metrics, tags, models, artifacts, and other metadata. Parameters are inputs like `max_length`, metrics are evaluation outputs like accuracy, and artifacts are files like the serialized model. A [flavor](https://mlflow.org/docs/latest/models.html#storage-format) is an MLflow format for serializing models using the underlying ML library's format plus metadata. For more information, see the [LLM Tracking page](https://mlflow.org/docs/latest/llms/llm-tracking/index.html). Tip: Wrap your model development workflow with `with mlflow.start_run():` to start and end the MLflow run explicitly, a best practice for production code. See the [API doc](https://mlflow.org/docs/latest/python_api/mlflow.html#mlflow.start_run) for more details.

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from mlflow.transformers import generate_signature_output


# It is valuable to log a "signature" with the model telling MLflow the input and output schema for the model.
output = generate_signature_output(summarizer, text_to_summarize)
signature = infer_signature(text_to_summarize, output)
print(f"Signature:\n{signature}\n")


# Set experiment path
# (located on the left hand sidebar under Machine Learning -> Experiments)
experiment_name = f"/Users/{DA.username}/GenAI-As-04-Batch-Demo"
mlflow.set_experiment(experiment_name)
model_artifact_path = "summarizer" # Name of folder containing serialized model

with mlflow.start_run():
    # LOG PARAMS
    mlflow.log_params(
        {
            "hf_model_name": hf_model_name,
            "min_length": min_length,
            "max_length": max_length,
            "truncation": truncation,
            "do_sample": do_sample,
        }
    )

    # ---------
    # LOG MODEL
    # We next log our LLM pipeline as an MLflow model.
    # This packages the model with useful metadata, such as the library versions used to create it.
    # This metadata makes it much easier to deploy the model downstream.
    # Under the hood, the model format is simply the ML library's native format (Hugging Face for us), plus metadata.

    # For mlflow.transformers, if there are inference-time configurations,
    # those need to be saved specially in the log_model call (below).
    # This ensures that the pipeline will use these same configurations when re-loaded.
    inference_config = {
        "min_length": min_length,
        "max_length": max_length,
        "truncation": truncation,
        "do_sample": do_sample,
    }

    # Logging a model returns a handle `model_info` to the model metadata in the tracking server.
    # This `model_info` will be useful later in the notebook to retrieve the logged model.
    model_info = mlflow.transformers.log_model(
        transformers_model=summarizer,
        artifact_path=model_artifact_path,
        task="summarization",
        inference_config=inference_config,
        signature=signature,
        input_example="This is an example of a long news article which this pipeline can summarize for you.",
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query the MLflow Tracking server
# MAGIC
# MAGIC **MLflow Tracking API**: We briefly show how to query the logged model and metadata in the MLflow Tracking server, by loading the logged model.  See the [MLflow API](https://mlflow.org/docs/latest/python_api/mlflow.html) for more information about programmatic access.
# MAGIC
# MAGIC **MLflow Tracking UI**: You can also use the UI.  In the right-hand sidebar, click the [MLflow experiments](/#ml/experiments) to view the run list, and then click through to access the Tracking server UI.  There, you can see the logged metadata and model.  Note in particular that our LLM inputs and outputs have been logged as a CSV file under model artifacts.
# MAGIC
# MAGIC GIF of MLflow UI:
# MAGIC ![GIF of MLflow UI](https://files.training.databricks.com/images/llm/llmops.gif)

# COMMAND ----------

# Grab most recent run (which logged the model) using our experiment ID
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
runs = mlflow.search_runs([experiment_id])
last_run_id = runs.sort_values('start_time', ascending=False).iloc[0].run_id

# Construct model URI based on run_id
model_uri = f"runs:/{last_run_id}/{model_artifact_path}"

# COMMAND ----------

model_uri

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load Model Back as a Pipeline
# MAGIC
# MAGIC Now, we can load the pipeline back from MLflow as a [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) and use the `.predict()` method to summarize an example document.

# COMMAND ----------

loaded_summarizer = mlflow.pyfunc.load_model(model_uri=model_uri)
loaded_summarizer.predict(text_to_summarize)

# COMMAND ----------

# MAGIC %md
# MAGIC **Note :** The `.predict()` method can handle more than one document at a time (like a `pd.Series()` or `list()`)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Register the Model to Unity Catalog
# MAGIC
# MAGIC Register the pipeline to Unity-Catalog's Model Registry and set a model alias to label model as ready for staging/QA for example
# MAGIC
# MAGIC We track our progress here using **Unity Catalog's Model Registry** [AWS](https://docs.databricks.com/en/machine-learning/manage-model-lifecycle/index.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/machine-learning/manage-model-lifecycle/) | [GCP](https://docs.gcp.databricks.com/en/machine-learning/manage-model-lifecycle/index.html). 
# MAGIC This metadata and model store organizes models as follows:
# MAGIC * **A registered model** is a named model in the registry (respecting 3-level namespace convention ***`catalog.schema.model_name`***), in our case corresponding to our summarization model.  It may have multiple *versions*.
# MAGIC    * **A model version** is an instance of a given model.  As you update your model, you will create new versions.  Each version could be designated as being in a particular stage of deployment via:
# MAGIC       * **An `@alias`** is a unique - free text- alias describing which stage of deployment (e.g. `challenger` (development), `champion` (production), `baseline` or `archived`).
# MAGIC
# MAGIC The model we registered above starts with 1 version and no @alias.
# MAGIC
# MAGIC In the workflow below, we will programmatically modify/set the `@alias` for given model versions in order to mark their stage.  For more information on the Model Registry API, see the [Model Registry docs](https://mlflow.org/docs/latest/model-registry.html).  Alternatively, you can edit the registry and set model @aliases via the UI.

# COMMAND ----------

from mlflow import MlflowClient


# Define model name in the Model Registry
model_name = f"{DA.catalog_name}.{DA.schema_name}.summarizer"

# Point to Unity-Catalog registry and log/push artifact
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(
    model_uri=model_uri,
    name=model_name,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Manage Model Stage
# MAGIC
# MAGIC Set latest model version as `@champion`

# COMMAND ----------

def get_latest_model_version(model_name_in):
    """
    Helper method to programmatically get latest model's version from the registry
    """
    client = MlflowClient()
    model_version_infos = client.search_model_versions("name = '%s'" % model_name_in)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# Set @alias
client = mlflow.tracking.MlflowClient()
current_model_version = get_latest_model_version(model_name)

client.set_registered_model_alias(
  name=model_name, alias="champion",
  version=current_model_version
  )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create a Production Workflow for Batch Inference
# MAGIC
# MAGIC In production the goals are (a) to write scale-out code which can meet scaling demands in the future and (b) to simplify deployment by using MLflow to write model-agnostic deployment code.  Step-by-step, we will:
# MAGIC * Load the latest production LLM pipeline from the Model Registry.
# MAGIC * Apply the pipeline to an Apache Spark DataFrame.
# MAGIC * Append the results to a Delta Lake table.
# MAGIC
# MAGIC Here, we will show batch inference using Apache Spark DataFrames, with Delta Lake format.  Spark allows simple scale-out inference for high-throughput, low-cost jobs, and Delta allows us to append to and modify inference result tables with ACID transactions.  See the [Apache Spark page](https://spark.apache.org/) and the [Delta Lake page](https://delta.io/) more more information on these technologies.
# MAGIC
# MAGIC *Model URIs*: Below, we use model URIs to tell MLflow which model and version we are referring to.  Two common URI patterns for the MLflow Model Registry are:
# MAGIC * `f"models:/{model_name}/{model_version}"` to refer to a specific model version by number
# MAGIC * `f"models:/{model_name}@{alias}"` to refer to the model version given unique @alias

# COMMAND ----------

# MAGIC %md
# MAGIC Before we start, let's load the input texts to summarize into a spark dataframe

# COMMAND ----------

prod_data_table = f"{DA.catalog_name}.{DA.schema_name}.m4_1_prod_data"
prod_data_df = spark.read.table(prod_data_table).limit(10)
display(prod_data_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Single-node Batch Inference
# MAGIC
# MAGIC For single-node batch inference, the native `.predict()` method can be used

# COMMAND ----------

latest_model = mlflow.pyfunc.load_model(
  model_uri=f"models:/{model_name}/{current_model_version}"
)
latest_model

# COMMAND ----------

from pprint import pprint


prod_data_sample_pdf = prod_data_df.limit(2).toPandas()
summaries_sample = latest_model.predict(prod_data_sample_pdf["document"])
[pprint(s+'\n') for s in summaries_sample]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Multinode Batch Inference
# MAGIC Below, we load the model using `mlflow.pyfunc.spark_udf`.  This returns the model as a Spark User Defined Function which can be applied efficiently to big data.  *Note that the deployment code is library-agnostic: it never references that the model is a Hugging Face pipeline.*  This simplified deployment is possible because MLflow logs environment metadata and "knows" how to load the model and run it.

# COMMAND ----------

# Grab `Champion` model (supposed to belatest Production version)
prod_model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri=f"models:/{model_name}@champion",
    env_manager="local",
    result_type="string",
)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC When you load the model via MLflow above, you may see warnings about the Python environment.  It is very important to ensure that the environments for development, staging, and production match.
# MAGIC * For this demo notebook, everything is done within the same notebook environment, so we do not need to worry about libraries and versions.  However, in the Production the `env_manager` argument should be passed to the method while loading the saved MLflow model, to indicate what tooling to use to recreate the environment.
# MAGIC * To create a genuine production job, make sure to install the needed libraries.  MLflow saves these libraries and versions alongside the logged model; see the [MLflow docs on model storage](https://mlflow.org/docs/latest/models.html#storage-format) for more information.  While using Databricks for this course, you can also generate an example inference notebook which includes code for setting up the environment; see [the model inference docs](https://docs.databricks.com/machine-learning/manage-model-lifecycle/index.html#use-model-for-inference) for batch or streaming inference for more information.

# COMMAND ----------

# Run inference by appending a new column to the DataFrame

batch_inference_results_df = prod_data_df.withColumn("generated_summary", prod_model_udf("document"))
display(batch_inference_results_df)

# COMMAND ----------

prod_data_summaries_table_name = f"{DA.catalog_name}.{DA.schema_name}.m4_1_batch_inference"
batch_inference_results_df.write.mode("append").saveAsTable(prod_data_summaries_table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference Using `ai_query()`
# MAGIC
# MAGIC Another alternative & common method to run "batch-like" jobs using LLMs made available via the databricks foundation models API is to use the `ai_query()` **SQL** function [AWS](https://docs.databricks.com/en/sql/language-manual/functions/ai_query.html) | [Azure](https://learn.microsoft.com/en-us/azure/databricks/sql/language-manual/functions/ai_query)

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE ai_query_inference AS (
# MAGIC   SELECT
# MAGIC   id
# MAGIC   ,ai_query(
# MAGIC     "databricks-dbrx-instruct",
# MAGIC     CONCAT("Based on the following document, provide a summary in less than 100 words. Document: ", document)
# MAGIC   ) as generated_summary
# MAGIC  FROM m4_1_prod_data LIMIT 10
# MAGIC )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM ai_query_inference

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.
# MAGIC

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC In this demo, we developed a batch inference workflow using a small language model. First, we created a pipeline to summarize text. Then, we developed a model and registered it with Unity Catalog. We demonstrated how to track the model development process, query the model from the tracking server, and load the model back as a pipeline. We also showed how to manage model lifecycle with @aliases and concluded by showcasing two methods of batch inference: single-node and multi-node batch inference. Finally, we showed how to use `ai_query` for batch inference.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>