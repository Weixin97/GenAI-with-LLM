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
# MAGIC # Serving External Models with Model Serving
# MAGIC
# MAGIC **In this demo, we will focus on deploying GenAI applications.**
# MAGIC
# MAGIC Deployment is a key part of operationalizing our LLM-based applications. We will explore deployment options within Databricks and demonstrate how to achieve each one.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to:*
# MAGIC
# MAGIC * Determine the right deployment strategy for your use case.
# MAGIC * Deploy an external model to a Databricks Model Serving endpoint.
# MAGIC * Deploy a custom application to a Databricks Model Serving endpoint.

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
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Install required libraries.

# COMMAND ----------

# MAGIC %pip install -U --quiet databricks-sdk mlflow
# MAGIC
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Overview
# MAGIC
# MAGIC In this demo, we will walk through basic deployment capabilities in Databricks. We'll discuss this in the following steps:
# MAGIC
# MAGIC 1. Access to custom model in Databricks Marketplace.
# MAGIC
# MAGIC 1. Deploy an external model to a Databricks Model Serving endpoint
# MAGIC
# MAGIC 1. Deploy a custom application to a Databricks Model Serving endpoint

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy an External Model with Databricks Model Serving
# MAGIC
# MAGIC While we have described and used tools like the AI Playground and Foundation Model APIs for querying common LLMs, there is sometimes a need to deploy more specific models as part of our applications.
# MAGIC
# MAGIC To achieve this, we can use **Databricks Model Serving**. Databricks Model Serving is a production-ready, serverless solution that simplifies real-time (and other types of) ML model deployment.
# MAGIC
# MAGIC Next, we will demonstrate the basics of Model Serving.
# MAGIC
# MAGIC **🚨 Important: Deploying custom models requires Model Serving with provisioned throughput and consumes significant compute resources. Therefore, this demo is intended to be presented by the instructor only. If you are running this course in your own environment and have the necessary permissions to deploy models, feel free to follow these instructions.**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Getting an External Model
# MAGIC
# MAGIC The simplest way to deploy a model in Model Serving is by getting an existing external model from the **Databricks Marketplace**.
# MAGIC
# MAGIC Let's explore the Marketplace for the Databricks-provided **CodeLlama Models**:
# MAGIC
# MAGIC 1. Head to the **[Databricks Marketplace](/marketplace)**.
# MAGIC
# MAGIC 1. Filter to "Models" **products provided by "Databricks"**.
# MAGIC
# MAGIC 1. Click on the **"CodeLlama Models"** tile.
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-marketplace-llama-code.png" width="100%"/>
# MAGIC
# MAGIC
# MAGIC These models are designed to help with generating code – there are a series of fine-tuned versions.
# MAGIC
# MAGIC We are interested in deploying one of these models using Databricks Model Serving, so we'll need to follow the below steps:
# MAGIC
# MAGIC 1. Click on the **Get instant access** button on the models page
# MAGIC
# MAGIC 1. Specify our parameters, including that we want to use the model in Databricks and our `catalog name`.
# MAGIC
# MAGIC 1. Acknowledge the terms and conditions
# MAGIC
# MAGIC 1. Click **Get instant access**
# MAGIC
# MAGIC This will clone the models to the specified catalog. We can see them in the Catalog Explorer. Note that the catalog is created under **shared catalogs**.
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-catalog-llama-code-.png" width="100%">
# MAGIC
# MAGIC **Note:** An important point here is that these models are now stored in Unity Catalog. This means that they're secure and we can govern access to them using the familiar, general Unity Catalog tooling.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploying a Model using Model Serving
# MAGIC
# MAGIC Once these models are in our catalog, we can deploy them directly to Databricks Model Serving by following the below steps:
# MAGIC
# MAGIC 1. Navigate to a specific model page in the Catalog.
# MAGIC
# MAGIC 1. Click the **Serve this Model** button.
# MAGIC
# MAGIC 1. Configure the served entity.
# MAGIC     * Name: `CodeLlama_13b_Python_hf`.
# MAGIC     * For served entities, select the model.
# MAGIC
# MAGIC 1. Click the **Confirm** button.
# MAGIC
# MAGIC 1. Configure the Model Serving endpoint.
# MAGIC
# MAGIC 1. Click the **Create** button.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confirming the Deployed Model
# MAGIC
# MAGIC When the Model Serving endpoint is operational, we'll see a screen like this:
# MAGIC
# MAGIC <br>
# MAGIC
# MAGIC <img src="https://s3.us-west-2.amazonaws.com/files.training.databricks.com/images/genai/genai-as-04-serving-llama-endpoint.png" width="100%">
# MAGIC
# MAGIC **Note:** Notice the "Serving Deployment Status" field on the page. This will say "Not ready" until the model is deployed.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Query the Deployed Model
# MAGIC
# MAGIC More realistically, we can query the deployed model directly from our serving applications.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query via the UI
# MAGIC
# MAGIC We can query the model directly in Databricks to confirm everything is working using the **Query endpoint** capability.
# MAGIC
# MAGIC Sample query:
# MAGIC `{"prompt": "from spark.sql import functions as"}`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query the Deployed Model in AI Playground
# MAGIC
# MAGIC To test the model with AI Playground, select the deployed model and use chatbox to send queries.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query the Deployed Model with the SDK
# MAGIC
# MAGIC **💡 Tip:** Change the number of `max_tokens` to change the length of suggested code completion.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

dataframe_records = [
    {"prompt": "from spark.sql import functions as", "max_tokens": 3}
]

w = WorkspaceClient()
response = w.serving_endpoints.query(
    name="CodeLlama_13b_Python_hf", #name of the model serving endpoint
    dataframe_records=dataframe_records,
)

print(response.as_dict()["predictions"][0]["candidates"][0]["text"])

# COMMAND ----------

# MAGIC %md
# MAGIC This workflow is similar for external models like CodeLlama and **any other model that's in Unity Catalog**. In the next demo, we will deploy a custom model (RAG pipeline) using Model Serving.
# MAGIC
# MAGIC **Question:** What do you think of the results of the query? How could we improve the application to return better results?

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC At this point, you should be able to:
# MAGIC
# MAGIC * Determine the right deployment strategy for your use case.
# MAGIC * Deploy an external model to a Databricks Model Serving endpoint.
# MAGIC * Deploy a custom application to a Databricks Model Serving endpoint.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>