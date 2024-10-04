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
# MAGIC # Exploring Licensing of Datasets
# MAGIC
# MAGIC
# MAGIC **In this demo, we will focus on reviewing the licenses associated with data we might want to use for our model.** To do this, we will explore the Databricks Marketplace. There, we will review a specific dataset that we might want to use for one of our models and find its licensing information.
# MAGIC
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Recognize potential legal concerns over usage of datasets for your AI applications.
# MAGIC
# MAGIC * Find datasets for your application in Databricks Marketplace.
# MAGIC
# MAGIC * Assess with your legal counsel whether the license for a given dataset would be permit your intended usage.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To walk through this demo, you simply need a Databricks environment with access to the **[Databricks Marketplace](/marketplace)**.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Overview
# MAGIC
# MAGIC The power of modern GenAI applications exists in their ability to become intimately familiar with your enterprise's data. However, there are also use cases where we might want to augment our training data or contextual knowledge data with general industry data. We can find supplementary data using the Databricks Marketplace.
# MAGIC
# MAGIC In this demo, we will explore a few datasets on the Databricks Marketplace, with a particular focus on licensing issues and determining whether they'd be a good candidate for using in a particular type of GenAI application.
# MAGIC
# MAGIC Here are all the detailed steps:
# MAGIC
# MAGIC - Get familiar with Databricks Marketplace and how to find it.
# MAGIC - Look at the license for an example dataset that would be OK for our application.
# MAGIC - Explore another example dataset that might not be OK for us to use.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Marketplace
# MAGIC
# MAGIC The Databricks Marketplace allows users to discover, explore, and access data and AI resources from a variety of providers. It offers resources such as datasets you can use to power RAG applications or add as features or attributes to your existing pipelines, pre-trained machine learning models, and solution accelerators to help you get from concept to production more quickly and efficiently.
# MAGIC
# MAGIC ![marketplace_home](files/images/generative-ai-evaluation-and-governance-1.0.2/marketplace_home.png)
# MAGIC
# MAGIC In this demo, we'll be focused on datasets we can use for our RAG application.
# MAGIC
# MAGIC There are **two main reasons** we might want to find additional data for our RAG application:
# MAGIC
# MAGIC 1. We might want additional reference data to augment our prompt as context that we can query at runtime using Vector Search against a vector index.
# MAGIC 2. In some scenarios, we may need additional data to use for fine-tuning or further pretraining the LLM powering our overall RAG application – note that this can also be applied outside of RAG applications.
# MAGIC
# MAGIC ### Why do we want third-party data?
# MAGIC
# MAGIC It's true that a lot of your application's contextual data is likely to come from within your company. However, there might be times when information provided by third-party providers can be useful as well. We need to make sure that we're allowed to use that data.
# MAGIC
# MAGIC As practitioners, it is critical to be aware of potential licensing restrictions that these datasets may ship with that could prevent you from using that particular dataset for your intended purpose. Every dataset on the Databricks Marketplace should have a license that you can review to determine (along with your appropriate legal counsel) whether it is usable or not.
# MAGIC
# MAGIC **Note:** While the focus here is on Databricks Marketplace, the same ideas are equally applicable to data you find anywhere else on the interface, including other data hubs like the one hosted and operated by Hugging Face. Regardless of where you find data for your application, be sure you know the licensing and legal implications.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bright Data Amazon Reviews Dataset
# MAGIC
# MAGIC In the first example, we'll look at an Amazon Reviews dataset provided by Bright Data. 
# MAGIC
# MAGIC To locate the dataset, follow the below steps:
# MAGIC
# MAGIC 1. Navigate to the Databricks Marketplace
# MAGIC 2. Filter the **Product** filter to "Tables"
# MAGIC 3. Filter the **Provider** filter to "Bright Data"
# MAGIC 4. Open the "Amazon - Best Seller Products /Reviews/Products Dataset"
# MAGIC
# MAGIC ![bright_data_amazon_reviews_dataset_overview](files/images/generative-ai-evaluation-and-governance-1.0.2/bright_data_amazon_reviews_dataset_overview.png)
# MAGIC
# MAGIC Let's explore the data to see if it could work for our use case:
# MAGIC
# MAGIC * What is the record-level of the dataset?
# MAGIC * What fields does it include?
# MAGIC * How many records does it have?
# MAGIC * How often is the data updated?
# MAGIC
# MAGIC Even if the dataset meets our requirements, we need to review its **license** to determine whether or not we're allowed to use it.
# MAGIC
# MAGIC To find the license information:
# MAGIC
# MAGIC 1. Take a look at the **Product links** section under the provider info
# MAGIC 2. Click on **License** to open the license information
# MAGIC 3. Review the agreement for something like "Acceptable Use"
# MAGIC
# MAGIC You'll need to carefully review the license to determine whether it permits your intended purpose with your application. It is highly recommended to take advantage of legal counsel in making any licensing determination.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rearc Personal Inform | FRED
# MAGIC
# MAGIC In the next example, we'll look at a Personal Income dataset from Rearc.
# MAGIC
# MAGIC ![rearc_personal_income_fred_overview](files/images/generative-ai-evaluation-and-governance-1.0.2/rearc_personal_income_fred_overview.png)
# MAGIC
# MAGIC Let's get started with a few questions:
# MAGIC
# MAGIC * How would this data be useful for a GenAI application?
# MAGIC * Where is the license information?
# MAGIC * Are we able to use this data for a hypothetical commercial application?
# MAGIC
# MAGIC The same advice applies here. Carefully review the license, ideally with the help of appropriate legal counsel before integrating the data into your application or using it to train or fine-tune your models.
# MAGIC
# MAGIC ### Ingesting the Data
# MAGIC
# MAGIC Let's assume we have legal permission to use the data for our use case.
# MAGIC
# MAGIC We can ingest it into our environment directly from Databricks Marketplace by clicking the **Get instant access** button on the dataset page in Marketplace.
# MAGIC
# MAGIC This will prompt us to:
# MAGIC
# MAGIC * determine the name of the catalog in which the data will reside (new or existing).
# MAGIC * accept the terms and conditions.
# MAGIC
# MAGIC And then we'll be able to open, explore, and use our data!

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC Before you use any external dataset for your application, you need to review the licensing information under which it is provided. Data made available through the Databricks Marketplace should always have a place to find the license information, which we recommend you review yourself as a practitioner as well as with the advice of appropriate legal counsel. Other popular hubs such as Hugging Face should also have license information available for each dataset.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>