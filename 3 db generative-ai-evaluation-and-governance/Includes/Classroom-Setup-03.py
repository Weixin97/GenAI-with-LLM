# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic init, schemas and catalogs
spark.sql(f"USE CATALOG {DA.catalog_name}")
spark.sql(f"USE SCHEMA {DA.schema_name}")

# COMMAND ----------

from databricks.sdk.service.serving import ChatMessage
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

# COMMAND ----------

def query_chatbot_system(input: str) -> str:
   
    messages = [
        {
            "role": "system",
            "content": "You are a chatbot that specializes in assisting with Databricks-related questions. You should remain impartial when providing assistance and answering questions."
        },
        { 
            "role": "user", 
            "content": input 
        }
    ]
    messages = [ChatMessage.from_dict(message) for message in messages]
    chat_response = w.serving_endpoints.query(
        name="databricks-llama-2-70b-chat",
        messages=messages,
        max_tokens=128
    )

    return chat_response.as_dict()["choices"][0]["message"]["content"]

# COMMAND ----------

def query_product_summary_system(input: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that summarizes text. Given a text input, you need to provide a one-sentence summary. You specialize in summiarizing reviews of grocery products. Please keep the reviews in first-person perspective if they're originally written in first person. Do not change the sentiment. Do not create a run-on sentence – be concise."
        },
        { 
            "role": "user", 
            "content": input 
        }
    ]
    messages = [ChatMessage.from_dict(message) for message in messages]
    chat_response = w.serving_endpoints.query(
        name="databricks-llama-2-70b-chat",
        messages=messages,
        max_tokens=128
    )

    return chat_response.as_dict()["choices"][0]["message"]["content"]

# COMMAND ----------

DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

print("\nThe examples and models presented in this course are intended solely for demonstration and educational purposes.\n Please note that the models and prompt examples may sometimes contain offensive, inaccurate, biased, or harmful content.")