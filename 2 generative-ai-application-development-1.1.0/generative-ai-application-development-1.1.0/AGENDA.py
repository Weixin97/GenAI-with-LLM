# Databricks notebook source
#INCLUDE_HEADER_TRUE
#INCLUDE_FOOTER_TRUE

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Generative AI Application Development (Compound AI Systems)
# MAGIC
# MAGIC This course provides participants with information and practical experience in building advanced LLM (Large Language Model) applications using multi-stage reasoning LLM chains and agents. In the initial section, participants will learn how to decompose a problem into its components and select the most suitable model for each step to enhance business use cases. Following this, participants will construct a multi-stage reasoning chain utilizing LangChain and HuggingFace transformers. Finally, participants will be introduced to agents and will design an autonomous agent using generative models on Databricks.
# MAGIC
# MAGIC
# MAGIC
# MAGIC ## Course agenda
# MAGIC
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **[Foundations of Compound AI Systems]($./01 - Deconstruct and Plan a Use Case)**    | Defining Compound AI Systems </br> Designing Compound AI Systems <br>[Demo: Deconstruct and Plan a Use Case]($./01 - Deconstruct and Plan a Use Case/1.1 - Planning a Compound AI System Architecture) <br>[Lab: Planning an AI System for Product Quality Complaints]($./01 - Deconstruct and Plan a Use Case/1.LAB - Planning an AI System for Product Quality Complaints)|
# MAGIC | **[Building Multi-stage Reasoning Chains]($./02 - Building Multi-stage Reasoning)** | Introduction to Multi-stage Reasoning Chains </br> [Demo: Building Multi-stage Reasoning Chain in Databricks]($./02 - Building Multi-stage Reasoning/2.1 - Building Multi-stage AI Systems) </br> [Lab: Building Multi-stage AI System]($./02 - Building Multi-stage Reasoning/2.LAB - Building Multi-stage AI System) | 
# MAGIC | **[Agents and Cognitive Architectures]($./03 - Agent Design in Databricks)** | Introduction to Agents </br> Introduction to Mosaic AI Vector Search </br> [Demo: Agent Design in Databricks]($./03 - Agent Design in Databricks/3.1 - Agent Design in Databricks) </br> [Lab: Create a ReAct Agent]($./03 - Agent Design in Databricks/3.LAB - Create a ReAct Agent) | 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**