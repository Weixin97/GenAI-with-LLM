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
# MAGIC
# MAGIC
# MAGIC ## Generative AI Evaluation and Governance
# MAGIC
# MAGIC This course introduces learners to evaluating and governing generative artificial intelligence (AI) systems. First, learners will explore the meaning behind and motivation of building evaluation and governance/security systems. Next, learners will be introduced to a variety of evaluation techniques for LLMs and their tasks. Next, the course will discuss how to govern and secure AI systems using Databricks. And finally, the course will conclude with an analysis of evaluating entire AI systems with respect to performance and cost.
# MAGIC
# MAGIC
# MAGIC ## Course Agenda
# MAGIC | Module &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; | Lessons &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
# MAGIC |:----:|-------|
# MAGIC | **[The Importance of Evaluating GenAI Applications)]($./01 - Data Legality and Guardrails)**    | Why Evaluating GenAI Applications <br>[Demo: Exploring Licensing of Datasets]($./01 - Data Legality and Guardrails/1.1 - Explore Licensing of Datasets Demo) <br>[Demo: Prompts and Guardrails Basics]($./01 - Data Legality and Guardrails/1.2 - Prompts and Guardrails Basics Demo) <br> [Lab: Implementing and Testing Guardrails for LLMs]($./01 - Data Legality and Guardrails/1.LAB - Implement and Test Guardrails for LLMs)|
# MAGIC | **[Securing and Governing GenAI Applications]($./02 - Securing and Governing AI Systems)** | AI System Security </br> [Demo: Implementing AI Guardrails]($./02 - Securing and Governing AI Systems/2.1 - Implementing AI Guardrails) </br> [Lab: Implementing AI Guardrails]($./02 - Securing and Governing AI Systems/2.LAB - Implementing AI Guardrails) | 
# MAGIC | **[Gen AI Evaluation Techniques]($./03 - Evaluating LLMs)** | Evaluation Techniques </br> [Demo: Benchmark Evaluation]($./03 - Evaluating LLMs/3.1 - Benchmark Evaluation) </br> [Demo: LLM-as-a-Judge]($./03 - Evaluating LLMs/3.2 - LLM-as-a-Judge) </br> [Lab: Domain-Specific Evaluation]($./03 - Evaluating LLMs/3.LAB - Domain-Specific Evaluation)|

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run demo and lab notebooks, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>