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
# MAGIC # LAB - Create a ReAct Agent
# MAGIC
# MAGIC In this lab we will create a virtual stock advisor agent using that uses the **ReAct** prompting. This agent will reason and act based on the prompts and tools provided. 
# MAGIC
# MAGIC Of course, the agent is build for demonstration purposes and it the answers shouldn't be used as investment advise! 
# MAGIC
# MAGIC
# MAGIC **Lab Outline:**
# MAGIC
# MAGIC In this lab, you will need to complete the following tasks;
# MAGIC
# MAGIC * **Task 1 :** Define the agent brain
# MAGIC
# MAGIC * **Task 2 :** Define the agent tools
# MAGIC
# MAGIC * **Task 3 :** Define an agent logic
# MAGIC
# MAGIC * **Task 4 :** Create the agent 
# MAGIC
# MAGIC * **Task 5 :** Run the agent
# MAGIC
# MAGIC **📝 Your task:** Complete the **`<FILL_IN>`** sections in the code blocks and follow the other steps as instructed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the lab, run the provided classroom setup scripts. These scripts will install the required libraries and configure variables which are needed for the lab.

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet langchain==0.1.16 langchain_community==0.0.36 yfinance==0.2.38 wikipedia==1.4.0 youtube-search 
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-LAB

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
# MAGIC ## Task 1: Define the Brain of the Agent
# MAGIC
# MAGIC Let's start by defining the agent brain using one of the available LLMs on Databricks.
# MAGIC
# MAGIC * For this lab, use **databricks-dbrx-instruct** 
# MAGIC
# MAGIC * Define the maximum number of tokens

# COMMAND ----------

from langchain.agents import AgentType, initialize_agent
from langchain_community.chat_models import ChatDatabricks

# define the brain
llm_dbrx = <FILL_IN>

# COMMAND ----------

# let's test the brain
llm_dbrx.invoke("Hi! How are you?")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 2 : Define Tools that the Agent Will Use
# MAGIC
# MAGIC For an agent the toolset is a crucial component to complete the task define. The agent will reason for the next action to take and an appropriate tool will be used to complete the action.
# MAGIC
# MAGIC For our stock advisor agent, we will use following tools;
# MAGIC
# MAGIC * **Wikipedia tool**: For searching company details. [[LangChain Doc](https://python.langchain.com/docs/integrations/tools/wikipedia/)]
# MAGIC
# MAGIC * **YahooFinance News tool**: This will be used for aggregating latest news about a stock. [[LangChain Doc](https://python.langchain.com/docs/integrations/tools/yahoo_finance_news/)]
# MAGIC
# MAGIC * **YouTube search tool**: This will be used to search for review videos about the latest financial reports. [[LangChain Doc](https://python.langchain.com/docs/integrations/tools/youtube/)]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wikipedia Tool

# COMMAND ----------

from langchain.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = <FILL_IN>
tool_wiki = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### YahooFinance News Tool

# COMMAND ----------

from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool

tool_finance = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### YouTube Search Tool

# COMMAND ----------

from langchain_community.tools import YouTubeSearchTool

tool_youtube = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Toolset
# MAGIC
# MAGIC Next, add all tools together to be used in the next step.

# COMMAND ----------

tools = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tasks 3: Define Agent Logic
# MAGIC
# MAGIC Agent needs a planning logic while doing reasoning and defining actions. Here you will use a common LangChain prompt template.

# COMMAND ----------

template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

# COMMAND ----------

from langchain.prompts import PromptTemplate

prompt = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 4: Create the Agent
# MAGIC
# MAGIC The final step is to create the agent using the LLM, planning prompt and toolset.

# COMMAND ----------

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

agent = <FILL_IN>
agent_exec  = <FILL_IN>

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 5: Run the Agent
# MAGIC
# MAGIC Let's test the agent!
# MAGIC
# MAGIC **Use this prompt:**
# MAGIC <blockquote>
# MAGIC What do you think about investing in Apple stock? Provide an overview of the latest news about the company. Next, search a video on YouTube which reviews Apple's latest financials. 
# MAGIC
# MAGIC Format the final answer as HTML.
# MAGIC </blockquote>
# MAGIC

# COMMAND ----------

response = <FILL_IN>
displayHTML(response["output"])

# COMMAND ----------

# MAGIC %md
# MAGIC It seems like the agent uses company name only when searching for videos. **Let's play with the prompt to get better results.**
# MAGIC
# MAGIC In real-world, the agent would be used in a conversational way and we would just chat with the agent to tweak the search query or to adjust the response in a desired format. 
# MAGIC
# MAGIC **Use this prompt:**
# MAGIC <blockquote>
# MAGIC What do you think about investing in Apple stock? Provide an overview of the latest news about the company. Next, search a video on YouTube which reviews Apple's latest financials.
# MAGIC
# MAGIC When searching on YouTube use such as "Apple  stock finance investment earnings valuation".
# MAGIC
# MAGIC Write a nice summary of the results.
# MAGIC
# MAGIC Format the final answer as HTML.
# MAGIC </blockquote>

# COMMAND ----------

response = <FILL_IN>
displayHTML(response["output"])

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
# MAGIC In this lab, you successfully created a virtual stock advisor agent using the ReAct prompting framework. By defining the agent's brain, tools, logic, and execution flow, you demonstrated how the agent can interact with various data sources to provide investment insights.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>