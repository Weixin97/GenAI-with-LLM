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
# MAGIC # Agent Design in Databricks
# MAGIC
# MAGIC In the previous demo, we build a multi-stage AI system by manually stitching them together. With Agents, we can build the same system in an autonomous way. An agent, typically, has a brain which make the decisions, a planning outline and tools to use. 
# MAGIC
# MAGIC In this demo, we will create two types of agents. The first agent will use **a search engine, Wikipedia, and Youtube** to recommend a movie, collect data about the movie and show the trailer video. 
# MAGIC
# MAGIC The second agent is a verys specific type agent; it will allow us to "talk with data" using natural language queries. 
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Build semi-automated systems with LLM agents to perform internet searches and dataset analysis using LangChain.
# MAGIC
# MAGIC * Use appropriate tool for the agent task to be achieved.
# MAGIC
# MAGIC * Explore LangChain’s built-in agents for specific, advanced workflows.
# MAGIC
# MAGIC * Create a Pandas DataFrame Agent to interact with a Pandas DataFrame as needed.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **14.3.x-cpu-ml-scala2.12 14.3.x-scala2.12**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install --upgrade --quiet langchain==0.1.16 langchain-core langchain_community==0.0.36 langchain-experimental youtube_search wikipedia==1.4.0 duckduckgo-search
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-04

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
# MAGIC ## Create an Autonomous Agent (Brixo 🤖)
# MAGIC
# MAGIC In the previous demo, we create chains using various prompts and tools combinations go solve a problem defined by the prompt. In chains, we need to define the input parameters and prompts. 
# MAGIC
# MAGIC In this demo, we will create an agent that can **autonomously reason** about the steps to take and select **the tools** to use for each task.
# MAGIC
# MAGIC **🤖 Agent name: Brixo :)**
# MAGIC
# MAGIC **✅ Agent Abilities: This agent can help you by suggesting fun activities, pick videos and even write code.**

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Brain of the Agent
# MAGIC
# MAGIC LLM is the brain of the agent. We will use **Databricks' DBRX model** as the brain of our agent.

# COMMAND ----------

from langchain_community.chat_models import ChatDatabricks

# play with max_tokens to define the length of the response
llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Tools that the Agent Can Use
# MAGIC
# MAGIC Agent can use various tools for completing a task. Here we will define the tools that can be used by **Brixo 🤖**.

# COMMAND ----------

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain_community.tools import YouTubeSearchTool

from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

from langchain_community.tools import DuckDuckGoSearchRun

# Wiki tool for info retrieval
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool_wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# tool to search youtube videos
tool_youtube = YouTubeSearchTool()

# web search tool
search = DuckDuckGoSearchRun()

# tool to write python code
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

# toolset
tools = [tool_wiki, tool_youtube, search, repl_tool]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define Planning Logic
# MAGIC
# MAGIC While working on tasks, our agent will need to done some reasoning and planning. We can define the format of this plan by passing a prompt.

# COMMAND ----------

from langchain.prompts import PromptTemplate

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

prompt= PromptTemplate.from_template(template)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the Agent
# MAGIC
# MAGIC The final step is to put all these together and build an agent.

# COMMAND ----------

from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent

agent = create_react_agent(llm_dbrx, tools, prompt)
brixo  = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
brixo.invoke({"input": 
    """What would be a nice movie to watch in rainy weather. Follow these steps.
    
    First, decide which movie you would recommend.

    Second, show me the trailler video of the movie that you suggest. 

    Next, collect data about the movie using search tool and  draw a bar chart using Python libraries. If you can't find latest data use some dummy data as we to show your abilities to the learners. Don't use ``` for python code. Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. The output must be the result of executed code.

    Finally, tell a funny joke about agents.
    """})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create an Autonomous Agent 2 (DataQio 🤖)
# MAGIC
# MAGIC In this section we will create a quite different agent; this agent will allow us to communicate with our **Pandas dataframe** using natural language.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prepare Dataset
# MAGIC
# MAGIC First, let's download a dataset from 🤗 and convert it to Pandas dataframe.

# COMMAND ----------

from datasets import load_dataset

dataset = load_dataset("maharshipandya/spotify-tracks-dataset")
df = dataset["train"].to_pandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Define the Brain and Tools
# MAGIC
# MAGIC Next we will define the model(brain) of our agent and define the toolset to use.

# COMMAND ----------

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_community.chat_models import ChatDatabricks

llm_dbrx = ChatDatabricks(endpoint="databricks-dbrx-instruct", max_tokens = 500)

prefix = """ Input should be sanitized by removing any leading or trailing backticks. if the input starts with ”python”, remove that word as well. Use the dataset provided. The output must start with a new line."""

dataqio = create_pandas_dataframe_agent(
    llm_dbrx,
    df,
    verbose=True,
    max_iterations=3,
    prefix=prefix,
    agent_executor_kwargs={
        "handle_parsing_errors": True
    }
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Talk with DataQio 🤖
# MAGIC
# MAGIC We are ready to talk with our agent to ask questions about the data.

# COMMAND ----------

dataqio.invoke("What is the artist name of most popular song based on popularity?")

# COMMAND ----------

dataqio.invoke("What is the total number of rows?")

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
# MAGIC In this demo, we explored agent design in Databricks, moving beyond manual system stitching to autonomous agent-based systems. Agents, equipped with decision-making branches, planning outlines, and tools, streamline the process. We created two types of agents: one utilizing a search engine, Wikipedia, and YouTube to recommend movies and another enabling natural language data queries. By leveraging LangChain's capabilities, participants learned to build semi-automated systems, choose appropriate tools, and utilize built-in agents for advanced workflows, including interacting with Pandas DataFrames.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Hepful Resources
# MAGIC
# MAGIC * **The Databricks Generative AI Cookbook ([https://ai-cookbook.io/](https://ai-cookbook.io/))**: Learning materials and production-ready code to take you from initial POC to high-quality production-ready application using Mosaic AI Agent Evaluation and Mosaic AI Agent Framework on the Databricks platform.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>