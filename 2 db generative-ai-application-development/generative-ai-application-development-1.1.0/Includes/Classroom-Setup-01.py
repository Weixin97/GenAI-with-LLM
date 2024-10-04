# Databricks notebook source
def get_multistage_html():
    import re
    from graphviz import Digraph

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'LR'
    dot.graph_attr['splines'] = 'ortho'
    dot.edge_attr.update(arrowhead='normal', arrowsize='1')
    dot.attr('node', shape='rectangle')

    def component_link(component,
                       ttip=''):
        url = "https://curriculum-dev.cloud.databricks.com"
        path = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-1])
        path = path.replace(" ","%20")
        return {'tooltip': ttip, 'href': '', 'target': "_blank",
                'width': "1.5"}

    with dot.subgraph(name='cluster_workflow') as w:
        w.body.append('label="Model Serving"')
        w.body.append('style="filled"')
        w.body.append('color="#808080"')
        w.body.append('fillcolor="#F5F5F5"')

        w.node('question', 'question', fillcolor='#FFD580', style='filled', shape='oval',
                   **component_link('question'))
        with w.subgraph(name='cluster_app') as a:
            a.body.append('label="Compound_rag_app Class"')
            a.body.append('style="filled"')
            a.body.append('color="#808080"')
            a.body.append('fillcolor="#DCDCDC"')


            with a.subgraph(name='cluster_main') as m:
                m.body.append('label="MAIN_HEADER"')
                m.body.append('style="filled"')
                m.body.append('color="#808080"')
                m.body.append('fillcolor="#00BFFF"')

                m.node('run_search', 'run_search()', fillcolor='#87CEFA', style='filled',
                           **component_link('run_search'))

                with m.subgraph(name='cluster_run_augment') as ra:
                    ra.body.append('label="RUN_AUGMENT_HEADER"')
                    ra.body.append('style="filled"')
                    ra.body.append('color="#808080"')
                    ra.body.append('fillcolor="#87CEFA"')

                    ra.node('run_summary', 'run_summary()', fillcolor='#CAD9EF', style='filled',
                           **component_link('run_summary'))
                    
                m.node('run_get_context', 'run_get_context()', fillcolor='#87CEFA', style='filled',
                           **component_link('run_get_context'))
                m.node('run_qa', 'run_qa()', fillcolor='#87CEFA', style='filled',
                           **component_link('run_get_context'))
    
        w.node('answer', 'answer', fillcolor='#FFD580', style='filled', shape='oval',
                       **component_link('run_get_context'))

        dot.edge('question', 'run_search')
        dot.edge('run_search', 'run_summary')
        dot.edge('run_summary', 'run_get_context')  
        dot.edge('run_get_context', 'run_qa')
        dot.edge('run_qa', 'answer')

    with dot.subgraph(name='cluster_vectorsearch') as v:
        v.attr(labelloc="b")
        v.body.append('label="Data Serving"')
        v.body.append('style="filled"')
        v.body.append('color="#808080"')
        v.body.append('fillcolor="#F5F5F5"')
        v.node('vector_search', 'Databricks\nVectorSearch', fillcolor="#DCDCDC", style='filled', **component_link('run_search'))

    dot.node('hidden_search_offset', 'hso', style="invis")
    dot.edge('hidden_search_offset','run_search', style="invis")
    dot.edge('hidden_search_offset','vector_search', style="invis")
    dot.edge('vector_search', 'run_search')
    dot.edge('run_search', 'vector_search')

    with dot.subgraph(name='cluster_summary') as s:
        s.attr(labelloc="b")
        s.body.append('label="Model Serving"')
        s.body.append('style="filled"')
        s.body.append('color="#808080"')
        s.body.append('fillcolor="#F5F5F5"')
        s.node('summary_llm', 'Summary LLM', fillcolor="#DCDCDC", style='filled', **component_link('run_search'))

    dot.edge('vector_search', 'summary_llm', style="invis")
    dot.edge('run_summary', 'summary_llm')
    dot.edge('summary_llm', 'run_summary')

    with dot.subgraph(name='cluster_qa') as q:
        q.attr(labelloc="b")
        q.body.append('label="Model Serving"')
        q.body.append('style="filled"')
        q.body.append('color="#808080"')
        q.body.append('fillcolor="#F5F5F5"')
        q.node('qa_llm', 'QA LLM', fillcolor="#DCDCDC", style='filled', **component_link('run_search'))

    
    dot.edge('run_get_context', 'qa_llm', style="invis")
    dot.edge('qa_llm', 'run_qa')
    dot.edge('run_qa', 'qa_llm')



    html = dot._repr_image_svg_xml()

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="1000pt" aligned=center', html)

    #html = re.sub(r'MAIN_HEADER', f'<a href="{component_link("main")["href"]}" target=\"_blank\">main()</a>', html)
    html = re.sub(r'MAIN_HEADER', 'main()', html)

    #html = re.sub(r'RUN_AUGMENT_HEADER', f'<a href="{component_link("run_augment")["href"]}" target=\"_blank\">run_augment()</a>', html)
    html = re.sub(r'RUN_AUGMENT_HEADER', 'run_augment()', html)        

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html

# COMMAND ----------

def get_stage_html(stage=''):
    import re
    from graphviz import Digraph

    dot = Digraph('pt')
    dot.attr(compound='true')
    dot.graph_attr['rankdir'] = 'LR'
    dot.graph_attr['splines'] = 'ortho'
    dot.edge_attr.update(arrowhead='normal', arrowsize='1')
    dot.attr('node', shape='rectangle')

    def component_link(component,
                       ttip=''):
        url = "https://curriculum-dev.cloud.databricks.com"
        path = "/".join(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get().split("/")[:-1])
        path = path.replace(" ","%20")
        return {'tooltip': ttip, 'href': '', 'target': "_blank", 'width': "1.5"}

    dot.node('question', 'question', fillcolor='#FFD580', style='filled', shape='oval', **component_link('question'))

    with dot.subgraph(name='cluster_main') as m:
        m.body.append('label="MAIN_HEADER"')
        m.body.append('style="filled"')
        m.body.append('color="#808080"')
        m.body.append('fillcolor="yellow"' if stage=='main' else 'fillcolor="#00BFFF"')

        m.node('run_search', 'run_search()', fillcolor="yellow" if stage=='search' else '#87CEFA', style='filled', **component_link('run_search'))

        with m.subgraph(name='cluster_run_augment') as ra:
            ra.body.append('label="RUN_AUGMENT_HEADER"')
            ra.body.append('style="filled"')
            ra.body.append('color="#808080"')
            ra.body.append('fillcolor="yellow"' if stage=='augment' else 'fillcolor="#87CEFA"')

            ra.node('run_summary', 'run_summary()', fillcolor="yellow" if stage=='summary' else '#CAD9EF', style='filled', **component_link('run_summary'))
                    
        m.node('run_get_context', 'run_get_context()', fillcolor="yellow" if stage=='get_context' else '#87CEFA', style='filled', **component_link('run_get_context'))
        m.node('run_qa', 'run_qa()', fillcolor="yellow" if stage=='qa' else '#87CEFA', style='filled', **component_link('run_get_context'))
    
    dot.node('answer', 'answer', fillcolor='#FFD580', style='filled', shape='oval', **component_link('run_get_context'))

    dot.edge('question', 'run_search')    
    dot.edge('run_search', 'run_summary')
    dot.edge('run_summary', 'run_get_context')  
    dot.edge('run_get_context', 'run_qa')
    dot.edge('run_qa', 'answer')

    html = dot._repr_image_svg_xml()

    html = re.sub(r'<svg width=\"\d*pt\" height=\"\d*pt\"',
                  '<div style="text-align:center;"><svg width="800pt" aligned=center', html)

    #html = re.sub(r'MAIN_HEADER', f'<a href="{component_link("main")["href"]}" target=\"_blank\">main()</a>', html)
    html = re.sub(r'MAIN_HEADER', 'main()', html)

    #html = re.sub(r'RUN_AUGMENT_HEADER', f'<a href="{component_link("run_augment")["href"]}" target=\"_blank\">run_augment()</a>', html)        
    html = re.sub(r'RUN_AUGMENT_HEADER', 'run_augment()', html)    

    html = re.sub(r'stroke-width=\"2\"', 'stroke-width=\"4\"', html)

    return html

# COMMAND ----------

html_run_search_1 ='''<style>
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1><i>run_search</i> Stage Attributes</h1>

<p class="ptext">Our initial stage is run_search. Our intent is to get a list of candidate content that will be useful to augment our qa_model. Check out <a href="https://docs.databricks.com/en/generative-ai/create-query-vector-search.html#query-a-vector-search-endpoint">Query a Vector Search Endpoint</a> to see examples using Databricks Vector Search.</p>

[SEARCH_GRAPHIC]

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="run_search" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>run_search</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Dependencies</td>
      <td>If we use a Databricks <a href="https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.client.VectorSearchClient" taget="_blank">VectorSearchClient</a> with a <a href="https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex" taget="_blank">VectorSearchIndex</a>, which methods would we use?</td>
      <td><input type="text" id="txb_dependencies" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><a href="https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.client.VectorSearchClient.get_index" taget="_blank">databricks.vector_search.index.VectorSearchIndex.get_index</a></br></br><a href="https://api-docs.databricks.com/python/vector-search/databricks.vector_search.html#databricks.vector_search.index.VectorSearchIndex.similarity_search" taget="_blank">databricks.vector_search.index.VectorSearchIndex.similarity_search</a></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Application-Arguments</td>
      <td>What configurations for this stage would we want to set as an application configuration?</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>We'll want to have the vector search index set during instantiation. The two arguments we'll need for that are:</i></br></br><b>endpoint_name</b>: str</br><b>index_name</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>What input will we provide to our search?</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>We'll want to provide the question being asked to search against. To provide our question as text similarity_search uses:</i></br></br><b>query_text</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Params</td>
      <td>What parameters can be provided to our search?</td>
      <td><textarea name="txa_params" rows="6" class="txb-wide"></textarea></td>
      <td><div class="toggle-div" display=false><i>Anything that isn't our input could be a parameter. For similarity_search those are:</i></br></br><b>columns</b>: [str]</br><b>filters</b>: str</br><b>num_results</b>: int</br><b>debug_level</b>: str</li></ul></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>What kind of output should we define for this stage? The output of similarity_search string of dict. Add some structure and define the output as a <a href="https://docs.python.org/3/library/dataclasses.html#module-dataclasses">dataclass</a>.</td>
      <td><textarea name="txa_output" rows="6" class="txb-wide"></textarea></td>
      <td><div class="toggle-div" display=false></i>Similarity_search returns a dict that includes all n search results. In this MVP we'll define that result as a dataclass based on the response structure to simplify handling in our next stage run_augment:</br></br>@dataclass</br>class SimilaritySearchResult:</br>&emsp;&emsp;<b>manifest</b>: dict = ...</br>&emsp;&emsp;<b>result</b>: dict = ...</br>&emsp;&emsp;<b>next_page_token</b>: str</br>&emsp;&emsp;<b>debug_info</b>: dict = ...</div></td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>'''

# COMMAND ----------

html_run_search_2 ='''<style>
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1><i>run_summary</i> Stage Attributes</h1>

<p class="ptext">We'll look at the run_summary method before we look at what needs to be done to run it asynchronously with run_augment. Recall that the output of our previous stage will be a dataclass, but ultimately that dataclass will need to be transformed into it's essential parts as a list of tuple(id, content). Each iteration of run_summary will therefore have content and id variables in context from a single search result.</p>

[SUMMARY_GRAPHIC]

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="run_summary" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>run_summary</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Dependencies</td>
      <td>If we use a <a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html?highlight=mlflow%20deployments#mlflow.deployments.DatabricksDeploymentClient" taget="_blank">DatabricksDeploymentClient</a> to run a completion llm. Which methods would we use?</td>
      <td><input type="text" id="txb_dependencies" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html#mlflow.deployments.get_deploy_client" taget="_blank">mlflow.deployments.get_deploy_client</a></br></br><a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html#mlflow.deployments.DatabricksDeploymentClient.predict" taget="_blank">mlflow.deployments.DatabricksDeploymentClient.predict</a></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Application-Arguments</td>
      <td>What configurations for this stage coreoutine would we want to set as an application configuration? Assume that we want to use the same model endpoint for all summary predicts within the application.</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>We'll want to have the deploy_client set during instantiation. Since we know the deploy client will be Databricks, we can instantiate with a static argument, <b>get_deploy_client("databricks")</b> To keep the model_endpoint consistant across call, we'll make the model_endpoint used for summary provided as an application argument. Thus, the <b>predict</b> method will have one argument populated from an application argument:</i></br></br><b>endpoint_name</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>What input will we provide to the completion model to get a summary and relavance score? Assume you already know that the required input is a prompt. What two variables should the prompt template take?</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>From <a href="https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-text-completion-model">Text Completion Docs</a> we see that we need to provide a prompt. We'll want to use both the content we are summarizing as well as the original question, thus:</i></br></br><b>content</b>: str</br><b>question</b>: str</td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Params</td>
      <td>What parameters can be provided to our summary model? Assume that the model type we are using is a completion model. Refer to the inputs from <a href="https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-text-completion-model">Text Completion Docs</a></td>
      <td><textarea name="txa_params" rows="6" class="txb-wide"></textarea></td>
      <td><div class="toggle-div" display=false><i>Anything that isn't prompt could be a parameter. For a completion LLM type we could parameterize:</i></br></br><b>max_tokens</b>: int</br><b>temperature</b>: float</br><b>stop</b>: [str]</br><b>n</b>: int</br><b>stream</b>: bool</br><b>extra_params</b>: dict</br></br>Above, we assumed the use of a prompt template. Thus another parameter that we'll have for our stage coroutine is that prompt template:</br><b>summary_prompt</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>What kind of output should we plan for this stage coroutine?</td>
      <td><input type="text" id="txb_output" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>Our output from predict will be a response dict. From that, we'll want to extract the summary and relevance score. For debugging purposes, we should also include the id and content from the original search and define as a dataclass:</br></br>@dataclass</br>class <b>SearchResultAugmentedContent</b>:</br>&emsp;&emsp;<b>id</b>: int</br>&emsp;&emsp;<b>content</b>: str</br>&emsp;&emsp;<b>summerization</b>: str</br>&emsp;&emsp;<b>relevanceScore</b>: float</div></td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>'''

# COMMAND ----------

html_run_search_3 ='''<style>
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1><b>(OPTIONAL) </b><i>run_augment</i> Stage Attributes</h1>

<p class="ptext">Now let's plan what's necessary to execute run_summary asynchronously. The actual lesson of how asyncio works is beyond the scope of this instruction. However, we can still go through the planning exercise for a stage that runs asynchronously. If you are uncomfortable that you are planning without understanding exactly how every library works, take comfort in the fact that a lot of library discovery takes place during planning.</p>

[AUGMENT_GRAPHIC]

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="run_augment" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>run_augment</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Dependencies</td>
      <td>Use <a href="https://docs.python.org/3/library/asyncio.html" taget="_blank">asyncio</a> to execute run_summary coroutines asynchronously. You will need to have an event loop, make the run_summary method execute as a coroutine, and gather the results.</td>
      <td><textarea id="txa_dependencies" name="txa_dependencies" rows="3" class="txb-wide"></textarea></td>
      <td><div class="toggle-div" display=false><a href="https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.get_event_loop" taget="_blank">asyncio.get_event_loop</a></br><a href="https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread" taget="_blank">asyncio.to_thread</a></br><a href="https://docs.python.org/3/library/asyncio-task.html#asyncio.gather" taget="_blank">asyncio.gather</a></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>What input will we provide to run_augment? Consider what output we have from the prior stage, run_search.</td>
      <td><input type="text" id="txb_input_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>Same as output from run_search: </br><b>SimilaritySearchResult</b></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>What kind of output will we define for this stage? The gather method from asyncio creates a list. Thus our output should be a list of run_summary output.</td>
      <td><input type="text" id="txb_output" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>[<b>SearchResultAugmentedContent</b>]</td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
  document.getElementById('txa_dependencies').value='asyncio.get_event_loop\\nasyncio.to_thread\\nasyncio.gather';
</script>'''

# COMMAND ----------

html_run_search_4 ='''<style>
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1><b></b><i>run_get_context</i> Stage Attributes</h1>

<p class="ptext">This stage we just need to take the results of our run_augment stage and convert into a context for our QA stage. That's right, a stage doesn't have to include an LLM. While this stage could have been made part of the stage before or after, it's broken out as it's own entity so this compound app could have a non-LLM stage.</p>

[GET_CONTEXT_GRAPHIC]

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="run_get_context" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>run_get_context</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Dependencies</td>
      <td>We have a list and we need to sort it. We can do this python pure. </td>
      <td><input type="text" id="input_dependencies" value="list.sort" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><a href="https://docs.python.org/3/library/stdtypes.html#list.sort" taget="_blank">list.sort</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>What input will we provide to run_get_context? Consider what output we have from the prior stage, run_augment.</td>
      <td><input type="text" id="txb_input_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>Same as output from run_augment: </br>[<b>SearchResultAugmentedContent</b>]</td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>What kind of output will we define for this stage? Intent is to simply consolidate the top three results into a single string and use that as context in the QA model.</td>
      <td><input type="text" id="txb_output" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>context</b>: str</td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>'''

# COMMAND ----------

html_run_search_5 ='''<style>
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1><i>run_qa</i> Stage Attributes</h1>

<p class="ptext">We'll look at the run_qa stage. To make things interesting we are going to use a Chat LLM type instead of a Completion LLM type like we used in run_summary. That means you should take a look at <a href="https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model">Chat Model Docs</a>. It may look the same as the completion model signature, but look closely, it's not.</p>

[QA_GRAPHIC]

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="run_qa" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>run_qa</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Dependencies</td>
      <td>If we use a <a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html?highlight=mlflow%20deployments#mlflow.deployments.DatabricksDeploymentClient" taget="_blank">DatabricksDeploymentClient</a> to run a chat llm. Which methods would we use? Hint: we can use the same dependencies for completion llms as we will use for chat llms.</td>
      <td><input type="text" id="txb_dependencies" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html#mlflow.deployments.get_deploy_client" taget="_blank">mlflow.deployments.get_deploy_client</a></br></br><a href="https://mlflow.org/docs/latest/python_api/mlflow.deployments.html#mlflow.deployments.DatabricksDeploymentClient.predict" taget="_blank">mlflow.deployments.DatabricksDeploymentClient.predict</a></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Application-Arguments</td>
      <td>What configurations for this stage  would we want to set as an application configuration? Assume that we want to use the same model endpoint for all QA predicts within the application.</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>We'll want to have the deploy_client set during instantiation. Since we know the deploy client will be Databricks, we can instantiate with a static argument, <b>get_deploy_client("databricks")</b> To keep the model_endpoint consistant across calls, we'll make the model_endpoint used for QA provided as an application argument. Thus, the <b>predict</b> method will have one argument populated from an application argument:</i></br></br><b>endpoint_name</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>What input will we provide to the chat model the uses our context and the original question? Assume that we will again use a prompt template. What two variables should the prompt template take?</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><i>From <a href="https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model">Chat Model Docs</a> we see that we need to provide messages. The messages format is a list of dict to handle message history, but we'll just need to coerse a prompt into this format. Our prompt will take the following:</br></br><b>context</b>: str</br><b>question</b>: str</td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Params</td>
      <td>What parameters can be provided to our qa model? Assume that the model type we are using is a chat model. Refer to the inputs from <a href="https://docs.databricks.com/en/machine-learning/model-serving/score-foundation-models.html#query-a-chat-completion-model">Chat Model Docs</a></td>
      <td><textarea name="txa_params" rows="6" class="txb-wide"></textarea></td>
      <td><div class="toggle-div" display=false><i>Anything that isn't prompt could be a parameter. For a chat LLM type we could parameterize:</i></br></br><b>max_tokens</b>: int</br><b>temperature</b>: float</br></br></br>Above, we assumed the use of a prompt template. Thus another parameter that we'll have for qa is our own:</br></br></br><b>qa_prompt</b>: str</div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>What kind of output should we plan for this stage? The final output of the model is an answer and single string. However, we would like the full response available as an output of this stage.</td>
      <td><input type="text" id="txb_output" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>Our output from predict will be a response dict. From that, we'll put that in a dataclass for ease of use:</br></br>@dataclass</br>class <b>SummaryModelResult</b>:</br>&emsp;&emsp;<b>id</b>: int</br>&emsp;&emsp;<b>object</b>: str</br>&emsp;&emsp;<b>model</b>: str</br>&emsp;&emsp;<b>choices</b>: [dict] = ...</br>&emsp;&emsp;<b>usage</b>: dict = ...</div></td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>'''

# COMMAND ----------

html_run_search_6 ='''<style>
  /* Python IDE Colors */
  .keyword { color: #569CD6; } /* Blue */
  .reserved { color: #800080; } /* Purple */  
  .constant { color: gray; } /* Brown */
  .operator { color: #6A9955; } /* Green */
  .ctype { color: #008000; } /* Green */
  /* Light gray box */
  .code-box {
    background-color: #f2f2f2;
    padding: 20px;
    border-radius: 8px;
    text-align: left;
    font-size: 18px;
    line-height: 1.7
  }
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1>(OPTIONAL) <i>main</i> Putting the stages together</h1>

<p class="ptext">Time to take all that work planning each individual stage and putting it all together. Take a look below how our individual state planning greatly simplifies chaining our stages together. Not only is it sussinct and easy to read, the logical separation and inclusion of dataclasses will also make the following future developer tasks much more straight forward:
<ul class="ptext">
  <li><b>Testing</b>: Test can be written by each module we've created making it easier than would have otherwise been if we wrote our code as a script.</li>
  <li><b>Tracing</b>: Beyond the scope of this instruction, but adding trace decorators each stage will allow better evaluation for future improvements.</li>
  <li><b>Parameterizing</b>: By extraction out what all parameters are in the solution, we now have an understanding of our parameter space and ability to create new model versions by simply modifying parameters with no code updates.</li>
</ul></p>

[MAIN_GRAPHIC]

<div class="code-box" >
  <pre>
    <code>
      <span class="reserved">def</span> <span class="keyword">main</span>(<span class="reserved">self</span>, question: <span class="ctype">str</span>) -> <span class="ctype">str</span>:
          <span>search_result:</span> <span class="ctype">SimilaritySearchResult</span> </span>=</span> </span class="type"></span><span class="reserved">self</span></span">.run_search(question)</span>
          <span>augmented_result:</span> <span class="ctype">Tuple[SearchResultAugmentedContent, ...]</span> </span>=</span> </span class="type"></span><span class="reserved">self</span></span">.run_augment(search_result, question)</span>
          <span>context:</span> <span class="ctype">str</span> </span>=</span> </span class="type"></span><span class="reserved">self</span></span">.run_get_context(augmented_result)</span>
          <span>qa_result:</span> <span class="ctype">QaModelResult</span> </span>=</span> </span class="type"></span><span class="reserved">self</span></span">.run_qa(question, context)</span>
          <span class="reserved">return</span>qa_result.get_answer()
    </code>
  </pre>
</div>

<table>
  <thead>
    <tr>
      <th>Attribute</th>
      <th>Considerations</th>
      <th>Student Answer</th>
      <th>Instruction Approach</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td bgcolor= "#FFFFE0">Name</td>
      <td>Name the method, be succinct</td>
      <td><input type="text" id="input_name" value="main" class="txb-wide"></td>
      <td><div class="toggle-div" display=false><b>main</b></div></td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Input</td>
      <td>Using the code above, what is our input?</td>
      <td><input type="text" id="txb_input" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>question: str</td>
    </tr>
    <tr>
      <td bgcolor= "#FFFFE0">Signature-Output</td>
      <td>Using the code above, what is our output?</td>
      <td><input type="text" id="txb_output" class="txb-wide"></td>
      <td><div class="toggle-div" display=false>str</div></td>
    </tr>
  </tbody>
</table>

</br><button id="toggleButton">Hide/Show Solution</button></br></br>


<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>'''

# COMMAND ----------

html_run_search_7 = '''<style>
  /* Python IDE Colors */
  .keyword { color: #569CD6; } /* Blue */
  .reserved { color: #800080; } /* Purple */  
  .constant { color: gray; } /* Brown */
  .operator { color: #6A9955; } /* Green */
  .ctype { color: #008000; } /* Green */
  /* Light gray box */
  .code-box {
    background-color: #f2f2f2;
    padding: 20px;
    border-radius: 8px;
    text-align: left;
    font-size: 18px;
    line-height: 1.7;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    border: 1px solid #ccc;
  }
  th, td {
    border: 1px solid #ccc;
    padding: 8px;
    text-align: left;
  }
  th {
    background-color: #f2f2f2;
  }
  .toggle-div {
    display: block;
    margin-bottom: 10px;
  }
  .hidden {
    display: none;
  }
  .txb-wide {
    width: 400px;
    padding: 8px;
    border: 1px solid #ccc;
    border-radius: 4px;
    box-sizing: border-box; /* Include padding and border in total width */
  }
  .ptext {
    font-size: 16px;
  }
</style>

<h1>(OPTIONAL) Compound App as a Pyfunc Model</h1>

<p class="ptext">Our main method is nice, it can take a string and return a string. However, while local execution is nice, we'd really like our model to be deployable and be evaluated just as if it were a foundation model or external model. MLFlow and Databricks accomplish model deployments through built-in model types. Since we are not using a model library, we can use MLFlow's general form of a pyfunc model, and we can use a specific subclass of mlflow.PythonModel.</p>

[MAIN_GRAPHIC]

</br><button id="toggleButton">Hide/Show Solution</button></br></br>

<script>
  document.getElementById('toggleButton').addEventListener('click', function() {
    var divs = document.querySelectorAll('.toggle-div');
    divs.forEach(function(div) {
      div.classList.toggle('hidden');
    });
  });
  document.getElementById('toggleButton').click();
</script>
'''