from flask import Flask, render_template, request, jsonify
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.utilities import OpenWeatherMapAPIWrapper
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from typing import Annotated,TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel 
from langchain_core.tools import StructuredTool
import os

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field


class InternetTool(BaseModel):
    """Input for the Tavily Tool"""
    query: str = Field(description="search query to look up")

@tool
def Internet_tool(query:str):
    """Input for the Tavily Tool"""
    wrapper = TavilySearchAPIWrapper()
    tavily_tool = TavilySearchResults(
        name="Internet-Tool",
        api_wrapper=wrapper,
        max_results=5,
        args_schema=InternetTool,
    )
    return tavily_tool.run({"query": query})

from dotenv import load_dotenv
load_dotenv()

from langchain_core.tools import tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from pydantic import BaseModel, Field

class WikiInputs(BaseModel):
    """Inputs to the Wikipedia tool"""
    query:str = Field(description="Query to look up in wikipedia, should 3 or less words")


@tool
def wikipedia_tool(query:str):
    """Inputs to the Wikipedia tool"""
    wrapper = WikipediaAPIWrapper(top_k_results=2,doc_content_chars_max=2000)
    tool = WikipediaQueryRun(
        name="Wiki-Input",
        api_wrapper=wrapper,
        args_schema=WikiInputs,
        return_direct=True,
    )
    return tool.run({"query": query})


from dotenv import load_dotenv
load_dotenv()

import re
import json
import faiss
import mysql.connector
from langchain.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class SQLQueryInput(BaseModel):
    """Input for the SQL Query Processing Tool"""
    question: str = Field(description="User's natural language query about the database")


db = None
vector_db = None
llm = None
sql_chain = None

def initialize_components():
    """Initialize all components needed for the hybrid search system."""
    global db, vector_db, llm, sql_chain
    
   
    db_uri = "mysql+mysqlconnector://root:Mdcineluv12#@localhost:3306/chinook"
    db = SQLDatabase.from_uri(db_uri)
    
   
    llm = HuggingFaceEndpoint(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.2,
        max_new_tokens=200,
    )
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    dimension = 384
    index = faiss.IndexFlatL2(dimension)
    docstore = InMemoryDocstore({})
    index_to_docstore_id = {}
    
    vector_db = FAISS(
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
        embedding_function=embeddings
    )
    
    store_data_in_vector_db()
    
    sql_prompt = ChatPromptTemplate.from_template("""
    Based on the table schema below, write a query that would answer the user's question.
    Use only standard SQL syntax compatible with MySQL.
    
    Schema:
    {schema}
    
    Question: {question}
    SQL query:
    """)
    
    sql_chain = (
        RunnablePassthrough.assign(schema=get_schema)
        | sql_prompt
        | llm
        | StrOutputParser()
        | (lambda x: clean_sql_query(x))
    )

def get_schema(_):
    """Get the database schema information."""
    return db.get_table_info()

def clean_sql_query(query):
    """Remove any unwanted formatting from LLM-generated SQL queries."""
    return re.sub(r"```sql|```", "", query).strip()

def store_data_in_vector_db():
    """Fetch data from MySQL and store embeddings in FAISS."""
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="Mdcineluv12#",
        database="chinook"
    )
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("SELECT ArtistId, Name FROM Artist") 
    rows = cursor.fetchall()
    
    texts = [row["Name"] for row in rows]
    metadatas = [{"ArtistId": row["ArtistId"]} for row in rows]

    vector_db.add_texts(texts, metadatas)

    cursor.close()
    connection.close()
    print("✅ Data stored in FAISS Vector DB.")

def search_vector_db(query):
    """Perform a semantic search for the most relevant data."""
    results = vector_db.similarity_search(query, k=3) 
    return results

def run_query(query):
    """Execute the SQL query and return results."""
    try:
        connection = mysql.connector.connect(
            host="localhost",
            user="root",
            password="Mdcineluv12#",
            database="chinook"
        )
        cursor = connection.cursor(dictionary=True)
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        connection.close()
        return json.dumps(results, default=str)
    except Exception as e:
        return f"Error executing query: {str(e)}"

@tool
def hybrid_search_tool(question: str):
    """Process a user question using both SQL and semantic search.
    
    Args:
        question: The natural language question to answer about the database
        
    Returns:
        A natural language response answering the user's question
    """

    if db is None or vector_db is None or llm is None or sql_chain is None:
        initialize_components()
    
    try:
        vector_results = search_vector_db(question)
        vector_context = ""
        
        if vector_results:
            vector_context = "Vector Search Results: " + ", ".join([r.page_content for r in vector_results])
        
        query = sql_chain.invoke({"question": question})
        
        response = run_query(query)
        
        response_prompt = ChatPromptTemplate.from_template("""
        Based on the schema, question, SQL query, and SQL response below, write a natural language response.
        If there's an error, please explain what might have gone wrong.
        
        Schema:
        {schema}
        
        Question: {question}
        SQL Query: {query}
        SQL Response: {response}
        
        {vector_context}
        
        Natural language response:
        """)
        
        nl_response = response_prompt.format(
            schema=get_schema(None),
            question=question,
            query=query,
            response=response,
            vector_context=vector_context
        )
        
        final_response = llm.invoke(nl_response)
        return final_response
    except Exception as e:
        return f"Error processing query: {str(e)}"

initialize_components()

from langchain_core.tools import tool
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class TextSearchInput(BaseModel):
    """Input for the Sentence Embedding Search Tool"""
    query: str = Field(description="The text query to search for similar entries in the dataset")
    k: int = Field(default=2, description="Number of similar results to return")

encoder = None
index = None
df = None

def initialize_text_search_components():
    """Initialize the sentence transformer model and FAISS index."""
    global encoder, index, df
    
    try:
        df = pd.read_csv(r"D:\projects\LLM AI bot\my_data.csv")
        print(f"✅ CSV loaded successfully with {len(df)} rows")
        
        encoder = SentenceTransformer("all-mpnet-base-v2")
        
        texts = df.text.fillna("").astype(str).tolist()
        vectors = encoder.encode(texts)
        
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        
        index.add(vectors)
        
        print(f"✅ Initialized sentence embedding search with vector dimension: {dim}")
        return True
    except Exception as e:
        print(f"Error initializing components: {str(e)}")
        return False

@tool
def sentence_embedding_search(query: str, k: int = 3):
    """Search for semantically similar text entries in the CSV dataset.
    Use this tool when explicitly asked to search in CSV data or for queries about food, recipes, or items in the dataset.
    
    Args:
        query: The text query to search for similar entries
        k: Number of similar results to return (default: 3)
        
    Returns:
        The most similar text entries from the dataset with their categories
    """
    global encoder, index, df
    
    if encoder is None or index is None or df is None:
        success = initialize_text_search_components()
        if not success:
            return "Failed to initialize CSV search components. Please check file path and try again."
    
    try:
        if ":" in query:
            parts = query.split(":", 1)
            if len(parts) > 1:
                query = parts[1].strip()
        
        print(f"Searching CSV for: {query}")
        
        vec = encoder.encode([query])
        
        distances, indices = index.search(vec, k=k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(df):
                food_item = df.iloc[idx]['text']
                category = df.iloc[idx][' category'].strip() if ' category' in df.columns else 'Unknown'
                
                item = {
                    'food': food_item,
                    'category': category
                }
                results.append(item)
        
        if not results:
            return "No similar entries found in the CSV dataset."
            
        formatted_response = f"Here are similar food items from the CSV dataset:\n\n"
        for result in results:
            formatted_response += f"- {result['food']} (Category: {result['category']})\n"
        
        return formatted_response
    except Exception as e:
        return f"Error performing semantic search on CSV: {str(e)}"
    


initialize_success = initialize_text_search_components()
print(f"CSV search initialization: {'Success' if initialize_success else 'Failed'}")


from langchain_core.tools import tool
from pydantic import BaseModel, Field
import arxiv
from typing import List, Optional
import textwrap

class ArxivAPIInput(BaseModel):
    """Input for the ArXiv API Tool"""
    query: str = Field(description="The research topic to search for on arXiv")
    max_results: int = Field(default=5, description="Maximum number of papers to return")
    sort_by: str = Field(default="relevance", description="Sort order (relevance, lastUpdatedDate, submittedDate)")

@tool
def arxiv_api_tool(query: str, max_results: int = 5, sort_by: str = "relevance"):
    """Search for scientific papers on arXiv.
    
    Args:
        query: The research topic or search term to find papers about
        max_results: Maximum number of papers to return (default: 5)
        sort_by: Sort order (relevance, lastUpdatedDate, submittedDate)
        
    Returns:
        A summary of the most relevant scientific papers
    """
    try:
        sort_options = {
            "relevance": arxiv.SortCriterion.Relevance,
            "lastUpdatedDate": arxiv.SortCriterion.LastUpdatedDate,
            "submittedDate": arxiv.SortCriterion.SubmittedDate
        }
        
        sort_criterion = sort_options.get(sort_by, arxiv.SortCriterion.Relevance)
        
        client = arxiv.Client()
        
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion
        )
        
        results = list(client.results(search))
        
        if not results:
            return f"No papers found on arXiv for '{query}'."
        
        formatted_results = f"## Top {len(results)} arXiv papers on '{query}':\n\n"
        
        for i, paper in enumerate(results, 1):
            authors = paper.authors
            if len(authors) > 3:
                authors_text = f"{authors[0]}, {authors[1]}, {authors[2]}, et al."
            else:
                authors_text = ", ".join(str(author) for author in authors)
            
            published = paper.published.strftime("%d %b %Y") if paper.published else "N/A"
            
            summary = paper.summary.replace('\n', ' ').strip()
            if len(summary) > 300:
                summary = summary[:297] + "..."
            
            formatted_results += f"### {i}. {paper.title}\n"
            formatted_results += f"**Authors:** {authors_text}\n"
            formatted_results += f"**Published:** {published} | **Categories:** {', '.join(paper.categories)}\n"
            formatted_results += f"**Abstract:** {summary}\n"
            formatted_results += f"**Link:** [arXiv:{paper.entry_id.split('/')[-1]}]({paper.pdf_url})\n\n"
        
        return formatted_results
    
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"
    

    from langchain_core.tools import tool
from pydantic import BaseModel, Field
import requests
import os

class WeatherAPIInput(BaseModel):
    """Input for the Weather API Tool"""
    location: str = Field(description="The location to get weather information for (city name or coordinates)")
    units: str = Field(default="metric", description="The units to use for temperature (metric or imperial)")

@tool
def weather_api_tool(location: str, units: str = "metric"):
    """Get current weather information for a specific location.
    
    Args:
        location: The city name or coordinates to get weather information for
        units: The units to use for temperature (metric or imperial)
        
    Returns:
        A summary of the current weather conditions
    """
    try:
        api_key = os.getenv("OPENWEATHERMAP_API_KEY", "your_openweathermap_api_key_here")
        
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,  
            "appid": api_key,
            "units": units 
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            if "," in location:
                try:
                    lat, lon = location.split(",")
                    params = {
                        "lat": lat.strip(),
                        "lon": lon.strip(),
                        "appid": api_key,
                        "units": units
                    }
                    response = requests.get(url, params=params)
                except:
                    pass
        
        if response.status_code != 200:
            return f"Error fetching weather: {response.status_code} - {response.text}"
        
        data = response.json()
        
        city_name = data.get("name", "Unknown Location")
        country = data.get("sys", {}).get("country", "")
        
        weather_desc = data.get("weather", [{}])[0].get("description", "Unknown")
        weather_main = data.get("weather", [{}])[0].get("main", "Unknown")
        
        temp = data.get("main", {}).get("temp", "N/A")
        feels_like = data.get("main", {}).get("feels_like", "N/A")
        humidity = data.get("main", {}).get("humidity", "N/A")
        
        wind_speed = data.get("wind", {}).get("speed", "N/A")
        
        unit_symbol = "°C" if units == "metric" else "°F"
        wind_unit = "m/s" if units == "metric" else "mph"
        
        result = f"Current weather in {city_name}{', ' + country if country else ''}:\n\n"
        result += f"**Conditions:** {weather_desc.capitalize()}\n"
        result += f"**Temperature:** {temp}{unit_symbol} (Feels like: {feels_like}{unit_symbol})\n"
        result += f"**Humidity:** {humidity}%\n"
        result += f"**Wind:** {wind_speed} {wind_unit}\n"
        
        return result
    
    except Exception as e:
        return f"Error processing weather request: {str(e)}"
 
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",)

tools = [Internet_tool, wikipedia_tool, hybrid_search_tool, sentence_embedding_search,arxiv_api_tool,weather_api_tool]

llm_with_tools = llm.bind_tools(tools=tools)

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition, ToolNode

class State(TypedDict):
    messages: Annotated[list, add_messages]

def build_graph():
    graphbuilder = StateGraph(State)
    
    def chatbot(state: State):
        return {"messages": [llm_with_tools.invoke(state["messages"])]}
    
    graphbuilder.add_node("chatbot", chatbot)
    graphbuilder.add_edge(START, "chatbot")
    tool_node = ToolNode(tools=tools)
    graphbuilder.add_node("tools", tool_node)
    
    graphbuilder.add_conditional_edges(
        "chatbot",
        tools_condition
    )
    
    graphbuilder.add_edge("tools", "chatbot")
    graphbuilder.add_edge("chatbot", END)
    
    return graphbuilder.compile()

# Initialize the graph
graph = build_graph()

# Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_message = request.form['message']
    
    try:
        events = graph.stream(
            {"messages": [("user", user_message)]},
            stream_mode="values"
        )
        
        last_response = None
        for event in events:
            last_response = event["messages"][-1].content
        
        return jsonify({"response": last_response})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
