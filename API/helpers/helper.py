import time
import tiktoken
import ast
import logging
import os
import re
import datetime

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.tools import Tool, tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import (
    FewShotPromptTemplate,
    PromptTemplate,
    MessagesPlaceholder,
    ChatPromptTemplate,
)
from langgraph.prebuilt import create_react_agent
from langchain_community.vectorstores import FAISS
from API.helpers.dbReference import get_PBILinks
from langchain_core.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_core.documents import Document
from API.snowflake_connection import get_database_type
from API.helpers.utils import (
    SnowInfoSQLDatabaseTool,
    extract_examples_multi,
    group_tables_by_database,
    load_domain_file,
    load_domains_index,
    detect_domains,
    build_context_multi,
    load_multiple_domains,
    build_prompt,
)
from sqlalchemy import text

CURRENT_DB = None
# load_dotenv()
# In case we want to load from a specific path in future, provide the path to .env.
load_dotenv(
    dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"),
    override=True,
)

llm_name = os.environ.get("LLM_MODEL")
embedding_model_name = os.environ.get("EMBEDDING_MODEL_NAME")
openai_endpoint = os.environ.get("OPENAI_ENDPOINT")

azure_llm = AzureChatOpenAI(
    temperature=0,
    azure_endpoint=openai_endpoint,
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_version=os.environ.get("LLM_OPENAI_API_VERSION"),
    deployment_name=llm_name,
)

embeddings = AzureOpenAIEmbeddings(
    api_key=os.environ.get("OPENAI_API_KEY"),
    api_version=os.environ.get("EMBEDDING_OPENAI_API_VERSION"),
    azure_endpoint=openai_endpoint,
    azure_deployment=embedding_model_name,
)


# Vector Store Creation
def create_vector_stores(embeddings):
    """Creates and returns vector stores for MaterialFamily and Make."""
    snowflake_db, db = get_database_type(db_type="snowflake")

    def query_as_list(db, query):
        """Executes a query and returns a list of values with trimmed whitespace."""
        res = db.run(query)
        try:
            res = ast.literal_eval(res)
            res = [str(item).strip() for item in res]
        except (SyntaxError, ValueError):
            res = []
        return res

    # Query distinct values from DB
    MaterialFamily = query_as_list(
        db, "select distinct materialfamily from BOT_DeviceSales"
    )
    Make = query_as_list(
        db,
        "select distinct Make from BOT_DeviceSales union select distinct make from BOT_ACCESSORYSALES_V2",
    )
    # Create vector stores
    vector_store_MaterialFamily = FAISS.from_texts(MaterialFamily, embeddings)
    vector_store_Make = FAISS.from_texts(Make, embeddings)

    return vector_store_MaterialFamily, vector_store_Make


def create_dashboard_vector_store(embeddings):
    """Embeds Power BI dashboard metadata from dataframe into FAISS vector store."""
    documents = []
    excelLinks_df = get_PBILinks()
    for _, row in excelLinks_df.iterrows():
        # Include URL inside page_content
        content = (
            f"ReportName: {row['ReportName']}\n"
            f"KPIs: {row['KPIs']}\n"
            f"Description: {row['Description']}\n"
            f"URL: {row['URL']}"
        )

        metadata = {
            "report_name": row["ReportName"],
            "kpi": row["KPIs"],
            "description": row["Description"],
            "Url": row["URL"],
        }

        documents.append(Document(page_content=content, metadata=metadata))

    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


vector_store_MaterialFamily, vector_store_Make = create_vector_stores(embeddings)
vector_store_powerbi_url = create_dashboard_vector_store(embeddings)

# Create retrievers
retriever_MaterialFamily = vector_store_MaterialFamily.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)
retriever_Make = vector_store_Make.as_retriever(
    search_type="similarity", search_kwargs={"k": 5}
)
retriever_powerbiurl = vector_store_powerbi_url.as_retriever(
    search_type="similarity", search_kwargs={"k": 3}
)


@tool
def query_database_schema(user_question: str, table: dict) -> str:
    """Use LLM to determine relevant tables and retrieve schema details."""
    global CURRENT_DB

    table_selection_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="""
You are an expert database assistant. Identify which tables are needed based on the user query.

Rules:
*** ALWAYS check the database_type before quering on ANY TABLE, Whether it's belong to azure_sql or snowflake. ***
*** For any forecast related query ALWAYS consider BOT_DEVICESALESFORECAST table. ***
1. Use the 'BOT_BW_RETAIL_CSI' table only when the user is an Retail Allocation Planner or the question is about Retail allocation planning, do not consider any other tables.
2. If the user is not a Retail allocation planner, Strictly DO NOT USE 'BOT_BW_RETAIL_CSI' table.
3. When a user is a Demand Planner, For Sure use these three tables : BOT_DeviceSales for all sales information, BOT_DEVICEINVENTORY_CURRENT or BOT_DEVICEINVENTORY_HISTORY for all inventory information, and BOT_DEVICESALESFORECAST for all forecast information.
4. NPI Devices include: Galaxy Z Fold8, Samsung Galaxy Z Fold8, Galaxy Z Fold8 Ultra, Samsung Galaxy Z Fold8 Ultra, Galaxy Z Flip8, Samsung Galaxy Z Flip8, Pixel 11, Pixel 11 Pro, Pixel 11 Pro XL, Pixel 11 Pro Fold.
5. If the User is asking about NPI (New Product Integration), NPI devices performance, pre-orders, or anything related to NPI devices:
   - Strictly only consider 'BOT_NPI_PREORDERS' table to calculate 'Orders Till Date' for all NPI devices.
   - Strictly ALWAYS only consider the 'BOT_NPI_COMPARISON' table to show the YoY (Year over Year) comparison and the Forecast vs Actual percent difference.
   - DO NOT suggest general sales, inventory, promotions, or forecast tables for NPI devices or NPI queries, even if they contain similar metrics.
   - Collect Promotion, Pricing, launch details of NPI devices from the system information provided on the NPI devices, instead of referring to other tables.
   - Always stick to the the NPI format mentioned below.
6. For all other queries, analyze the table descriptions, table database_type and user question to determine which tables are relevant.
7. We do not have stocks in transit or in-route information, so never select tables for that.
8. When asked for Antennas, search in Accessories.
9. There are two tables for Inventory information — 'BOT_DEVICEINVENTORY_CURRENT' and 'BOT_DEVICEINVENTORY_HISTORY' which must be used exclusively based on the user’s requested inventory date.
   - The 'BOT_DEVICEINVENTORY_CURRENT' table contains the most recent and curent inventory snapshot, (today’s date), and also serves as the default source when a query explicitly requests about current or recent inventory.It is the default source for any user query referring to current or recent inventory.
   - The 'BOT_DEVICEINVENTORY_HISTORY' holds the historical inventory data for the last three months from the current date. It should only be used when the user query pertains strictly to dates that fall within that recent 3-month window.
   - Always by default use only the 'BOT_DEVICEINVENTORY_CURRENT' table for any queries related to recent or current inventory details unless user explicitly ask for historic data (within 3 months), in that case USE BOT_DEVICEINVENTORY_HISTORY.
   - Under no condition should both tables be queried together, and each must only be used within its defined date range.
10. When asked about Activations or Upgrades, do not consider sales table as activations or upgrades.
11. When user asks about RSO or Retail Stock Out of a particular Device or SKU, then consider either 'BOT_DEVICEINVENTORY_CURRENT' or 'BOT_DEVICEINVENTORY_HISTORY' table.
12. When User asks for store information, always filter for 'Open' in status column unless they specifically ask for other statuses.
13. When asked about inventory or sales at store level or something which is not relevant to the data we have, politely reply saying "We do not have information related to your question".
14. When user's query involves about dashboard or report, or contains phrases like "Power BI", "Dashboard", "Reports" or specific report names:
   - Use the retriever_powerbiurl tool to find the most relevant Power BI dashboard link.
   - Match user queries semantically using metadata (reportname, KPIs, description).
   - ALWAYS format Power BI links using markdown syntax: [Display Text](actual_full_url)
   - The markdown format [Display Text](URL) will render as ONLY the Display Text in blue - the URL is hidden.
   - Use emoji + short name format:
       📊 [Sales Dashboard](full_powerbi_url_here)
       📦 [Inventory Dashboard](full_powerbi_url_here)
       📈 [Forecast Dashboard](full_powerbi_url_here)
15. When the user asks sales regarding LOB (Line Of Business), always use the LOB column in the BOT_DeviceSales table.
16. When a user query contains the term "Open Order" or "Open" or "Open Orders" anywhere in the query, then query ONLY the BOT_OPENORDERS table with filter on STATUS Column as STATUS IN ('Open Order', 'BackOrder'). When the user mentions “TFB” for Open Orders, always map it to "TBG".
   - Apply a filter on the STATUS column as follows: STATUS IN ('Open Order', 'BackOrder')
17. For every user query that includes any of the following terms: "TBG", "TFB", or "Business" (e.g., "What were the past week sales for TGB?") — you must enforce the following rules without exception:
   Mandatory Follow-up: Ask only this question and nothing else before proceeding: "Do you want to see sales for TBG customers or sales made in the General Business/Business Direct channel?". Do not proceed until the user answers this question explicitly.
   Strict Filtering Rules: If the user responds indicating TBG customers, then filter only for the value TFB from the 'LOB' column. If the user responds indicating General Business/Business Direct, then filter only for the value TBG from the 'TransactionChannel' column. Do not include or infer any other values in either column under any circumstances.
   No Assumptions: Do not assume the intended column or value based on the user's original query. Always require a clear and explicit response to the follow-up question before taking any action.
   Reject Ambiguity: If the user's input is ambiguous or references both TBG and TFB without clarifying the column, pause and ask the follow-up question again. Do not continue until an unambiguous answer is provided.
20. Make sure that BOT_DEVICESALESFORECAST table belongs to snowflake database_type. Whenever you are fetching the forecast data make sure to connect with snowflake database and always use BOT_DEVICESALESFORECAST table.
21. If the User Question has Regarding POWERBI DASHBOARD, ALWAYS format links in markdown (NEVER show raw URLs):
   📊 [Sales Dashboard](https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/b1e04387-9e5d-4505-851b-0cb1c79ab07e/ReportSection2a1c3ad907d0096d6ec9?experience=power-bi)
   📦 [Inventory Dashboard](https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/a4e60792-2a53-4109-b5e3-e42dd1c4513b/daf95ac97bd2bde961b5?experience=power-bi)
   📈 [Forecast Dashboard](https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/19f46d0e-4c73-412c-859a-06bc7e233914/ReportSection36d7454aa94d135439b2?experience=power-bi)
   📱 [Samsung Galaxy Devices](https://app.powerbi.com/groups/me/rdlreports/0e9d590c-539a-401e-a4be-2a616c986f30?experience=power-bi)
22. Follow the response formatting guidelines and query constraints defined in this system prompt to structure your answer appropriately.
"""
            ),
            HumanMessage(
                content="Given the following table descriptions and user question, list the relevant tables."
            ),
            HumanMessage(content=f"Table Descriptions: {table}"),
            HumanMessage(content=f"User Question: {user_question}"),
            HumanMessage(
                content="If the User Question has Retail Allocation Planner, Use ONLY the 'BOT_BW_RETAIL_CSI' table."
            ),
            HumanMessage(
                content="Return only the table names as a comma-separated list."
            ),
        ]
    )
    relevant_tables_response = azure_llm.invoke(
        table_selection_prompt.format_messages()
    )
    relevant_tables = relevant_tables_response.content.strip().split(",")
    if relevant_tables is None or not relevant_tables or relevant_tables == [""]:
        return None, CURRENT_DB
    table = group_tables_by_database(
        relevant_tables=relevant_tables, table_descriptions=table
    )
    table_details = None
    snowflake_db, db = get_database_type(db_type="snowflake")
    snowflake_info_tool = SnowInfoSQLDatabaseTool(db=db)
    if "snowflake" in table:
        CURRENT_DB = "snowflake"
        tables = table["snowflake"]
        table_name_str = ", ".join(tables)
        table_details = snowflake_info_tool.invoke({"table_names": table_name_str})
    return table_details, CURRENT_DB


# Main dashboards that are ALWAYS returned (first) when the search text is about their topic,
# regardless of what the semantic search over PBI_Index.xlsx ranks. (topic regex, label, url)
_PINNED_PBI_LINKS = [
    (
        re.compile(r"sale|sold|revenue|activation|upgrade|order|units?", re.IGNORECASE),
        "Sales Dashboard",
        "https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/b1e04387-9e5d-4505-851b-0cb1c79ab07e/ReportSection2a1c3ad907d0096d6ec9?experience=power-bi",
    ),
    (
        re.compile(r"inventor|stock|wos|rso|weeks of supply|supply", re.IGNORECASE),
        "Inventory Dashboard",
        "https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/a4682582-5935-48f9-9954-cd8fff4ee6bd/741099a6e8dc7756a692?ctid=be0f980b-dd99-4b19-bd7b-bc71a09b026c&referrer=desktop&experience=power-bi",
    ),
    (
        re.compile(r"forecast|plan", re.IGNORECASE),
        "Forecast Dashboard",
        "https://app.powerbi.com/groups/me/apps/42220469-ada9-4925-b2b4-42df98e223c8/reports/19f46d0e-4c73-412c-859a-06bc7e233914/ReportSection36d7454aa94d135439b2?ctid=be0f980b-dd99-4b19-bd7b-bc71a09b026c&referrer=desktop&experience=power-bi&bookmarkGuid=Bookmark12efec79267425e7b8b1",
    ),
    (
        re.compile(r"samsung|galaxy", re.IGNORECASE),
        "Samsung Galaxy Devices",
        "https://app.powerbi.com/groups/me/rdlreports/0e9d590c-539a-401e-a4be-2a616c986f30?experience=power-bi",
    ),
]
_PBI_REPORT_ID = re.compile(r"/(?:rdl)?reports/([0-9a-f]{8}-[0-9a-f-]{27})", re.IGNORECASE)


def _pbi_report_id(url: str) -> str:
    m = _PBI_REPORT_ID.search(url or "")
    return m.group(1).lower() if m else (url or "")


def search_powerbi_dashboard(query: str) -> str:
    """Find Power BI dashboards: pinned main dashboards for the topic first, then index matches."""
    entries = []  # (label, url, kpi, description)
    seen = set()
    for pattern, label, url in _PINNED_PBI_LINKS:
        if pattern.search(query):
            seen.add(_pbi_report_id(url))
            entries.append((label, url, "N/A", "Main dashboard for this topic"))

    for doc in retriever_powerbiurl.invoke(query):
        url = doc.metadata.get("Url", "")
        if _pbi_report_id(url) in seen:
            continue
        seen.add(_pbi_report_id(url))
        entries.append(
            (
                doc.metadata.get("report_name", "N/A"),
                url,
                doc.metadata.get("kpi", "N/A"),
                doc.metadata.get("description", "N/A"),
            )
        )

    if not entries:
        return "No dashboards found."

    output = "Matching Dashboards (main dashboards listed first; format as markdown links):\n"
    for i, (name, url, kpi, desc) in enumerate(entries, 1):
        output += f"\n{i}. {name}"
        output += f"\n   Format as: 📊 [{name}]({url})"
        output += f"\n   KPI: {kpi}"
        output += f"\n   Description: {desc}\n"
    return output


@tool
def validate_snowflake_sql(sql_query: str) -> dict:
    """
    Validate Snowflake SQL using EXPLAIN.
    Does not execute the query.
    """
    snowflake_db, db = get_database_type(db_type="snowflake")
    try:
        with snowflake_db.connect() as conn:
            result = conn.execute(text(f"EXPLAIN USING TEXT {sql_query}"))
            rows = result.fetchall()

        return {"valid": True, "explain": [row[0] for row in rows]}

    except Exception as e:
        return {"valid": False, "message": str(e)}


@tool
def execute_snowflake_sql(sql_query: str) -> dict:
    """
    Execute Snowflake SQL and return rows + columns + server-computed totals.
    Fetches all rows for accurate totals; only the first 30 rows are returned
    as display_rows to keep context manageable.
    """
    snowflake_db, db = get_database_type(db_type="snowflake")
    try:
        logger.info("execution for snowflake")
        with snowflake_db.connect() as conn:
            result = conn.execute(text(sql_query))
            all_rows = result.fetchall()
            columns = list(result.keys())

        display_rows = all_rows[:30]

        # Compute numeric column totals from ALL rows in Python so the LLM
        # never has to sum large numbers itself (prevents hallucinated totals).
        column_totals = {}
        for i, col in enumerate(columns):
            try:
                col_sum = sum(
                    row[i]
                    for row in all_rows
                    if row[i] is not None and isinstance(row[i], (int, float))
                )
                if col_sum != 0:
                    column_totals[col] = col_sum
            except (TypeError, IndexError):
                pass

        result_dict = {
            "columns": columns,
            "rows": display_rows,
            "total_row_count": len(all_rows),
            "display_row_count": len(display_rows),
        }
        if column_totals:
            result_dict["server_computed_totals"] = column_totals
            result_dict["totals_note"] = (
                f"Totals computed server-side from ALL {len(all_rows)} rows "
                f"(only {len(display_rows)} are shown in display_rows). "
                "Use these exact values for the Grand Total row — do not recompute."
            )

        return result_dict

    except Exception as e:
        return {"error": str(e)}


retriever_tool_MaterialFamily = create_retriever_tool(
    retriever_MaterialFamily,
    name="search_proper_nouns_MaterialFamily",
    description=(
        "Use this tool to look up valid MaterialFamily values for filtering. "
        "Input is an approximate or fuzzy MaterialFamily name, and it returns the most likely valid match."
        "The returned value should be used with SQL ILIKE operator for filtering."
    ),
)

retriever_tool_Make = create_retriever_tool(
    retriever_Make,
    name="search_proper_nouns_Make",
    description=(
        "Use this tool to look up valid Make values for filtering. "
        "Input is an approximate or fuzzy Make name, and it returns the most likely valid match."
        "The returned value should be used with SQL ILIKE operator for filtering."
    ),
)

retriever_tool_powerbi = Tool(
    name="search_powerbi_dashboard",
    func=search_powerbi_dashboard,
    description=(
        "Search for Power BI dashboards. Returns report name and URL. "
        "ALWAYS format output as markdown: 📊 [ReportName](URL) "
        "Example: 📊 [Sales Dashboard](https://app.powerbi.com/...) "
        "User will only see 'Sales Dashboard' as blue clickable text."
    ),
)


def get_full_prompt(user_query, chat_history, role=None):
    """
    Params:
        user_query -> User input from FE
        chat_history -> For previous historical answer
    """
    current_date = datetime.datetime.now().strftime("%d")
    current_month = datetime.datetime.now().strftime("%m")
    current_year = datetime.datetime.now().strftime("%Y")

    # Build role-prefixed query used for all internal LLM calls (domain detection,
    # table selection). The clean user_query is kept for the user-facing prompt.
    _role_prefix_map = {
        "retail_allocation_planner": "I am a Retail Allocation Planner. ",
        "demand_planner": "I am a Demand Planner. ",
    }
    schema_query = (_role_prefix_map[role] + user_query) if role in _role_prefix_map else user_query

    # Follow-up fix: domain detection and table selection previously only saw the
    # current-turn text, so terse follow-ups (e.g. "give same for fold 8 ultra")
    # carried no domain signal and routing failed. Prepend the last user turn so
    # these steps can inherit context (device/topic/keywords) from the prior message.
    last_user_msg = ""
    for _msg in reversed(chat_history or []):
        if isinstance(_msg, dict) and _msg.get("role") == "user":
            last_user_msg = _msg.get("content", "") or ""
            break
    routing_query = f"{last_user_msg}\n{schema_query}" if last_user_msg else schema_query

    _role_domain_map = {
        "retail_allocation_planner": [{"domain": "config_retail_t_life", "sub_domain": "retail"}],
        "demand_planner": [{"domain": "config_inventory", "sub_domain": "retail_allocation"}],
    }
    if role in _role_domain_map:
        domain_key = _role_domain_map[role]
    else:
        domain_index = load_domains_index()
        domain_key = detect_domains(routing_query, domain_index)
    if not domain_key:
        return "No matching domain found"
    domain_data = load_multiple_domains(domain_key)
    example = extract_examples_multi(domain_data)
    if not example:
        raise ValueError("Few Shot Prompt not found")
    context, tables = build_context_multi(domain_data)
    logger.info("print the tables %s", tables)
    if not tables:
        raise ValueError("Relevant tables not found")
    table_schema, current_db = query_database_schema.invoke(
        {"user_question": routing_query, "table": tables}
    )
    context += f"\n\nDATABASE SCHEMA:\n{table_schema}"
    final_prompt = build_prompt(schema_query, context)
    example_prompt = PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    )

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        example, embeddings, FAISS, k=7, input_keys=["input"]
    )
    request_few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        input_variables=["input", "qry"],
        prefix=context,
        suffix="\nUser input: {input}\nSQL query:\n",
    )

    formatted_examples = request_few_shot_prompt.format(
        input=final_prompt,
        qry=user_query,
    )
    context_instructions = """
           CONVERSATIONAL CONTEXT INSTRUCTIONS:
           You MUST maintain context from previous messages in the conversation to handle follow-up questions effectively.

           **Context Retention Rules:**
           1. When a user asks a follow-up question that references previous results (e.g., "in these stores", "for those devices", "from that data", "which of these", "top 5 from above"), you MUST use the context from previous queries and results.
           2. If the user's question contains pronouns or references like "these", "those", "them", "it", "the above", "from this", look at the previous conversation to understand what they are referring to.
           3. When building follow-up queries, incorporate relevant filters and conditions from previous queries to maintain continuity.
           4. If a user asks about a subset of previous results (e.g., "Which of these 5 stores are in California?"), first identify the entities (store names, device names, etc.) from the previous response, then apply the new filter.

           **Follow-up Query Examples:**
           - User: "What are iPhone sales in December?" -> You query iPhone sales for December
           - User: "Give me the top 5 stores" -> You query top 5 stores for iPhone sales in December (maintaining iPhone + December context)
           - User: "Which of these are in the West region?" -> You filter the 5 stores by West region (maintaining the store list from previous query)

           **How to Handle Context:**
           - Review the conversation history provided to understand the full context
           - Extract key entities (devices, time periods, regions, stores, etc.) from previous queries and responses
           - When the current query is ambiguous, use the conversation context to resolve it
           - If context is truly unclear, ask a clarifying question before proceeding

           **DATA ACCURACY RULES (MANDATORY):**
           1. NEVER add a LIMIT, TOP, ROWNUM, or FETCH FIRST clause to any SQL query. Display row limits are handled in the response formatting, not in SQL. Always retrieve the complete result set so that totals are accurate.
           2. The SQL result includes a "server_computed_totals" field with totals computed from ALL rows in Python. When present, you MUST use those exact values for the Grand Total row — do not recompute, estimate, or sum the visible rows yourself.
           3. NEVER add a window aggregate column (e.g., SUM() OVER (), COUNT() OVER()) that repeats a grand total or running total alongside every data row, unless the user explicitly asks for a running total or cumulative sum column.
           4. KEY INSIGHTS (MANDATORY FOR SALES RESPONSES): Whenever your response includes a sales data table (data from BOT_DeviceSales or BOT_ACCESSORYSALES_V2), you MUST append a '### Key Insights' section immediately after the table. This section is NON-NEGOTIABLE and must never be omitted. Include: (a) Total Device Sales using the server_computed_totals grand total, (b) Top-Selling Device/SKU, and (c) 1-3 additional meaningful trends from the data.
"""
    full_system_content = formatted_examples + context_instructions
    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=full_system_content),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{user_input}"),
        ]
    )
    return full_prompt.format(
        user_input=schema_query,
        chat_history=chat_history,
    )


SNOWFLAKE_TOOLS = [
    validate_snowflake_sql,
    execute_snowflake_sql,
    retriever_tool_powerbi,
    retriever_tool_Make,
    retriever_tool_MaterialFamily,
]


def count_tokens(messages):
    encoding = tiktoken.encoding_for_model(llm_name)
    total_tokens = 0
    for message in messages:
        total_tokens += len(encoding.encode(message["content"]))
    return total_tokens


def convert_langchain_to_openai(messages):
    """
    Convert conversation history messages to OpenAI format.
    Handles multiple input formats:
    1. LangChain serialized format: {id: [..., "HumanMessage"], kwargs: {content: "..."}}
    2. Plain dict format: {role: "user", content: "..."}
    3. LangChain message objects with .content attribute
    """
    openai_messages = []
    if not messages:
        return openai_messages

    for message in messages:
        try:
            content = None
            role = None

            # Format 1: LangChain serialized format {id: [...], kwargs: {content: ...}}
            if isinstance(message, dict) and "kwargs" in message and "id" in message:
                content = message["kwargs"].get("content", "")
                msg_id = message.get("id", [])
                # id is typically a list like ["langchain_core", "messages", "HumanMessage"]
                id_str = str(msg_id) if isinstance(msg_id, list) else msg_id
                if "HumanMessage" in id_str:
                    role = "user"
                elif "AIMessage" in id_str:
                    role = "assistant"
                else:
                    role = "system"

            # Format 2: Plain dict format {role: "user", content: "..."}
            elif isinstance(message, dict) and "role" in message:
                content = message.get("content", "")
                role = message.get("role", "user")

            # Format 3: LangChain message object with .content attribute
            elif hasattr(message, "content"):
                content = message.content
                # Determine role from class name
                class_name = type(message).__name__
                if "Human" in class_name:
                    role = "user"
                elif "AI" in class_name:
                    role = "assistant"
                else:
                    role = "system"

            # Format 4: Simple string (treat as user message)
            elif isinstance(message, str):
                content = message
                role = "user"

            else:
                continue

            if content and role:
                openai_messages.append({"role": role, "content": str(content)})

        except Exception as e:
            logger.debug(f"[DEBUG] Error processing message: {e}, skipping message")
            continue

    return openai_messages


def get_executor(qry, hist, sID, hashID, threadID, queryID, role=None, progress_cb=None):
    global CURRENT_DB
    max_history_tokens = 100000
    messages = convert_langchain_to_openai(hist)
    config = {"configurable": {"thread_id": f"Thread-{threadID}"}}
    tools = SNOWFLAKE_TOOLS

    _t = time.perf_counter()
    full_prompt = get_full_prompt(qry, messages, role=role)
    logger.info(
        "[SQL timing] get_full_prompt (table-select LLM + schema fetch)=%.2fs",
        time.perf_counter() - _t,
    )
    # Stage 2: done finding the dataset; the agent is about to write + run the SQL.
    if progress_cb:
        progress_cb("run")

    agent = create_react_agent(azure_llm, tools, prompt=full_prompt)

    all_steps = []
    final_result = None
    _format_signaled = False

    _t = time.perf_counter()
    for step in agent.stream(
        {"messages": messages}, config=config, stream_mode="values"
    ):
        all_steps.append(step)
        final_result = step
        # Stage 3: once a tool has returned results, the model is formatting the answer.
        if progress_cb and not _format_signaled:
            try:
                msgs = step.get("messages") if isinstance(step, dict) else None
                last = msgs[-1] if msgs else None
                if last is not None and type(last).__name__ == "ToolMessage":
                    _format_signaled = True
                    progress_cb("format")
            except Exception:
                pass
    logger.info(
        "[SQL timing] react agent.stream=%.2fs (%d steps / LLM+tool round-trips)",
        time.perf_counter() - _t, len(all_steps),
    )

    return final_result, all_steps, CURRENT_DB
