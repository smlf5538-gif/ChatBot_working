from API import _tls_bootstrap  # noqa: F401  # MUST be first — combined CA bundle before Azure imports
 
import os
import re
import json
import time
import asyncio
from collections import Counter as _Counter
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Annotated
from urllib.parse import quote, urlsplit, urlunsplit
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from typing_extensions import TypedDict as _TypedDict
from abc import ABC, abstractmethod
from dataclasses import dataclass
from azure.core.credentials import AzureKeyCredential
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.core.pipeline.transport import RequestsTransport
from azure.search.documents import SearchClient
from datetime import datetime, timedelta, timezone
from API.helpers.helper import get_executor
from dotenv import load_dotenv
from azure.search.documents.models import VectorizedQuery
from openai import AzureOpenAI, AsyncAzureOpenAI
import logging
import pathlib as _pathlib
 
from API.onboarding.retrieval import onboarding_search, data_search, extract_team, ONBOARDING_TEAMS
from API import replenishment_info as _repl  # Replenishment Information card (Retail Allocation Planner)
 
load_dotenv()
 
logger = logging.getLogger(__name__)
 
# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("LLM_OPENAI_API_VERSION")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
LLM_Model = os.getenv("LLM_MODEL")
 
# Azure Search Configuration
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")
CONTRACT_SEARCH_INDEX_NAME = os.getenv("CONTRACT_SEARCH_INDEX_NAME")
CONTRACT_BLOB_BASE_URL = os.getenv("CONTRACT_BLOB_BASE_URL", "").rstrip("/")
_BLOB_ACCOUNT_NAME = os.getenv("azure_storage_account_name", "")
_BLOB_ACCOUNT_KEY = os.getenv("azure_storage_account_key", "")
_BLOB_CONTAINER = os.getenv("container_name", "psc-data-analytics")
VECTOR_DIMENSIONS = os.getenv("AZURE_SEARCH_VECTOR_DIMENSIONS")
 
# Convert VECTOR_DIMENSIONS to int if set
if VECTOR_DIMENSIONS:
    VECTOR_DIMENSIONS = int(VECTOR_DIMENSIONS)
 
 
def _make_blob_sas_url(blob_name: str, page_num=None) -> str:
    """Generate a 2-hour read SAS URL for a blob. Returns empty string if credentials missing."""
    if not _BLOB_ACCOUNT_NAME or not _BLOB_ACCOUNT_KEY:
        return ""
    blob_name = blob_name.lstrip("/")
    try:
        sas_token = generate_blob_sas(
            account_name=_BLOB_ACCOUNT_NAME,
            container_name=_BLOB_CONTAINER,
            blob_name=blob_name,
            account_key=_BLOB_ACCOUNT_KEY,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.now(timezone.utc) + timedelta(hours=2),
        )
        encoded_name = quote(blob_name, safe="/:@!$&*+,;=")
        url = f"https://{_BLOB_ACCOUNT_NAME}.blob.core.windows.net/{_BLOB_CONTAINER}/{encoded_name}?{sas_token}"
        if page_num:
            url += f"#page={page_num}"
        return url
    except Exception as e:
        logger.error("[CONTRACT URL] SAS generation failed: %s", e)
        return ""
 
 
# SQL Agent
 
 
def get_sql_agent_response(
    query: str,
    conversation_history: List[Dict] = None,
    session_id: str = None,
    hash_id: str = None,
    thread_id: str = None,
    query_id: str = None,
    role: str = None,
    progress_cb=None,
) -> Dict[str, Any]:
    """
    Execute SQL Agent query using the actual implementation from helpers.
    """
    if conversation_history is None:
        conversation_history = []
    if session_id is None:
        session_id = "default_session"
    if hash_id is None:
        hash_id = "default_hash"
    if thread_id is None:
        thread_id = "default_thread"
    if query_id is None:
        query_id = f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
 
    final_result = None
    try:
        final_result, all_steps, database_type = get_executor(
            qry=query,
            hist=conversation_history,
            sID=session_id,
            hashID=hash_id,
            threadID=thread_id,
            queryID=query_id,
            role=role,
            progress_cb=progress_cb,
        )
        if (
            final_result
            and isinstance(final_result, dict)
            and "messages" in final_result
        ):
            messages = final_result["messages"]
 
            if messages and len(messages) > 0:
                last_message = messages[-1]
 
                if hasattr(last_message, "content"):
                    answer = last_message.content
                else:
                    answer = str(last_message)
 
                updated_history = conversation_history + messages
 
                return {
                    "query": query,
                    "answer": answer,
                    "step_count": len(all_steps),
                    "steps": all_steps,
                    "conversation_history": updated_history,
                    "final_result": final_result,
                    "database_type": database_type,
                }
 
        return {
            "query": query,
            "answer": "No response generated from SQL Agent.",
            "step_count": 0,
            "steps": [],
            "conversation_history": conversation_history,
            "final_result": final_result,
        }
 
    except Exception as e:
        logger.error("Error executing Snowflake query: %s", e, exc_info=True)
        return {
            "query": query,
            "answer": f"Error executing Snowflake query: {str(e)}",
            "step_count": 0,
            "steps": [],
            "conversation_history": conversation_history,
            "final_result": None,
        }
 
 
# Azure OpenAI clients
 
# Sync client (used for embeddings)
client = AzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
)
 
# Async client (streaming completions)
async_client = AsyncAzureOpenAI(
    api_key=OPENAI_API_KEY,
    api_version=OPENAI_API_VERSION,
    azure_endpoint=OPENAI_ENDPOINT,
)
 
# Azure Search client
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
)
 
contract_search_client = (
    SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=CONTRACT_SEARCH_INDEX_NAME,
        credential=AzureKeyCredential(AZURE_SEARCH_API_KEY),
    )
    if CONTRACT_SEARCH_INDEX_NAME
    else None
)
 
 
# ---------------------------------------------------------------------------
# Follow-up context threading (inlined from followup_context.py)
# ---------------------------------------------------------------------------
#
# Scenarios handled:
#   Scenario 1: q1m1 → q2m1 → q3m1
#       Each inherits its immediate same-mode predecessor (-1 parent)
#
#   Scenario 2: q1m1, q2m2, q3m2
#       q3 gets q2's context only; q1 excluded (different mode, chain broken)
#
#   Scenario 3: q0m2, q1m1, q2m2, q3m2
#       q3 gets q2's context only — __mode__ tag from q1m1 breaks the chain
#       even though q0 and q2/q3 share the same mode
#
#   Scenario 4: same mode, topically unrelated questions
#       LLM detects INDEPENDENT → agent answers fresh with no history
# ---------------------------------------------------------------------------
 
_FOLLOWUP_DETECT_PROMPT = """You are a query classifier and rewriter for InsightBot, a T-Mobile internal chatbot.
 
You will be shown:
  - The PREVIOUS question and answer (the parent turn)
  - The CURRENT question
 
Step 1 — Decide: is the CURRENT question a FOLLOW-UP to the previous one, or is it an INDEPENDENT new question?
 
Check INDEPENDENT first — if either rule below applies, it is INDEPENDENT, full stop,
even if it mentions the same device/brand as the previous question:
 
INDEPENDENT means EITHER of:
- It names its own explicit metric/ask-item (e.g. "sales", "ESD", "inventory",
  "forecast", "activations", "promotions", "upgrades", "6-week average", etc.) —
  ANY explicit metric in the current question makes it independent, regardless of
  whether the entity/device is the same as, or different from, the previous turn.
  EXCEPTION: if the current question names ONLY the metric — whether bare (e.g.
  "forecast") or with a backward pronoun (e.g. "inventory for the same") — with NO new
  device of its own, and the previous question/answer already named one specific device,
  treat it as a FOLLOW-UP for that metric on that same device instead — go to the
  FOLLOW-UP rules below, and in the rewrite carry that device forward from the previous
  question (e.g. previous "sales for iPhone 17", current "inventory for the same" →
  "inventory for iPhone 17").
- It is a generic device/brand overview or status ask that NAMES ITS OWN device/brand, with
  NO metric named (e.g. "overview of Galaxy Fold8", "how is iPhone doing", "tell me about
  Motorola", "status of Galaxy S26", "is iPhone in stock") — this is always its own
  independent intent (a device overview request), never a continuation of whatever metric
  the previous question asked about.
  EXCEPTION: if the overview/status ask names NO device of its own — only a backward
  pronoun (e.g. "device overview for the same", "give me the overview for it") — and the
  previous question/answer already named one specific device, treat it as a FOLLOW-UP
  instead — go to the FOLLOW-UP rules below, and in the rewrite carry that device forward
  from the previous question (e.g. previous "launch performance of Pixel 11 Pro Fold",
  current "device overview for the same" → "device overview for Pixel 11 Pro Fold").
- It states a NEW specific identifier in the CURRENT question itself — a
  material number, SKU, order ID, or similar numeric/alphanumeric code (e.g.
  "what tiers is 195950476552 associated with", "lookup 8801643123456 for
  this") — this ALWAYS makes it independent, even if the sentence also
  contains a pronoun like "this"/"it"/"that", because the identifier stated
  IN THE CURRENT QUESTION fully specifies the subject on its own and overrides
  the pronoun. This rule only applies when the identifier itself appears in
  the current question — do NOT apply it just because the previous turn
  mentioned an identifier.
 
Only if NEITHER rule above applies, consider FOLLOW-UP:
FOLLOW-UP means:
- It refers back to the previous answer with no metric AND no new identifier
  of its own (e.g. "what about X", "same for Y", "and Z?", "give me that for
  Y", "give me the inventory for the same") — assume it wants the SAME
  ask/metric/identifier the previous question established, applied to the new
  object. Example: previous turn named a material/device, current question is
  just "give me the inventory for the same" — no new identifier is stated, so
  this stays FOLLOW-UP and inherits the identifier from the previous turn.
- It continues the same topic/entity without restating it (elliptical reference)
- It asks for a comparison, elaboration, or variation of what was just answered
- Pronouns or demonstratives clearly point back ("it", "that", "those", "the
  same") and no new identifier/metric is named in the current question
 
Step 2 — If FOLLOWUP, rewrite the CURRENT question into a fully self-contained
standalone question: carry over the ask/metric, timeframe, and any other qualifiers
from the previous question, but substitute in whatever the current question is
explicitly changing (e.g. a different device/brand/region). Keep it concise and
in the same style as a user would phrase it — do not add explanation.
 
When carrying over a device name from the previous question into the rewritten
question, ALWAYS use the bare device/model name and DROP the OEM/brand prefix
(e.g. "Google", "Samsung", "Apple", "Motorola") — e.g. if the previous question
was about "Google Pixel 11 Pro XL", write "sales for Pixel 11 Pro XL", NOT
"sales for Google Pixel 11 Pro XL". Similarly "Samsung Galaxy Z Fold8" becomes
"Galaxy Z Fold8" and "Apple iPhone 18 Pro" becomes "iPhone 18 Pro" Downstream data lookups match on the bare model name only.
 
Previous question: {prev_q}
Previous answer (truncated): {prev_a}
 
Current question: {curr_q}
 
Output EXACTLY one of:
  INDEPENDENT
  <the rewritten self-contained question, if FOLLOWUP>
 
No explanation. No quotes. No preamble. Just the single word INDEPENDENT, or the rewritten question."""
 
 
def _parse_mode_tag(content: str) -> Tuple[str, Optional[str]]:
    """
    Mode tags are injected as system messages:
      "__mode__:<mode>"               (no agent info — legacy/older history)
      "__mode__:<mode>|agent:<agent>" (also records which agent actually answered —
                                        e.g. mode=sql but agent=device_health, since
                                        device_health/inventory_health have no tab
                                        of their own and are reached as a redirect
                                        from the sql tab)
    Returns (tag_mode, tag_agent) — tag_agent is None if not present.
    """
    body = content[len("__mode__:"):]
    tag_mode, _, tag_agent = body.partition("|agent:")
    return tag_mode.strip().lower(), (tag_agent.strip() or None)
 
 
def _get_same_mode_parent(
    conversation_history: List[Dict],
    current_mode: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    """
    Walk history backwards to find the single most-recent same-mode Q&A pair.
 
    Rules:
    - Only returns the LAST same-mode exchange (the -1 parent). No further.
    - Stops immediately if a DIFFERENT __mode__ system tag is encountered first.
      This enforces Scenarios 2 & 3 — a mode change in between breaks the chain.
    - Mode tags are injected as system messages: {"role": "system", "content": "__mode__:sql"}
 
    Returns (user_question, assistant_answer, parent_agent) or None. parent_agent is
    the actual agent that answered the parent turn (e.g. "device_health"), which may
    differ from current_mode — None if the tag predates agent-tracking or is absent.
    """
    if not conversation_history or not current_mode:
        return None
 
    i = len(conversation_history) - 1
    while i >= 0:
        msg = conversation_history[i]
        role = (msg.get("role") or "").lower()
        content = msg.get("content", "")
 
        # Hit a mode tag — check if it matches current mode
        if role == "system" and isinstance(content, str) and content.startswith("__mode__:"):
            tag_mode, _ = _parse_mode_tag(content)
            if tag_mode != current_mode.lower():
                # Chain broken by a different mode (Scenarios 2 & 3)
                logger.debug(
                    "[followup] Chain broken: found mode tag '%s' != current '%s'",
                    tag_mode, current_mode,
                )
                return None
            # Same mode tag — keep scanning backwards for the actual Q&A pair
            i -= 1
            continue
 
        if role in ("assistant", "ai"):
            # Found an assistant turn — pair it with the nearest preceding user turn
            assistant_ans = content
            user_q = None
            parent_agent = None
            j = i - 1
            while j >= 0:
                prev = conversation_history[j]
                prev_role = (prev.get("role") or "").lower()
                prev_content = prev.get("content", "")
                if prev_role in ("user", "human"):
                    user_q = prev_content
                    # The tag for THIS triplet sits one slot further back (structure
                    # is [tag, user, assistant]) — read it to learn which agent
                    # actually answered, if recorded.
                    if j - 1 >= 0:
                        tag_msg = conversation_history[j - 1]
                        tag_role = (tag_msg.get("role") or "").lower()
                        tag_content = tag_msg.get("content", "")
                        if tag_role == "system" and isinstance(tag_content, str) and tag_content.startswith("__mode__:"):
                            _, parent_agent = _parse_mode_tag(tag_content)
                    break
                # If we hit a mode tag while looking for the user turn, check it
                if prev_role == "system" and isinstance(prev_content, str) and prev_content.startswith("__mode__:"):
                    tag_mode, _ = _parse_mode_tag(prev_content)
                    if tag_mode != current_mode.lower():
                        logger.debug(
                            "[followup] Chain broken at user search: mode tag '%s' != '%s'",
                            tag_mode, current_mode,
                        )
                        return None
                j -= 1
 
            if user_q is not None:
                logger.debug(
                    "[followup] Found same-mode parent — Q: '%s...' A: '%s...' agent=%s",
                    user_q[:60], assistant_ans[:60], parent_agent,
                )
                return user_q, assistant_ans, parent_agent
            return None  # assistant turn with no preceding user — stop
 
        i -= 1
 
    return None
 
 
async def _llm_detect_followup(
    current_query: str,
    parent_qa: Tuple[str, str],
) -> Tuple[bool, str]:
    """
    Detects whether current_query is a follow-up to parent_qa, and if so, rewrites
    it into a self-contained standalone question (carrying over the ask/metric from
    the parent question, substituting in whatever the current question changes —
    e.g. "give me same for iphone" -> "6-week average weekly sales for iPhone").
 
    Single LLM call combining classification + rewrite (max ~120 tokens output).
    Fails OPEN → returns (False, current_query) — treat as independent — on any error.
 
    Returns:
        (is_followup, resolved_query)
        If FOLLOWUP:    resolved_query is the rewritten standalone question.
        If INDEPENDENT or on error: resolved_query == current_query.
    """
    prev_q, prev_a = parent_qa
    prev_a_truncated = prev_a[:400] + ("..." if len(prev_a) > 400 else "")
 
    prompt = _FOLLOWUP_DETECT_PROMPT.format(
        prev_q=prev_q[:300],
        prev_a=prev_a_truncated,
        curr_q=current_query[:300],
    )
 
    try:
        response = await async_client.chat.completions.create(
            model=LLM_Model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=120,
        )
        raw = response.choices[0].message.content.strip()
        logger.info("[followup] LLM verdict for '%s...': %s", current_query[:50], raw[:120])
        if not raw or raw.strip().upper() == "INDEPENDENT":
            return False, current_query
        return True, raw
    except Exception as e:
        logger.error("[followup] LLM detect call failed: %s — treating as INDEPENDENT", e)
        return False, current_query  # fail open
 
 
def _build_parent_context_messages(parent_qa: Tuple[str, str]) -> List[Dict]:
    """
    Wrap the parent Q&A as exactly 2 messages to inject as the agent's history.
    This gives the agent the -1 parent context and nothing older.
    """
    prev_q, prev_a = parent_qa
    return [
        {"role": "user",      "content": prev_q},
        {"role": "assistant", "content": prev_a},
    ]
 
 
def _inject_mode_tag(
    conversation_history: List[Dict],
    mode: str,
    user_query: str,
    assistant_answer: str,
    agent_used: Optional[str] = None,
) -> List[Dict]:
    """
    Append a mode-tagged Q&A triplet to history:
      [system  __mode__:<mode>[|agent:<agent_used>]  ]
      [user    <query>          ]
      [assistant <answer>       ]
 
    The __mode__ system tag is the boundary marker that lets _get_same_mode_parent()
    detect mode changes and enforce Scenarios 2 & 3 correctly. agent_used additionally
    records which agent actually answered — needed because device_health/inventory_health
    have no tab of their own and are reached as a redirect from the "sql" tab, so a
    follow-up to one of those answers should route back to the same card, not to
    whatever the tab's own default agent is.
    """
    tag = f"__mode__:{mode}"
    if agent_used:
        tag += f"|agent:{agent_used}"
    updated = list(conversation_history)
    updated.append({"role": "system",    "content": tag})
    updated.append({"role": "user",      "content": user_query})
    updated.append({"role": "assistant", "content": assistant_answer})
    return updated
 
 
async def _resolve_followup_context(
    current_query: str,
    current_mode: str,
    conversation_history: List[Dict],
) -> Tuple[bool, List[Dict], str, Optional[str]]:
    """
    Main follow-up resolver. Called once per query in run_orchestrator_query.
 
    Returns:
        (is_followup, effective_history, resolved_query, parent_agent)
 
    If FOLLOWUP:    effective_history = [parent_user_msg, parent_assistant_msg]
                    (exactly the -1 parent — 2 messages)
                    resolved_query = rewritten, self-contained version of current_query
                    parent_agent = the agent that answered the parent turn (e.g.
                    "device_health"), so the router can send the follow-up back to
                    the same agent instead of the tab's default — None if unknown.
    If INDEPENDENT: effective_history = [], resolved_query = current_query (unchanged),
                    parent_agent = None
    """
    parent = _get_same_mode_parent(conversation_history, current_mode)
 
    if parent is None:
        logger.debug("[followup] No same-mode parent found → INDEPENDENT")
        return False, [], current_query, None
 
    prev_q, prev_a, parent_agent = parent
    is_followup, resolved_query = await _llm_detect_followup(current_query, (prev_q, prev_a))
 
    if is_followup:
        logger.info(
            "[followup] Detected FOLLOWUP for mode='%s' (parent_agent=%s) — rewrote '%s...' -> '%s...'",
            current_mode, parent_agent, current_query[:50], resolved_query[:80],
        )
        return True, _build_parent_context_messages((prev_q, prev_a)), resolved_query, parent_agent
    else:
        logger.info(
            "[followup] Detected INDEPENDENT question on mode='%s'", current_mode
        )
        return False, [], current_query, None
 
 
# ---------------------------------------------------------------------------
# Base classes & types
# ---------------------------------------------------------------------------
 
 
class AgentExecutionError(Exception):
    """Raised when an orchestrator agent fails during execution."""
 
 
class AgentOutput(TypedDict, total=False):
    answer: str
    steps: List[Any]
    step_count: int
    conversation_history: List[Dict]
    sources: List[Dict[str, Any]]
    retrieved_docs: List[Dict[str, Any]]
    final_result: Any
 
 
@dataclass
class AgentContext:
    """Container for information an agent needs while making decisions."""
 
    query: str
    state: "OrchestratorState"
    conversation_history: List[Dict]
    # The literal text the user/frontend sent THIS turn, before any follow-up
    # rewrite. Defaults to `query` when not otherwise available (e.g. legacy
    # callers) so existing code that ignores this field keeps working unchanged.
    raw_query: str = ""
 
    def __post_init__(self):
        if not self.raw_query:
            self.raw_query = self.query
 
 
class OrchestratorAgent(ABC):
    """Common interface every orchestrator agent must implement."""
 
    def __init__(self, name: str, priority: int = 100) -> None:
        self.name = name
        self.priority = priority
 
    @abstractmethod
    async def run_stream(self, context: AgentContext):
        """Execute the agent logic with streaming and yield chunks."""
        yield  # pragma: no cover
 
 
# Embedding & Search helpers
 
 
def get_embedding(text: str) -> list[float]:
    """Convert text to embedding vector using Azure OpenAI."""
    embedding_kwargs = {"model": EMBEDDING_MODEL_NAME, "input": str(text)}
    if VECTOR_DIMENSIONS is not None:
        embedding_kwargs["dimensions"] = VECTOR_DIMENSIONS
 
    response = client.embeddings.create(**embedding_kwargs)
    embeddings = response.data[0].embedding
    return list(map(float, embeddings))
 
 
def similarity_search(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    logger.debug("[SEARCH] Starting similarity search for query: '%s...'", query[:50])
    logger.debug(
        "[SEARCH] Using index: %s, endpoint: %s",
        AZURE_SEARCH_INDEX_NAME,
        AZURE_SEARCH_ENDPOINT,
    )
 
    try:
        query_vector = get_embedding(str(query))
        logger.debug(
            "[SEARCH] Embedding generated, vector length: %d", len(query_vector)
        )
    except Exception as e:
        logger.error("[SEARCH] Failed to generate embedding: %s", e)
        return []
 
    vector_query = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=top_k, fields="contentVector"
    )
 
    try:
        results = search_client.search(
            search_text=None,
            vector_queries=[vector_query],
            select=["content", "title", "sourcefile", "id"],
        )
        docs = list(results)
        logger.debug("[SEARCH] Retrieved %d documents from Azure Search", len(docs))
    except Exception as e:
        logger.error("[SEARCH] Azure Search query failed: %s", e)
        return []
 
    retrieved_docs = []
    for idx, result in enumerate(docs, 1):
        doc = {
            "rank": idx,
            "content": result.get("content", ""),
            "title": result.get("title", "Unknown"),
            "source": result.get("sourcefile", "Unknown"),
            "id": result.get("id", "Unknown"),
            "score": result.get("@search.score", 0.0),
        }
        retrieved_docs.append(doc)
 
    if retrieved_docs:
        logger.debug(
            "[SEARCH] Top result - Title: %s, Score: %s",
            retrieved_docs[0]["title"],
            retrieved_docs[0]["score"],
        )
    else:
        logger.debug("[SEARCH] No documents retrieved!")
 
    return retrieved_docs
 
 
def contract_similarity_search(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Two-stage retrieval based on pipeline_bot_enhanced.py approach.
    """
    logger.debug("[CONTRACT SEARCH] Starting search for query: '%s...'", query[:50])
    logger.debug("[CONTRACT SEARCH] Using index: %s", CONTRACT_SEARCH_INDEX_NAME)
 
    if contract_search_client is None:
        logger.error("[CONTRACT SEARCH] Contract search client not configured.")
        return []
 
    try:
        query_vector = get_embedding(str(query))
        logger.debug(
            "[CONTRACT SEARCH] Embedding generated, vector length: %d",
            len(query_vector),
        )
    except Exception as e:
        logger.error("[CONTRACT SEARCH] Failed to generate embedding: %s", e)
        return []
 
    program_code = _extract_program_code(query)
    listy = _is_list_question(query)
    comparey = _is_compare_question(query)
    oems = _detect_oems_in_query(query)
    multi_oem = len(oems) >= 2
 
    logger.debug(
        "[CONTRACT SEARCH] program_code=%s listy=%s comparey=%s oems=%s",
        program_code,
        listy,
        comparey,
        oems,
    )
 
    anchor_text = program_code if program_code else query
    stageA_top = 100 if (program_code or listy or comparey) else 50
 
    try:
        hitsA = _dedupe_contract_hits(
            _raw_contract_search(anchor_text, query_vector, top=stageA_top)
        )
    except Exception as e:
        logger.error("[CONTRACT SEARCH] Stage A failed: %s", e)
        return []
 
    logger.debug("[CONTRACT SEARCH] Stage A returned %d unique hits", len(hitsA))
 
    if oems and not program_code and not listy and not comparey:
        final_raw = hitsA
        logger.debug(
            "[CONTRACT SEARCH] Broad OEM evidence path: using global Stage A (%d hits, no OEM filter)",
            len(hitsA),
        )
 
    elif oems:
        all_hits: List[dict] = []
 
        for oem in oems:
            oem_hitsA = [
                h for h in hitsA if _blob_path_matches_oem(h.get("blob_path", ""), oem)
            ]
 
            if not oem_hitsA:
                try:
                    tmp = _raw_contract_search(query, query_vector, top=150)
                    oem_hitsA = _dedupe_contract_hits(
                        [
                            h
                            for h in tmp
                            if _blob_path_matches_oem(h.get("blob_path", ""), oem)
                        ]
                    )
                except Exception as e:
                    logger.error(
                        "[CONTRACT SEARCH] OEM fallback for %s failed: %s", oem, e
                    )
 
            n_blobs = 5 if listy else 2
            top_blobs = _top_blob_paths(oem_hitsA, n=n_blobs)
            expanded_oem: List[dict] = []
            for bp in top_blobs:
                try:
                    filt = f"blob_path eq '{_escape_odata(bp)}'"
                    expanded_oem.extend(
                        _raw_contract_search(
                            query, query_vector, top=120, filter_expr=filt
                        )
                    )
                except Exception as e:
                    logger.error(
                        "[CONTRACT SEARCH] Stage B blob %s OEM %s failed: %s",
                        bp,
                        oem,
                        e,
                    )
            expanded_oem = _dedupe_contract_hits(expanded_oem) or oem_hitsA
            _pipe_per_oem = _CONTRACT_MAX_CONTEXT_CHUNKS // max(1, len(oems))
            _pipe_top_k_per_oem = 50 // max(1, len(oems))
            all_hits.extend(expanded_oem[: max(_pipe_per_oem, _pipe_top_k_per_oem)])
 
        final_raw = _dedupe_contract_hits(all_hits)
 
    elif program_code or listy:
        top_blobs = _top_blob_paths(hitsA, n=2)
        expanded: List[dict] = []
        for bp in top_blobs:
            try:
                filt = f"blob_path eq '{_escape_odata(bp)}'"
                exp_top = 120 if listy else max(top_k, 75)
                expanded.extend(
                    _raw_contract_search(
                        query, query_vector, top=exp_top, filter_expr=filt
                    )
                )
            except Exception as e:
                logger.error("[CONTRACT SEARCH] Stage B blob %s failed: %s", bp, e)
 
        expanded = _dedupe_contract_hits(expanded)
        final_raw = expanded if listy else expanded
        if not final_raw:
            final_raw = hitsA
 
    else:
        final_raw = hitsA
 
    final_raw = final_raw[:40]
 
    retrieved_docs = []
    for idx, result in enumerate(final_raw, 1):
        doc = _enrich_contract_hit(result)
        doc["rank"] = idx
        retrieved_docs.append(doc)
 
    if retrieved_docs:
        logger.debug(
            "[CONTRACT SEARCH] Top result - Title: %s, Score: %s",
            retrieved_docs[0]["title"],
            retrieved_docs[0]["score"],
        )
    else:
        logger.debug("[CONTRACT SEARCH] No documents retrieved!")
 
    return retrieved_docs
 
 
# ---------------------------------------------------------------------------
# Contract search helpers
# ---------------------------------------------------------------------------
 
_PROGRAM_CODE_RE = re.compile(r"\b\d{2}[A-Z]{2,4}\d{3}\b", re.IGNORECASE)
 
 
def _extract_program_code(q: str) -> Optional[str]:
    m = _PROGRAM_CODE_RE.search(q or "")
    return m.group(0).upper() if m else None
 
 
def _is_list_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(
        k in ql
        for k in [
            "list",
            "all programs",
            "which programs",
            "all devices",
            "which devices",
            "all skus",
            "which skus",
            "eligible products",
            "eligible devices",
            "eligible models",
            "list of",
        ]
    )
def _is_compare_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(
        k in ql for k in ["compare", "difference", "differ", "versus", "vs", "between"]
    )
 
 
_OEM_ALIASES: Dict[str, str] = {
    "samsung": "samsung",
    "galaxy": "samsung",
    "apple": "apple",
    "iphone": "apple",
    "motorola": "motorola",
    "google": "google",
    "pixel": "google",
    "lg": "lg",
    "oneplus": "oneplus",
    "tcl": "tcl",
    "revvl": "tcl",
}
 
_OEM_BLOB_FOLDERS: Dict[str, str] = {
    "samsung": "/samsung/",
    "apple": "/apple/",
    "motorola": "/motorola/",
    "google": "/google/",
    "lg": "/lg/",
    "oneplus": "/oneplus/",
    "tcl": "/tcl/",
}
 
 
def _detect_oems_in_query(query: str) -> List[str]:
    ql = query.lower()
    found: List[str] = []
    seen: set = set()
    for alias, oem in [
        ("apple", "apple"),
        ("iphone", "apple"),
        ("samsung", "samsung"),
        ("galaxy", "samsung"),
        ("motorola", "motorola"),
        ("google", "google"),
        ("pixel", "google"),
        ("lg", "lg"),
        ("oneplus", "oneplus"),
        ("tcl", "tcl"),
        ("revvl", "tcl"),
    ]:
        if alias in ql and oem not in seen:
            seen.add(oem)
            found.append(oem)
    return found
 
 
def _blob_path_matches_oem(blob_path: str, oem: str) -> bool:
    bp = (blob_path or "").lower()
    folder = _OEM_BLOB_FOLDERS.get(oem, f"/{oem}/")
    return folder in bp or folder.replace("/", "\\") in bp
 
 
def _escape_odata(s: str) -> str:
    return (s or "").replace("'", "''")
 
 
def _dedupe_contract_hits(hits: List[dict]) -> List[dict]:
    out: List[dict] = []
    seen: set = set()
    for h in hits:
        bp = h.get("blob_path", "")
        pg = h.get("page_num", -1)
        txt = h.get("content") or ""
        key = (bp, pg, txt[:200])
        if key in seen:
            continue
        seen.add(key)
        out.append(h)
    return out
 
 
def _top_blob_paths(hits: List[dict], n: int = 2) -> List[str]:
    paths = [h.get("blob_path") for h in hits if h.get("blob_path")]
    if not paths:
        return []
    counts = _Counter(paths)
    return [bp for bp, _ in counts.most_common(n)]
 
 
_CONTRACT_SELECT_FIELDS = [
    "id",
    "content",
    "blob_path",
    "program_name",
    "document_type",
    "oem",
    "promotion_year",
    "effective_date",
    "expiration_date",
    "signature_dates",
    "parties_json",
    "eligible_products_json",
    "incentive_type",
    "minimum_discount",
    "payment_terms",
    "termination_rights",
    "eligible_customers",
    "eligible_channels",
    "reporting_requirements",
    "funding_cap",
    "governing_agreement",
    "hierarchy_clause",
    "source",
    "page_num",
]
 
 
def _raw_contract_search(
    search_text: Optional[str],
    query_vector: list,
    top: int,
    filter_expr: Optional[str] = None,
) -> List[dict]:
    """Single hybrid (or pure-vector) call to the contract Azure Search index."""
    vq = VectorizedQuery(
        vector=query_vector, k_nearest_neighbors=top, fields="embedding"
    )
    results = contract_search_client.search(
        search_text=search_text,
        vector_queries=[vq],
        select=_CONTRACT_SELECT_FIELDS,
        top=top,
        filter=filter_expr,
    )
    return list(results)
 
 
def _enrich_contract_hit(result: dict) -> dict:
    """Build enriched doc dict from a raw Azure Search result."""
    structured_parts = []
    for field, label in [
        ("program_name", "Program Name"),
        ("document_type", "Document Type"),
        ("oem", "OEM"),
        ("promotion_year", "Promotion Year"),
        ("effective_date", "Effective Date"),
        ("expiration_date", "Expiration Date"),
        ("parties_json", "Parties"),
        ("incentive_type", "Incentive Type"),
        ("minimum_discount", "Minimum Discount"),
        ("payment_terms", "Payment Terms"),
        ("termination_rights", "Termination Rights"),
        ("eligible_customers", "Eligible Customers"),
        ("eligible_channels", "Eligible Channels"),
        ("reporting_requirements", "Reporting Requirements"),
        ("funding_cap", "Funding Cap"),
        ("governing_agreement", "Governing Agreement"),
        ("hierarchy_clause", "Hierarchy Clause"),
        ("eligible_products_json", "Eligible Products"),
    ]:
        val = result.get(field)
        if val:
            structured_parts.append(f"{label}: {val}")
 
    sig = result.get("signature_dates")
    if sig:
        structured_parts.append(f"Signature Date(s): {', '.join(sig)}")
 
    raw_content = result.get("content", "")
    if len(raw_content) > 2000:
        raw_content = raw_content[:2000] + "..."
 
    enriched = "\n".join(structured_parts)
    if raw_content.strip():
        enriched = (
            (enriched + "\n\nDocument Content:\n" + raw_content)
            if enriched
            else raw_content
        )
 
    blob_path = result.get("blob_path", "")
    title = result.get("program_name") or (
        blob_path.split("/")[-1] if blob_path else "Unknown"
    )
    source_raw = result.get("source") or blob_path or ""
    page_num = result.get("page_num")
 
    blob_name = blob_path.lstrip("/")
    if not blob_name and source_raw.startswith("https://"):
        parts = urlsplit(source_raw)
        path_parts = parts.path.lstrip("/").split("/", 1)
        blob_name = path_parts[1] if len(path_parts) == 2 else ""
    doc_url = _make_blob_sas_url(blob_name, page_num) if blob_name else ""
 
    return {
        "blob_path": blob_path,
        "content": enriched,
        "title": title,
        "source": source_raw or "Unknown",
        "url": doc_url,
        "page_num": page_num,
        "id": result.get("id", "Unknown"),
        "score": result.get("@search.score", 0.0),
    }
 
 
# ---------------------------------------------------------------------------
# Metadata-first contracts helpers
# ---------------------------------------------------------------------------
 
_CONTRACT_ALLOWED_FIELDS = {
    "eligible_products_json",
    "payment_terms",
    "funding_cap",
    "effective_date",
    "expiration_date",
    "signature_dates",
    "governing_agreement",
    "hierarchy_clause",
    "program_name",
    "parties_json",
    "reporting_requirements",
    "termination_rights",
    "minimum_discount",
    "incentive_type",
    "eligible_customers",
    "eligible_channels",
}
 
_CONTRACT_FIELD_ALIASES: Dict[str, List[str]] = {
    "eligible_products_json": [
        "eligible devices",
        "eligible device",
        "eligible products",
        "device list",
        "devices listed",
        "models",
        "model list",
        "skus",
        "sku list",
        "eligible models",
    ],
    "payment_terms": [
        "payment terms",
        "paid within",
        "net",
        "invoice",
        "remittance",
        "payment schedule",
    ],
    "funding_cap": [
        "funding cap",
        "cap",
        "budget",
        "max funding",
        "maximum spend",
        "top spend",
        "not to exceed",
        "nte",
        "maximum exposure",
        "spend cap",
    ],
    "effective_date": ["effective date", "start date", "begins", "commences"],
    "expiration_date": ["expiration date", "end date", "expires", "ends"],
    "governing_agreement": [
        "governing agreement",
        "master agreement",
        "supply agreement",
        "base agreement",
    ],
    "hierarchy_clause": [
        "hierarchy clause",
        "order of precedence",
        "conflict",
        "prevails",
        "shall govern",
    ],
}
 
 
def _is_count_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(
        k in ql
        for k in [
            "how many",
            "number of",
            "count",
            "total contracts",
            "total docs",
            "docs are in",
        ]
    )
 
 
def _contract_rules_router(query: str) -> Dict[str, Any]:
    ql = (query or "").lower()
    program_code = _extract_program_code(query)
    oems = _detect_oems_in_query(query)
    comparey = _is_compare_question(query)
    listy = _is_list_question(query)
    county = _is_count_question(query)
 
    matched_fields: List[str] = []
    for field, phrases in _CONTRACT_FIELD_ALIASES.items():
        for p in phrases:
            if p in ql:
                matched_fields.append(field)
                break
 
    matched_fields = list(dict.fromkeys(matched_fields))
 
    if listy and "eligible_products_json" not in matched_fields:
        matched_fields.insert(0, "eligible_products_json")
 
    if county:
        return {
            "intent": "count",
            "program_code": program_code,
            "oems": oems,
            "fields": [],
            "confidence": 0.9,
            "needs_clarification": False,
        }
 
    if comparey:
        if not matched_fields:
            matched_fields = ["payment_terms"]
        return {
            "intent": "metadata_compare",
            "program_code": program_code,
            "oems": oems,
            "fields": [f for f in matched_fields if f in _CONTRACT_ALLOWED_FIELDS],
            "confidence": 0.8,
            "needs_clarification": False if oems else True,
        }
 
    if matched_fields:
        return {
            "intent": "metadata_lookup",
            "program_code": program_code,
            "oems": oems,
            "fields": [f for f in matched_fields if f in _CONTRACT_ALLOWED_FIELDS],
            "confidence": 0.85,
            "needs_clarification": False,
        }
 
    return {
        "intent": "evidence",
        "program_code": program_code,
        "oems": oems,
        "fields": [],
        "confidence": 0.6,
        "needs_clarification": False,
    }
 
 
def _search_contract_lexical(
    query: str,
    top: int = 20,
    filter_expr: Optional[str] = None,
    select: Optional[List[str]] = None,
) -> List[dict]:
    """Lexical (keyword-only) search on the contract Azure Search index."""
    if contract_search_client is None:
        return []
    if not select:
        select = [
            "blob_path",
            "program_name",
            "oem",
            "promotion_year",
            "payment_terms",
            "eligible_products_json",
            "funding_cap",
        ]
    results = contract_search_client.search(
        search_text=query, filter=filter_expr, top=top, select=select
    )
    return [{k: r.get(k) for k in select} for r in results]
 
 
def _choose_best_contract_blob_path(rows: List[dict]) -> Optional[str]:
    paths = [r.get("blob_path") for r in rows if r.get("blob_path")]
    if not paths:
        return None
    return _Counter(paths).most_common(1)[0][0]
 
 
def _get_contract_doc_metadata(blob_path: str, fields: List[str]) -> Dict[str, Any]:
    if contract_search_client is None:
        return {}
    filter_expr = f"blob_path eq '{_escape_odata(blob_path)}'"
    base = ["blob_path", "program_name", "oem", "promotion_year"]
    extra = [f for f in fields if f not in base]
    select = base + extra
    results = contract_search_client.search(
        search_text="*", filter=filter_expr, top=1, select=select
    )
    for r in results:
        return {k: r.get(k) for k in select}
    return {}
 
 
def _resolve_contract_blob_path(program_code: str, oems: List[str]) -> Optional[str]:
    rows = _search_contract_lexical(
        program_code, top=30, select=["blob_path", "program_name", "oem"]
    )
    if oems:
        filtered = [
            r
            for r in rows
            if any(_blob_path_matches_oem(r.get("blob_path", ""), o) for o in oems)
        ]
        if filtered:
            rows = filtered
    return _choose_best_contract_blob_path(rows)
 
 
def _parse_eligible_products(eligible_products_json: str) -> List[dict]:
    if not eligible_products_json:
        return []
    try:
        data = json.loads(eligible_products_json)
        if isinstance(data, list):
            return data
        return []
    except Exception:
        return []
 
 
def _format_eligible_products(items: List[dict]) -> str:
    if not items:
        return "No eligible products/devices were found in metadata for this program."
    items_sorted = sorted(items, key=lambda x: (x.get("model_name") or "").lower())
    lines = []
    for i, it in enumerate(items_sorted, 1):
        model = it.get("model_name", "").strip()
        sku = it.get("sku", "").strip()
        mem = it.get("memory_variant", "").strip()
        credit = it.get("credit_amount", None)
        cond = (it.get("pricing_conditions") or "").strip()
 
        bits = [model] if model else []
        if mem:
            bits.append(mem)
        if sku:
            bits.append(f"SKU: {sku}")
        if credit is not None:
            bits.append(f"Credit: {credit}")
        if cond:
            bits.append(f"Cond: {cond}")
 
        lines.append(f"{i}. " + " | ".join(bits) if bits else f"{i}. (unknown item)")
    return "\n".join(lines)
 
 
def _generate_contract_metadata_answer(
    user_query: str,
    conversation_history: List[Dict],
) -> Optional[tuple]:
    plan = _contract_rules_router(user_query)
    intent = plan["intent"]
    fields = plan.get("fields", [])
    program_code = plan.get("program_code")
    oems = plan.get("oems", [])
 
    logger.debug("[Contract Metadata] route_plan: %s", plan)
 
    if intent not in ("metadata_lookup", "metadata_compare", "count"):
        return None
 
    if intent == "count":
        rows = _search_contract_lexical(
            user_query, top=100, select=["blob_path", "program_name", "oem"]
        )
        if oems:
            rows = [
                r
                for r in rows
                if any(_blob_path_matches_oem(r.get("blob_path", ""), o) for o in oems)
            ]
        unique_paths = len({r.get("blob_path") for r in rows if r.get("blob_path")})
        oem_str = ", ".join(oems) if oems else "all OEMs"
        answer = (
            f"Found approximately **{unique_paths}** unique contract document(s) "
            f"matching your query ({oem_str})."
        )
        return answer, None
 
    if intent == "metadata_compare":
        if not fields:
            fields = ["payment_terms"]
        if oems and not program_code:
            per_oem_summaries = []
            for o in oems:
                rows = _search_contract_lexical(
                    user_query,
                    top=30,
                    select=["blob_path", "program_name", "oem", "promotion_year"]
                    + [
                        f
                        for f in fields
                        if f
                        not in ("blob_path", "program_name", "oem", "promotion_year")
                    ],
                )
                rows = [
                    r for r in rows if _blob_path_matches_oem(r.get("blob_path", ""), o)
                ]
                seen_bp: set = set()
                chosen = []
                for r in rows:
                    bp = r.get("blob_path")
                    if not bp or bp in seen_bp:
                        continue
                    seen_bp.add(bp)
                    chosen.append(r)
                    if len(chosen) >= 5:
                        break
                per_oem_summaries.append((o, chosen))
 
            if any(not docs for _, docs in per_oem_summaries):
                return None
 
            context_lines = []
            citations: List[str] = []
            idx = 1
            for o, docs in per_oem_summaries:
                context_lines.append(f"== OEM: {o} ==")
                if not docs:
                    context_lines.append("(no documents found)")
                    continue
                for d in docs:
                    bp = d.get("blob_path", "")
                    pn = d.get("program_name", "")
                    yr = d.get("promotion_year", "")
                    context_lines.append(f"[{idx}] {pn} | year={yr} | {bp}")
                    for f in fields:
                        val = d.get(f) or ""
                        if val:
                            context_lines.append(f"  - {f}: {val}")
                    citations.append(f"[{idx}] {bp}")
                    idx += 1
 
            meta_context = "\n".join(context_lines)
            if not meta_context.strip():
                return None
 
            _MAX_HIST = _CONTRACT_MAX_HISTORY_TURNS * 2
            msgs: List[Dict] = [{"role": "system", "content": _CONTRACT_SYSTEM_PROMPT}]
            if conversation_history:
                valid_roles = {"user", "assistant", "system"}
                clipped = [
                    m
                    for m in conversation_history
                    if isinstance(m, dict)
                    and m.get("role") in valid_roles
                    and m.get("content")
                ][-_MAX_HIST:]
                msgs.extend(clipped)
            context_instruction = (
                "Context:\n"
                f"{meta_context}\n\n"
                "Sources:\n" + "\n".join(citations) + "\n\n"
                "Instructions:\n"
                "- Cite sources inline like [1], [2].\n"
                "- If no relevant context, say so explicitly."
            )
            msgs.append({"role": "system", "content": context_instruction})
            msgs.append({"role": "user", "content": user_query})
            resp = client.chat.completions.create(
                model=LLM_Model,
                messages=msgs,
                temperature=0.2,
                max_completion_tokens=700,
            )
            answer = resp.choices[0].message.content.strip()
            return answer, None
 
        return None
 
    if intent == "metadata_lookup":
        blob_path = None
        if program_code:
            blob_path = _resolve_contract_blob_path(program_code, oems)
        if not blob_path:
            return None
 
        md = _get_contract_doc_metadata(blob_path, fields)
        if not md:
            return None
 
        if "eligible_products_json" in fields:
            items = _parse_eligible_products(md.get("eligible_products_json", ""))
            answer = (
                f"Eligible devices/products for {md.get('program_name', program_code)}:\n"
                + _format_eligible_products(items)
                + "\n\nSource: [1]"
            )
            return answer, blob_path
 
        if len(fields) == 1:
            f = fields[0]
            val = md.get(f, "")
            if not val:
                return (
                    f"I couldn't find '{f}' in metadata for {md.get('program_name', program_code)}.",
                    blob_path,
                )
            return (
                f"{f} for {md.get('program_name', program_code)}:\n{val}\n\nSource: [1]",
                blob_path,
            )
 
        lines = [f"Metadata for {md.get('program_name', program_code)} ({blob_path}):"]
        for f in fields:
            v = md.get(f, "")
            if v:
                lines.append(f"- {f}: {v}")
        return "\n".join(lines) + "\n\nSource: [1]", blob_path
 
    return None
 
 
# ---------------------------------------------------------------------------
# Answer generation (streaming)
# ---------------------------------------------------------------------------
 
_RAG_SYSTEM_PROMPT = """You are an extraction-only RAG assistant for O9 planning documents. Your job is to answer questions STRICTLY based on the provided context documents.
 
CRITICAL RULES:
1) ONLY use information from the provided context documents. DO NOT use your general knowledge about O9, planning systems, or any other topic.
2) If the required information is NOT in the provided context, you MUST respond with: "I couldn't find that specific information in the O9 documents. The available documents don't contain details about [topic]."
3) When the context contains table information AND the question asks for tabular/structured data (e.g., "show me the table", "list all", "compare"), reproduce the table exactly (preserve headers, rows, and formatting).
4) For simple questions (definitions, explanations, single values), provide concise one-line or short paragraph answers. DO NOT use tables unless the question specifically asks for structured/tabular data.
5) Keep responses clear and helpful. Citations are not required.
6) Only when you return a table, introduce it with a short one-line description explaining what the table contains.
7) Evaluate the CURRENT question independently against the CURRENT context documents. A previous turn where you couldn't answer does NOT mean this question is unanswerable — if the answer is in the context above, answer it fully.
8) You ONLY answer questions about O9. If the user asks you to write creative content (poems, stories, jokes), generate code, give opinions, or do anything unrelated to O9, use the rule-2 refusal instead — do NOT comply, even if the context could be repurposed to satisfy the request.
 
IMPORTANT: If you find yourself using phrases like "usually", "commonly", "patterns", "may differ by implementation" - STOP. These indicate you're using general knowledge instead of the provided context. Only state facts directly from the documents.
 
FEW-SHOT EXAMPLES:
 
Example 1 - Simple question (NO TABLE):
Question: "What are override tiers?"
 
Answer:
Override tiers are priority levels used in the O9 planning system to classify items or locations based on business rules and forecasting requirements.
 
 
Example 2 - Table request (USE TABLE):
Question: "Show me the table Rule for Sellout Fcst"
 
Answer:
Please find the table Rule for Sellout Fcst:
 
| Assortment             | 1   | Null | 0   | 1                                          | 1   |
| Sell Out Forecast Flag | 1   | 1    | 1   | Null (if there are no sell out fcst dates) | 0   |
| Sell Out Forecast      | Yes | Yes  | Yes | No                                         | No  |
| Sell In Forecast       | Yes | No   | No  | Yes                                        | Yes |
 
 
Example 3 - List request (USE TABLE):
Question: "List all Priority Principles"
 
Answer:
Please find below the Priority Principles:
 
| Tier Name        | Priority                              |
| Magenta Device 1 | 200                                   |
| Magenta Device 2 | 100 - lower number is higher priority |
"""
 
 
async def generate_answer_stream(
    query: str,
    context_docs: List[Dict[str, Any]],
    conversation_history: List[Dict] = None,
):
    """Generate answer using GPT-4 with streaming (Async)."""
    context_parts = []
    for doc in context_docs:
        context_parts.append(f"[Source: {doc['title']}]\n{doc['content']}")
    context = "\n\n---\n\n".join(context_parts)
 
    messages = [{"role": "system", "content": _RAG_SYSTEM_PROMPT}]
 
    if conversation_history:
        valid_roles = {"user", "assistant", "system", "human", "ai"}
        role_mapping = {"human": "user", "ai": "assistant"}
 
        for msg in conversation_history:
            if (
                isinstance(msg, dict)
                and msg.get("role") in valid_roles
                and msg.get("content")
            ):
                # Skip internal mode tags — don't send __mode__:X to the LLM
                content = msg.get("content", "")
                if isinstance(content, str) and content.startswith("__mode__:"):
                    continue
                messages.append(
                    {
                        "role": role_mapping.get(msg["role"], msg["role"]),
                        "content": str(content),
                    }
                )
 
    messages.append(
        {
            "role": "user",
            "content": f"Context Documents from O9:\n\n{context}\n\n---\n\nQuestion: {query}\n\nIMPORTANT: Answer ONLY using the information in the context documents above. If the context doesn't contain the answer, say so explicitly. Do NOT use general knowledge about O9 or planning systems.",
        }
    )
 
    stream = await async_client.chat.completions.create(
        model=LLM_Model,
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=500,
        stream=True,
    )
 
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
 
 
_CONTRACT_SYSTEM_PROMPT = (
    "You are a helpful contracts assistant. "
    "Answer using ONLY the provided context or metadata provided. "
    "If information is missing, say so explicitly. "
    "You ONLY answer questions about OEM contracts and term sheets. If the user asks you "
    "to write creative content (poems, stories, jokes), generate code, give opinions, or do "
    "anything unrelated to contracts, decline and say that's outside what you can help with — "
    "do NOT comply, even if the context could be repurposed. "
    "Keep answers concise and include citations like [1], [2] when sources are provided."
)
 
_CONTRACT_MAX_HISTORY_TURNS = 4
_CONTRACT_MAX_CONTEXT_CHUNKS = 40
 
 
def _build_contract_context_block(
    docs: List[Dict[str, Any]],
) -> tuple:
    def _infer_oem_from_blob(bp: str) -> str:
        bpl = (bp or "").lower()
        if "/apple/" in bpl or "\\apple\\" in bpl:
            return "apple"
        if "/samsung/" in bpl or "\\samsung\\" in bpl:
            return "samsung"
        if "/motorola/" in bpl or "\\motorola\\" in bpl:
            return "motorola"
        if "/google/" in bpl or "\\google\\" in bpl:
            return "google"
        if "/lg/" in bpl or "\\lg\\" in bpl:
            return "lg"
        if "/tcl/" in bpl or "\\tcl\\" in bpl:
            return "tcl"
        return ""
 
    context_lines = []
    citations = []
 
    for i, doc in enumerate(docs[:_CONTRACT_MAX_CONTEXT_CHUNKS], 1):
        blob_path = doc.get("blob_path") or doc.get("source", "")
        page_num = doc.get("page_num")
        title = doc.get("title", "")
        oem_inferred = _infer_oem_from_blob(blob_path)
 
        raw_content = doc.get("content") or ""
        snippet = raw_content.strip()
 
        snippet = " ".join(snippet.split())
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "..."
 
        struct_meta = ""
        if "Document Content:" in raw_content:
            struct_meta = raw_content.split("Document Content:")[0].strip()
        _meta_fields = {}
        for line in struct_meta.splitlines():
            if ": " in line:
                k, v = line.split(": ", 1)
                _meta_fields[k.strip()] = v.strip()
 
        header_bits = []
        if title:
            header_bits.append(title)
        if oem_inferred:
            header_bits.append(f"path_oem={oem_inferred}")
        llm_oem = _meta_fields.get("OEM", "")
        if llm_oem:
            header_bits.append(f"llm_oem={llm_oem}")
        doc_type = _meta_fields.get("Document Type", "")
        if doc_type:
            header_bits.append(doc_type)
        header = " | ".join(header_bits) if header_bits else "Contract chunk"
 
        page_label = f" p.{page_num}" if page_num else ""
        context_lines.append(f"[{i}] ({blob_path}{page_label}) {header}\n{snippet}")
        citations.append(f"[{i}] {blob_path}{page_label}")
 
    return "\n\n".join(context_lines), citations
 
 
async def generate_contract_answer_stream(
    query: str,
    context_docs: List[Dict[str, Any]],
    conversation_history: List[Dict] = None,
):
    """Generate contract answers using the pipeline_bot_enhanced approach."""
 
    def _filter_history_for_llm(history: List[Dict]) -> List[Dict]:
        """Strip internal __mode__ system tags before sending to LLM."""
        out = []
        for m in (history or []):
            if not isinstance(m, dict):
                continue
            content = m.get("content", "")
            if isinstance(content, str) and content.startswith("__mode__:"):
                continue
            out.append(m)
        return out
 
    def _make_messages(context_block: str, citns: List[str]) -> List[Dict]:
        msgs: List[Dict] = [{"role": "system", "content": _CONTRACT_SYSTEM_PROMPT}]
 
        if conversation_history:
            valid_roles = {"user", "assistant", "system"}
            clean_history = _filter_history_for_llm(conversation_history)
            clipped = [
                m
                for m in clean_history
                if isinstance(m, dict)
                and m.get("role") in valid_roles
                and m.get("content")
            ][-(_CONTRACT_MAX_HISTORY_TURNS * 2):]
            msgs.extend(clipped)
 
        context_instruction = (
            "Context:\n"
            f"{context_block}\n\n"
            "Sources:\n" + "\n".join(citns) + "\n\n"
            "Instructions:\n"
            "- Cite sources inline like [1], [2].\n"
            "- If no relevant context, say so explicitly."
        )
        msgs.append({"role": "system", "content": context_instruction})
        msgs.append({"role": "user", "content": query})
        return msgs
 
    async def _stream_messages(messages: List[Dict]):
        stream = await async_client.chat.completions.create(
            model=LLM_Model,
            messages=messages,
            temperature=0.1,
            top_p=0.9,
            max_completion_tokens=1500,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
 
    context_block, citations = _build_contract_context_block(context_docs)
    try:
        async for token in _stream_messages(_make_messages(context_block, citations)):
            yield token
        return
    except Exception as e:
        err_str = str(e)
        is_content_filter = (
            "content_filter" in err_str or "content management policy" in err_str
        )
        if not is_content_filter:
            raise
        logger.warning(
            "[Contract Stream] Content filter on full context — retrying with structured fields only"
        )
 
    stripped_docs = []
    for doc in context_docs:
        d = dict(doc)
        content = d.get("content", "")
        if "Document Content:" in content:
            content = content.split("Document Content:")[0].strip()
        d["content"] = content
        stripped_docs.append(d)
    context_block2, citations2 = _build_contract_context_block(stripped_docs)
    try:
        async for token in _stream_messages(_make_messages(context_block2, citations2)):
            yield token
        return
    except Exception as e:
        err_str = str(e)
        is_content_filter = (
            "content_filter" in err_str or "content management policy" in err_str
        )
        if not is_content_filter:
            raise
        logger.warning(
            "[Contract Stream] Content filter on structured fields too — falling back to direct metadata"
        )
 
    if context_docs:
        best = context_docs[0]
        content = best.get("content", "")
        lines = [ln for ln in content.splitlines() if ln.strip()]
        yield (
            "\n".join(lines)
            if lines
            else (
                "I was unable to retrieve a structured answer due to content filter restrictions. "
                "Please open the source document directly using the link below."
            )
        )
    else:
        yield (
            "I was unable to answer this question due to content filter restrictions. "
            "Please open the source document directly using the link below."
        )
 
 # ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------
 
 
class RAGAgent(OrchestratorAgent):
    """Retrieval-augmented generation agent for O9 documents."""
 
    def __init__(self) -> None:
        super().__init__(name="rag", priority=10)
        self.session_start = datetime.now()
 
    async def run_stream(self, context: AgentContext):
        query = context.query
        logger.info("[RAG Stream] Starting for query: %s", query)
 
        yield {
            "type": "status",
            "content": "Searching O9 documents...",
            "agent": self.name,
        }
 
        retrieved_docs = await asyncio.to_thread(similarity_search, query)
        logger.info(
            "[RAG Stream] Found %d documents for query: '%s...'",
            len(retrieved_docs),
            query[:50],
        )
 
        if not retrieved_docs:
            yield {
                "type": "content",
                "content": "I couldn't find any relevant O9 documents to answer your question.",
                "agent": self.name,
            }
            yield {"type": "done", "agent": self.name}
            return
 
        yield {
            "type": "status",
            "content": f"Found {len(retrieved_docs)} O9 documents. Generating answer...",
            "agent": self.name,
        }
 
        working_history = list(context.conversation_history or [])
        logger.debug(
            "[RAG Stream] Using conversation history with %d messages",
            len(working_history),
        )
        answer = ""
        try:
            logger.info("[RAG Stream] Generating answer stream...")
            async for chunk in generate_answer_stream(
                query, retrieved_docs, working_history
            ):
                if chunk:
                    answer += chunk
                    yield {"type": "content", "content": chunk, "agent": self.name}
            logger.info(
                "[RAG Stream] Answer generation complete. Length: %d", len(answer)
            )
        except Exception as e:
            logger.error("[RAG Stream] Error generating answer: %s", e)
            yield {
                "type": "content",
                "content": f"Error generating answer: {str(e)}",
                "agent": self.name,
            }
 
        yield {
            "type": "final",
            "result": {
                "agent": self.name,
                "output": {
                    "query": query,
                    "answer": answer,
                    "sources": [
                        {"title": doc["title"], "source": doc["source"]}
                        for doc in retrieved_docs
                    ],
                    "retrieved_docs": retrieved_docs,
                    "conversation_history": working_history,
                },
            },
        }
 
 
class ContractManagementAgent(OrchestratorAgent):
    """Contract and term sheet RAG agent for promotional agreement queries."""
 
    def __init__(self) -> None:
        super().__init__(name="contract", priority=5)
        self.session_start = datetime.now()
 
    _DATA_INTENT_PATTERNS = re.compile(
        r"\b("
        r"what percent|how many|how much|sell.?through|channel mix|units sold"
        r"|sales by|% of|rate of|breakdown|sales trend|sold via|sales are"
        r"|sales in|sales done|sales were|per store|per rep|top \d|bottom \d"
        r"|rank|ranking|ytd|mtd|qtd|week over week|month over month|yoy|wow|mom"
        r")\b"
    )
 
 
    async def run_stream(self, context: AgentContext):
        query = context.query
        logger.info("[Contract Stream] Starting for query: %s", query)
 
        yield {
            "type": "status",
            "content": "Searching contract documents...",
            "agent": self.name,
        }
 
        _meta_plan = _contract_rules_router(query)
        if _meta_plan["intent"] in ("metadata_lookup", "metadata_compare", "count"):
            logger.info(
                "[Contract Stream] Metadata intent detected: %s — trying metadata-first path",
                _meta_plan["intent"],
            )
            _meta_result = await asyncio.to_thread(
                _generate_contract_metadata_answer,
                query,
                list(context.conversation_history or []),
            )
            if _meta_result is not None:
                _meta_answer, _meta_blob_path = _meta_result
 
                yield {"type": "content", "content": _meta_answer, "agent": self.name}
 
                if _meta_blob_path:
                    _meta_sas_url = _make_blob_sas_url(_meta_blob_path.lstrip("/"))
                    if _meta_sas_url:
                        sources_block = (
                            f"\n\n---\n**Source:**\n"
                            f"- [{_meta_blob_path}]({_meta_sas_url})"
                        )
                    else:
                        sources_block = f"\n\n---\n**Source:**\n- {_meta_blob_path}"
                    yield {
                        "type": "content",
                        "content": sources_block,
                        "agent": self.name,
                    }
                    _meta_answer += sources_block
 
                working_history = list(context.conversation_history or [])
                working_history.append({"role": "user", "content": query})
                working_history.append({"role": "assistant", "content": _meta_answer})
                if len(working_history) > 20:
                    working_history = working_history[-20:]
 
                yield {
                    "type": "final",
                    "result": {
                        "agent": self.name,
                        "output": {
                            "query": query,
                            "answer": _meta_answer,
                            "sources": (
                                [
                                    {
                                        "title": _meta_blob_path,
                                        "source": _meta_blob_path,
                                        "url": _make_blob_sas_url(
                                            _meta_blob_path.lstrip("/")
                                        ),
                                    }
                                ]
                                if _meta_blob_path
                                else []
                            ),
                            "retrieved_docs": [],
                            "conversation_history": working_history,
                        },
                    },
                }
                return
 
            logger.info(
                "[Contract Stream] Metadata path returned None — falling through to hybrid RAG"
            )
 
        retrieved_docs = await asyncio.to_thread(contract_similarity_search, query)
 
        _pc_match = re.search(r"\b([0-9]{2}[A-Z]{2,4}[0-9]{3})\b", query, re.IGNORECASE)
        _query_lower = query.lower()
        _is_multi_program = (
            _is_compare_question(query)
            or any(k in _query_lower for k in ["all programs", "which programs"])
            or len(_detect_oems_in_query(query)) >= 2
        )
        if _pc_match and not _is_multi_program:
            _pc = _pc_match.group(1).upper()
            _matched = [
                d
                for d in retrieved_docs
                if _pc.lower() in (d.get("title") or "").lower()
                or _pc.lower() in (d.get("source") or "").lower()
            ]
            if _matched:
                retrieved_docs = _matched
                logger.info(
                    "[Contract Stream] Trimmed to %d docs matching program code %s",
                    len(retrieved_docs),
                    _pc,
                )
 
        context_docs = retrieved_docs
        logger.info(
            "[Contract Stream] Passing %d docs to LLM (retrieved %d)",
            len(context_docs),
            len(retrieved_docs),
        )
 
        if not retrieved_docs:
            _dm = re.search(
                r"(?<!\w)([0-9]{1,2}[a-z]{1,2})(?!\w)", query, re.IGNORECASE
            )
            if _dm:
                _did = _dm.group(1).lower()
                not_found_msg = (
                    f"I couldn't find document **{_did}** in the contract index. "
                    f"The file may not have been uploaded or indexed yet. "
                    f"Please verify that a file starting with **{_did}** exists in the blob storage container and has been indexed."
                )
            else:
                not_found_msg = "I couldn't find any relevant contract documents to answer your question."
            yield {"type": "content", "content": not_found_msg, "agent": self.name}
            yield {"type": "done", "agent": self.name}
            return
 
        yield {
            "type": "status",
            "content": f"Found {len(context_docs)} contract documents. Generating answer...",
            "agent": self.name,
        }
 
        working_history = list(context.conversation_history or [])
 
        answer = ""
        try:
            logger.info("[Contract Stream] Generating answer stream...")
            async for chunk in generate_contract_answer_stream(
                query, context_docs, working_history
            ):
                if chunk:
                    answer += chunk
                    yield {"type": "content", "content": chunk, "agent": self.name}
            logger.info(
                "[Contract Stream] Answer generation complete. Length: %d", len(answer)
            )
        except Exception as e:
            logger.error("[Contract Stream] Error generating answer: %s", e)
            err_msg = "I encountered an error generating the answer. Please try again."
            answer += err_msg
            yield {"type": "content", "content": err_msg, "agent": self.name}
 
        if context_docs:
            cited_numbers = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
 
            cited_docs: List[dict] = []
            seen_titles: set = set()
            for i, doc in enumerate(context_docs, 1):
                if i not in cited_numbers:
                    continue
                title_key = (doc.get("title") or "").strip()
                if title_key and title_key not in seen_titles:
                    seen_titles.add(title_key)
                    cited_docs.append(doc)
 
            source_lines = []
            for doc in cited_docs:
                doc_url = doc.get("url", "")
                doc_title = doc.get("title", "Unknown")
                page_num = doc.get("page_num")
                page_label = f" (page {page_num})" if page_num else ""
                if doc_url:
                    source_lines.append(f"- [{doc_title}{page_label}]({doc_url})")
                else:
                    source_lines.append(f"- {doc_title}{page_label}")
 
            if source_lines:
                sources_header = (
                    "**Source:**" if len(cited_docs) == 1 else "**Sources:**"
                )
                sources_block = f"\n\n---\n{sources_header}\n" + "\n".join(source_lines)
                answer += sources_block
                yield {"type": "content", "content": sources_block, "agent": self.name}
 
        working_history.append({"role": "user", "content": query})
        working_history.append({"role": "assistant", "content": answer})
        if len(working_history) > 20:
            working_history = working_history[-20:]
 
        yield {
            "type": "final",
            "result": {
                "agent": self.name,
                "output": {
                    "query": query,
                    "answer": answer,
                    "sources": [
                        {
                            "title": doc["title"],
                            "source": doc["source"],
                            "url": doc.get("url", ""),
                            "page_num": doc.get("page_num"),
                        }
                        for doc in context_docs
                    ],
                    "retrieved_docs": context_docs,
                    "conversation_history": working_history,
                },
            },
        }
 
 
class SQLAgent(OrchestratorAgent):
    """Wrapper around the SQL executor."""
 
    def __init__(self) -> None:
        super().__init__(name="sql", priority=100)
 
 
    async def run_stream(self, context: AgentContext):
        logger.info("[SQL/O9 Stream] Starting for query: %s", context.query)
        yield {"type": "status", "content": "Finding the right data...", "agent": self.name}
 
        state = context.state
        # Use context.conversation_history (already threaded/effective history set by orchestrator)
        # Fall back to state dict only if context history is empty.
        conv_history = context.conversation_history or state.get("conversation_history", [])
        logger.debug(
            "[SQL Stream] Query: '%s...', History messages: %d",
            context.query[:50],
            len(conv_history),
        )
 
        _STAGE_TEXT = {"run": "Running the query...", "format": "Formatting the results..."}
        loop = asyncio.get_running_loop()
        status_q: asyncio.Queue = asyncio.Queue()
        _DONE = object()
 
        def progress_cb(stage):
            loop.call_soon_threadsafe(status_q.put_nowait, stage)
 
        _t0 = time.perf_counter()
        task = asyncio.create_task(
            asyncio.to_thread(
                get_sql_agent_response,
                query=context.query,
                conversation_history=conv_history,
                session_id=state.get("session_id"),
                hash_id=state.get("hash_id"),
                thread_id=state.get("thread_id"),
                query_id=state.get("query_id"),
                role=state.get("role"),
                progress_cb=progress_cb,
            )
        )
        task.add_done_callback(lambda _t: status_q.put_nowait(_DONE))
 
        while True:
            item = await status_q.get()
            if item is _DONE:
                break
            text = _STAGE_TEXT.get(item)
            if text:
                yield {"type": "status", "content": text, "agent": self.name}
 
        result = await task
        logger.info(
            "[SQL timing] get_sql_agent_response=%.2fs (react steps=%s)",
            time.perf_counter() - _t0, result.get("step_count"),
        )
 
        answer = result.get("answer", "")
        logger.info("[SQL/O9 Stream] Answer received. Length: %d", len(answer))
 
        import re as _re
        stream_answer = _re.sub(
            r"```chart_spec\s*\{[\s\S]*?\}\s*```", "", answer
        ).strip()
        stream_answer = _re.sub(
            r"```csv\s*.*?\s*```", "", stream_answer, flags=_re.DOTALL
        ).strip()
 
        words = stream_answer.split(" ")
        for word in words:
            yield {"type": "content", "content": word + " ", "agent": self.name}
 
        yield {"type": "final", "result": {"agent": self.name, "output": result}}
 
 
 # ---------------------------------------------------------------------------
# Onboarding agent
# ---------------------------------------------------------------------------
 
_ONBOARDING_SYSTEM_PROMPT = (
    "You are an onboarding assistant for new T-Mobile employees. "
    "Answer ONLY using the provided onboarding context. "
    "You ONLY handle onboarding questions. If the user asks you to write creative content "
    "(poems, stories, jokes), generate code, give opinions, or do anything unrelated to "
    "onboarding, respond with NO_INFO and briefly redirect — do NOT comply, even if the "
    "context could be repurposed to satisfy the request. "
    "FORMAT: your very first line must be a single token — `ANSWERED` if the context contains the "
    "answer, or `NO_INFO` if it does not — then put your actual reply on the following lines. "
    "If NO_INFO, briefly say you don't have that yet and suggest asking their onboarding buddy or "
    "team lead. Be concise and friendly; use bullet points where helpful. Refer to teams and roles "
    "by function, not by individual people's names."
)
 
_DATA_SYSTEM_PROMPT = (
    "You are a data-discovery assistant for T-Mobile's Device Reporting & Production team. "
    "Your job is to help people find WHERE data lives and HOW to get it: the source system or "
    "feed, the pipeline/scraper/report that produces it, where it lands, how it's refreshed, and "
    "who or what to go through for access. Answer ONLY using the provided context. "
    "FORMAT: your very first line must be a single token — `ANSWERED` if the context contains the "
    "answer, or `NO_INFO` if it does not — then put your actual reply on the following lines. "
    "If NO_INFO, briefly say you don't have that documented yet and suggest checking with the "
    "Device Reporting & Production team. Be concise; use bullet points and name the specific "
    "source/pipeline/report when the context gives it. Refer to roles by function, not by names."
)
 
 
def _filter_mode_tags_from_history(history: List[Dict]) -> List[Dict]:
    """Remove __mode__:X system tags before passing history to any LLM call."""
    return [
        m for m in (history or [])
        if not (
            isinstance(m, dict)
            and m.get("role") == "system"
            and isinstance(m.get("content", ""), str)
            and m["content"].startswith("__mode__:")
        )
    ]
 
 
async def generate_grounded_answer_stream(
    query, context_docs, system_prompt, conversation_history=None, context_label="Context"
):
    """Stream a grounded answer from retrieved docs under a given system prompt."""
    context = "\n\n---\n\n".join(
        f"[{d.get('title', '')}]\n{d.get('content', '')}" for d in context_docs
    )
    messages = [{"role": "system", "content": system_prompt}]
    if conversation_history:
        valid_roles = {"user", "assistant", "system", "human", "ai"}
        role_mapping = {"human": "user", "ai": "assistant"}
        # Strip internal mode tags before sending to LLM
        clean_history = _filter_mode_tags_from_history(conversation_history)
        for msg in clean_history:
            if isinstance(msg, dict) and msg.get("role") in valid_roles and msg.get("content"):
                messages.append(
                    {"role": role_mapping.get(msg["role"], msg["role"]), "content": str(msg["content"])}
                )
    messages.append(
        {
            "role": "user",
            "content": (
                f"{context_label}:\n\n{context}\n\n---\n\nQuestion: {query}\n\n"
                "Answer ONLY from the context above."
            ),
        }
    )
    stream = await async_client.chat.completions.create(
        model=LLM_Model,
        messages=messages,
        temperature=0.1,
        top_p=0.9,
        max_completion_tokens=600,
        stream=True,
    )
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content is not None:
            yield chunk.choices[0].delta.content
 
 
async def generate_onboarding_answer_stream(query, context_docs, conversation_history=None):
    async for c in generate_grounded_answer_stream(
        query, context_docs, _ONBOARDING_SYSTEM_PROMPT, conversation_history, "Onboarding context"
    ):
        yield c
 
 
class OnboardingAgent(OrchestratorAgent):
    """Team-scoped onboarding RAG over the wiki-sourced onboarding index."""
 
    def __init__(self) -> None:
        super().__init__(name="onboarding", priority=10)
 
 
    async def run_stream(self, context: AgentContext):
        team = (context.state or {}).get("team") or extract_team(context.query)
        yield {"type": "status", "content": "Searching onboarding docs...", "agent": self.name}
 
        if not team:
            team_labels = ", ".join(m["label"] for m in ONBOARDING_TEAMS.values())
            msg = (
                "Happy to help with onboarding! Which team are you asking about? "
                f"({team_labels})"
            )
            yield {
                "type": "team_request",
                "content": msg,
                "teams": [{"label": m["label"], "value": slug} for slug, m in ONBOARDING_TEAMS.items()],
                "original_query": context.query,
                "agent": self.name,
            }
            yield {"type": "content", "content": msg, "agent": self.name}
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": msg, "steps": [], "conversation_history": context.conversation_history}}}
            return
 
        docs = await asyncio.to_thread(onboarding_search, context.query, team)
        if not docs:
            msg = (
                f"I don't have onboarding info for that in the {team} space yet — "
                "try your onboarding buddy or team lead."
            )
            yield {"type": "content", "content": msg, "agent": self.name}
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": msg, "steps": [], "conversation_history": context.conversation_history}}}
            return
 
        answer = ""
        answered = True
        header_done = False
        buf = ""
        try:
            async for chunk in generate_onboarding_answer_stream(
                context.query, docs, context.conversation_history
            ):
                if not chunk:
                    continue
                if not header_done:
                    buf += chunk
                    if "\n" in buf or len(buf) > 16:
                        up = buf.lstrip().upper()
                        if up.startswith("NO_INFO"):
                            answered, remainder = False, buf.lstrip()[len("NO_INFO"):]
                        elif up.startswith("ANSWERED"):
                            answered, remainder = True, buf.lstrip()[len("ANSWERED"):]
                        else:
                            answered, remainder = True, buf
                        remainder = remainder.lstrip(" :-\n")
                        header_done = True
                        if remainder:
                            answer += remainder
                            yield {"type": "content", "content": remainder, "agent": self.name}
                    continue
                answer += chunk
                yield {"type": "content", "content": chunk, "agent": self.name}
            if not header_done and buf:
                up = buf.lstrip().upper()
                answered = not up.startswith("NO_INFO")
                remainder = buf.lstrip()
                for pfx in ("ANSWERED", "NO_INFO"):
                    if up.startswith(pfx):
                        remainder = remainder[len(pfx):]
                        break
                remainder = remainder.lstrip(" :-\n")
                if remainder:
                    answer += remainder
                    yield {"type": "content", "content": remainder, "agent": self.name}
        except Exception as e:
            logger.error("[Onboarding] generation error: %s", e)
            yield {"type": "content", "content": f"Error generating answer: {e}", "agent": self.name}
 
        citations, seen = [], set()
        for d in docs:
            for s in d.get("sources", []):
                url = s.get("url")
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"title": s.get("title", "Source"), "url": url})
        shown = citations if answered else []
        if shown:
            footer = "\n\n**Sources:**\n" + "\n".join(
                f"- [{c['title']}]({c['url']})" for c in shown
            )
            answer += footer
            yield {"type": "content", "content": footer, "agent": self.name}
 
        yield {"type": "final", "result": {"agent": self.name, "output": {
            "answer": answer, "steps": [], "sources": shown,
            "retrieved_docs": docs, "conversation_history": context.conversation_history}}}
 
 
class DataAgent(OrchestratorAgent):
    """Data-discovery RAG over the Device Reporting & Production wiki."""
 
    def __init__(self) -> None:
        super().__init__(name="data", priority=10)
 
    def score(self, context: AgentContext) -> float:
        return 0.0  # explicit activation only (Data Finder pill); see orchestrator mode routing
 
    async def run_stream(self, context: AgentContext):
        yield {"type": "status", "content": "Searching data docs...", "agent": self.name}
        docs = await asyncio.to_thread(data_search, context.query)
        if not docs:
            msg = (
                "I don't have that data documented yet — check with the Device Reporting & "
                "Production team."
            )
            yield {"type": "content", "content": msg, "agent": self.name}
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": msg, "steps": [], "conversation_history": context.conversation_history}}}
            return
 
        answer = ""
        answered = True
        header_done = False
        buf = ""
        try:
            async for chunk in generate_grounded_answer_stream(
                context.query, docs, _DATA_SYSTEM_PROMPT, context.conversation_history, "Data context"
            ):
                if not chunk:
                    continue
                if not header_done:
                    buf += chunk
                    if "\n" in buf or len(buf) > 16:
                        up = buf.lstrip().upper()
                        if up.startswith("NO_INFO"):
                            answered, remainder = False, buf.lstrip()[len("NO_INFO"):]
                        elif up.startswith("ANSWERED"):
                            answered, remainder = True, buf.lstrip()[len("ANSWERED"):]
                        else:
                            answered, remainder = True, buf
                        remainder = remainder.lstrip(" :-\n")
                        header_done = True
                        if remainder:
                            answer += remainder
                            yield {"type": "content", "content": remainder, "agent": self.name}
                    continue
                answer += chunk
                yield {"type": "content", "content": chunk, "agent": self.name}
            if not header_done and buf:
                up = buf.lstrip().upper()
                answered = not up.startswith("NO_INFO")
                remainder = buf.lstrip()
                for pfx in ("ANSWERED", "NO_INFO"):
                    if up.startswith(pfx):
                        remainder = remainder[len(pfx):]
                        break
                remainder = remainder.lstrip(" :-\n")
                if remainder:
                    answer += remainder
                    yield {"type": "content", "content": remainder, "agent": self.name}
        except Exception as e:
            logger.error("[Data] generation error: %s", e)
            yield {"type": "content", "content": f"Error generating answer: {e}", "agent": self.name}
 
        citations, seen = [], set()
        for d in docs:
            for s in d.get("sources", []):
                url = s.get("url")
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"title": s.get("title", "Source"), "url": url})
        shown = citations if answered else []
        if shown:
            footer = "\n\n**Sources:**\n" + "\n".join(
                f"- [{c['title']}]({c['url']})" for c in shown
            )
            answer += footer
            yield {"type": "content", "content": footer, "agent": self.name}
 
        yield {"type": "final", "result": {"agent": self.name, "output": {
            "answer": answer, "steps": [], "sources": shown,
            "retrieved_docs": docs, "conversation_history": context.conversation_history}}}
 
 
 
# ---------------------------------------------------------------------------
# Device Health (Device 360) agent
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Shared helper: recover the real device name after stripping a picker-round-trip
# marker ("— all variants" / "(only)") off the raw query text.
# ---------------------------------------------------------------------------
def _recover_device_name(
    raw: str, stripped: str, extract_device_query, fallback_device: Optional[str] = None,
) -> str:
    """`stripped` is `raw` with the trailing marker regex-removed. Three cases reach here:
 
    1. A genuine frontend button click: `stripped` IS the device name, verbatim
       (e.g. "Motorola Razr+ 2025"). extract_device_query() alone can under-match this
       (it only returns "Motorola" — see the comment at each call site) so we must NOT
       just trust extraction; we want to keep the fuller literal text.
    2. A follow-up query that got LLM-rewritten before reaching here (e.g. "overview
       card for the same" -> "device overview card for iPhone 18 — all variants").
       The marker-strip regex only removes the trailing marker, leaving leading prose
       ("device overview card for iPhone 18") — NOT just the device name.
    3. A typed (not clicked) reply to a variant picker that never repeats the device
       name at all (e.g. the picker asked about "iPhone 18" and the user just typed
       "both series aggregated" or "all variants"). Neither the marker-strip regex nor
       extract_device_query() has anything to find here — without `fallback_device`
       (the family name the picker prompt was actually about, threaded in via
       `_pending_picker_device` state) this used to silently return the literal typed
       text as the "device", producing a garbled all-N/A card instead of the real answer.
 
    We distinguish 1/2 by locating the regex-extracted device token inside `stripped`:
    if it starts at (or near) position 0, `stripped` is trustworthy as-is (case 1 —
    nothing precedes the device name). If it's found further in, there's leading prose
    to drop (case 2) — use the text from that point on instead. If no device token is
    found at all (case 3), prefer `fallback_device` when the caller supplied one;
    otherwise fall back to `stripped` (or `raw` if that's empty) rather than passing an
    empty string downstream.
    """
    if not stripped:
        return extract_device_query(raw) or fallback_device or raw
    candidate = extract_device_query(stripped)
    if not candidate:
        return fallback_device or stripped
    idx = stripped.lower().find(candidate.lower())
    if idx <= 0:
        return stripped
    return stripped[idx:].strip()
 
 
class DeviceHealthAgent(OrchestratorAgent):
    """Auto-detected 12-point device profile. Selected when a query is a GENERAL question about a
    device ("device health for X", "tell me about the iPhone 16", or just a bare device name),
    as opposed to a specific metric ("iPhone 16 sales last week") which the SQL agent handles.
 
    No LLM: it runs the curated device_health.py service and renders a deterministic markdown
    card, so the answer is fast and identically structured every time. See API/device_health.py."""
 
    # A device question reading as a general/overview ask → the whole card.
    _GENERAL_CUES = (
        "device health", "health", "overview", "tell me about", "how is", "how's", "how are",
        "summary", "profile", "rundown", "everything about", "details on", "detail on",
        "status of", "snapshot", "360", "doing",
    )
    # Specific-metric words: on their own (no general cue) these belong to the SQL agent.
    _SPECIFIC = (
        "sku", "activation", "upgrade", "remorse", "forecast", "inventory", "stock",
        "allocation", "promo", "purchase order", "open order", "backorder", "price", "frp",
        "how many", "units", "revenue", "compare", " vs ",
        # sales-intent words: "sales for iPhone 17" belongs to the SQL agent, not the health card
        "sales", "sold", "selling",
    )
 
    def __init__(self) -> None:
        super().__init__(name="device_health", priority=20)
 
    # Markers the variant-picker buttons round-trip back on (see run_stream).
    _AGG_MARKERS = ("all variants", "all variant", "aggregate")
    _EXACT_MARKER = "(only)"
 
    def score(self, context: AgentContext) -> float:
        from API.device_health import extract_device_query
        # Use the ORIGINAL, unrewritten text here — not context.query, which may have
        # been follow-up-rewritten and can contain marker words the user never typed
        # (see run_stream's identical guard below for the full explanation).
        q = (context.raw_query or context.query or "").lower()
        # DeviceHealthAgent is activated exclusively via mode="device_health" (the Device Health
        # Card button on the frontend). The only exception is picker round-trips, where the user
        # has already clicked aggregate/exact variant buttons from a previous card response.
        if self._EXACT_MARKER in q or any(m in q for m in self._AGG_MARKERS):
            if extract_device_query(q):
                return 0.95
        return 0.0
 
    async def _emit_card(self, device, exact, display):
        from API import device_health
        yield {"type": "status",
               "content": f"Building device overview card for {display}…", "agent": self.name}
        try:
            card = await asyncio.to_thread(device_health.compute_device_health, device, exact)
            card["device"] = display  # what the card title shows (e.g. "iPhone 17 — all variants")
            md = device_health.format_device_health_markdown(card)
        except Exception as e:
            logger.error("[DeviceHealth] failed for %r: %s", device, e, exc_info=True)
            md = f"Sorry — I couldn't build the device overview card for **{display}** ({e})."
            card = {}
        yield {"type": "final", "result": {"agent": self.name, "output": {
            "answer": md, "steps": [], "device_health": card,
            # Plain family/variant name, WITHOUT the "— all variants" display suffix —
            # used by _run_agent_node to remember "last device this agent discussed"
            # for follow-up fallback. Using `card["device"]` (the display string) there
            # instead would double-suffix a later aggregate request.
            "_resolved_device": device}}}
 
    async def run_stream(self, context: AgentContext):
        from API import device_health
        raw = context.query or ""
        ql = raw.lower()
        # Gate the picker-round-trip shortcut on the ORIGINAL, unrewritten text the
        # user/frontend actually sent (context.raw_query) — never on `ql`/context.query
        # alone, which may have been rewritten by the follow-up-detection step. Without
        # this, a fresh, independent question that merely mentions a multi-variant
        # family (e.g. "inventory card for iPhone 18") can get silently answered as
        # "all variants" — skipping the picker entirely — if the follow-up rewriter
        # happens to carry that phrasing forward from an earlier, unrelated turn.
        raw_ql = (context.raw_query or raw).lower()
 
        # Picker round-trip: explicit aggregate / single-variant selection → build that card.
        # IMPORTANT: do NOT use extract_device_query() here — it partially matches and
        # returns only the brand (e.g. "Motorola Razr+ 2025 (only)" → "Motorola").
        # Instead strip the marker directly to get the exact device name the frontend sent.
        # A typed (not clicked) reply to our own picker prompt may not repeat the
        # device name at all ("both series aggregated" / "all variants") — fall back
        # to the family name that prompt was actually about instead of treating the
        # literal reply text as the device (see _recover_device_name case 3).
        _pending = context.state.get("_pending_picker_device") or {}
        _fallback_device = _pending.get("device") if _pending.get("agent") == self.name else None
 
        if any(m in raw_ql for m in self._AGG_MARKERS):
            import re as _re
            stripped = _re.sub(r'\s*(—\s*)?(all variants?|all variant|aggregate(?:d)?)\s*$', '', raw, flags=_re.IGNORECASE).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
            async for c in self._emit_card(device, False, f"{device} — all variants"):
                yield c
            return
        if self._EXACT_MARKER in raw_ql:
            import re as _re
            stripped = _re.sub(r'\s*\(only\)\s*$', '', raw, flags=_re.IGNORECASE).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
            async for c in self._emit_card(device, True, device):
                yield c
            return
 
        # Require a recognisable device name in the query.
        # If no device found → this query doesn't belong to DeviceHealthAgent.
        # Make the LLM mismatch call to figure out the right agent and suggest
        # the user switch tab — exactly like the normal mismatch routing flow.
        device = device_health.extract_device_query(raw)
        if not device:
            ql_check = ql.strip()
            is_what_is = any(k in ql_check for k in (
                "what is", "what's", "whats", "explain", "tell me about",
                "describe", "how does", "how do i use", "what does",
                "device overview card", "overview card", "device health card", "health card",
            ))
            # "What is device overview card?" — this IS about DHC, answer it
            if is_what_is and any(k in ql_check for k in (
                "device overview", "overview card", "device health", "health card", "dhc",
            )):
                msg = (
                    "**Device Overview Card** gives you a 12-point snapshot of any T-Mobile device:\n\n"
                    "- **Launch date & FRP range**\n"
                    "- **Total lifetime sales** (Magenta / A-Stock)\n"
                    "- **Predecessor device**\n"
                    "- **Last 6-week sales trend**\n"
                    "- **Next 4-week forecast**\n"
                    "- **Inventory health & weeks of supply & RSO%**\n"
                    "- **Open purchase orders**\n"
                    "- **Open customer orders & backorders**\n"
                    "- **Active promotions**\n"
                    "- **Full SKU list**\n\n"
                    "Which device would you like the overview card for? "
                    "*(e.g. iPhone 17 Pro, Galaxy S26 Ultra, Pixel 9)*"
                )
                yield {"type": "content", "content": msg, "agent": self.name}
                yield {"type": "final", "result": {"agent": self.name, "output": {
                    "answer": msg, "steps": []}}}
                return
 
            # Not a DHC question and no device name. The router already ran the LLM
            # mismatch check before entering this node — if the query belonged elsewhere,
            # it would have been routed to that agent's own node instead of here (see
            # router_node's device_health/inventory_health handling). So reaching this
            # point means it's a genuine DHC question missing a device name.
            msg = (
                "**Device Overview Card** gives you a 12-point snapshot of any T-Mobile device "
                "— launch date, lifetime sales, inventory, forecast, open orders, promotions, SKUs & more.\n\n"
                "Which device would you like the overview card for? "
                "*(e.g. iPhone 17 Pro, Galaxy S26 Ultra, Pixel 9)*"
            )
            yield {"type": "content", "content": msg, "agent": self.name}
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": msg, "steps": []}}}
            return
 
        # A device name was found. The query may name a SPECIFIC metric word (sales,
        # inventory, SKU, forecast, etc.) that would normally belong elsewhere per
        # agent_metadata.json's "not_for" rules — but the router already ran the LLM
        # mismatch check before entering this node and would have routed a genuine
        # mismatch to that agent's own node instead. Reaching this point means the
        # query legitimately belongs here (e.g. general phrasing that happens to
        # contain a cue word), so just build the overview card.
 
        # Existence/lifecycle validation (non-existent / not-yet-launched devices) now
        # happens centrally in router_node's device-existence gate, BEFORE this node ever
        # runs — see _device_existence_gate(). Nothing to do here anymore.
 
        # Fresh device question: if it names a multi-variant family, ask which view they want
        # (aggregate vs a specific model) instead of silently aggregating.
        yield {"type": "status", "content": f"Looking up {device}…", "agent": self.name}
        try:
            variants = await asyncio.to_thread(device_health.list_variants, device)
        except Exception as e:
            logger.warning("[DeviceHealth] variant lookup failed for %r: %s", device, e)
            variants = []
 
        if len(variants) > 1:
            # If the user named a specific model and it's an exact match in the list,
            # skip the picker and build the card directly — no need to ask when they
            # already told us what they want (e.g. "iPhone 17 Pro" shouldn't surface
            # "iPhone 17 Pro Max" as an option).
            device_lower = device.lower()
            exact = next((v for v in variants if v.lower() == device_lower), None)
            if exact:
                async for c in self._emit_card(exact, True, exact):
                    yield c
                return
 
            options = [{"label": f"All {device} (aggregate)", "query": f"{device} — all variants"}]
            options += [{"label": v, "query": f"{v} (only)"} for v in variants]
            prompt = (
                f"**{device}** has {len(variants)} variants. Want the whole series aggregated, "
                "or a specific model?"
            )
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": prompt, "steps": [],
                "device_options": {"device": device, "options": options, "agent": self.name}}}}
            return
 
        # Exactly one variant resolved (e.g. the query already named a specific model) —
        # use the correctly-cased value from the DB for the display heading, not the
        # raw/mis-cased user text (e.g. "iphone 17 PRo Max" → "iPhone 17 Pro Max").
        if len(variants) == 1:
            v = variants[0]
            async for c in self._emit_card(v, True, v):
                yield c
            return
 
        # Unresolved device (no variants found) → build the card directly with raw text.
        async for c in self._emit_card(device, False, device):
            yield c
 
 # ---------------------------------------------------------------------------
# NPI launch-performance agent (deterministic)
# ---------------------------------------------------------------------------
class NpiLaunchAgent(OrchestratorAgent):
    """Deterministic NPI agent — handles ALL metric intents covered by npi_metrics.py:
    launch performance (orders + YoY + actual vs forecast), inventory health,
    activations/upgrades, top promos, and forward forecast. All for NPI devices only."""
 
    _INVENTORY_CUES = ("inventory", "stock", "weeks of supply", "wos", "rso",
                       "in-transit", "in transit", "stores carrying", "meio",
                       "inventory health", "network inventory")
    _ACTIV_CUES     = ("activation", "upgrade", "new ban", "aal", "add-a-line")
    _PROMO_CUES     = ("promo", "promotion", "promo utilization", "promo takers", "top promo")
    _FORECAST_CUES  = ("next 4 week", "4 week forecast", "forward forecast", "upcoming forecast")
 
    # Markers the variant-picker buttons round-trip back on (see run_stream) — same
    # convention as DeviceHealthAgent.
    _EXACT_MARKER = "(only)"
    _AGG_MARKERS  = ("all variants", "all variant", "aggregate")
 
    def __init__(self) -> None:
        super().__init__(name="npi_launch", priority=15)
 
    def score(self, context: AgentContext) -> float:
        from API.device_health import extract_device_query
        q = (context.query or "").lower()
        if not extract_device_query(q):
            return 0.0
        # Picker round-trip ("iPhone 18 Pro (only)" / "iPhone 18 — all variants") — claim
        # it regardless of cue words, same convention as DeviceHealthAgent/InventoryHealthAgent.
        # Gated on the ORIGINAL, unrewritten text (context.raw_query) — same guard as
        # run_stream below — so a stale marker word the follow-up rewriter bled in from
        # an unrelated earlier turn can't make a fresh, independent question falsely
        # claim to be a round-trip.
        _raw_q = (context.raw_query or context.query or "").lower()
        if self._EXACT_MARKER in _raw_q or any(m in _raw_q for m in self._AGG_MARKERS):
            return 0.95
        if "launch performance" in q or "launch" in q:
            return 0.95
        # Also claim inventory/activations/promos/forecast for NPI devices
        if any(c in q for c in self._INVENTORY_CUES + self._ACTIV_CUES +
               self._PROMO_CUES + self._FORECAST_CUES):
            return 0.90
        return 0.0
 
    async def run_stream(self, context: AgentContext):
        import re as _re
        from API import device_health, npi_metrics
        raw       = context.query or ""
        q         = raw.lower()
        _extracted = device_health.extract_device_query(raw)
        device    = _extracted or raw
 
        # Gate picker-round-trip detection on the ORIGINAL, unrewritten text the user/
        # frontend actually sent (context.raw_query) — never on `q`/context.query alone,
        # which may have been rewritten by the follow-up-detection step. Without this, a
        # fresh, independent question that merely mentions the same device family as an
        # earlier, unrelated aggregate turn (e.g. "what about iPhone 16?" right after
        # asking to aggregate iPhone 18) can get misread as another aggregate round-trip
        # if the rewriter carries that phrasing forward — same guard as DeviceHealthAgent/
        # InventoryHealthAgent's identical comment.
        raw_q = (context.raw_query or raw).lower()
 
        # A typed (not clicked) reply to our own picker prompt may not repeat the
        # device name at all ("both series aggregated" / "all variants") — fall back
        # to the family name that prompt was actually about instead of treating the
        # literal reply text as the device (see _recover_device_name case 3).
        _pending = context.state.get("_pending_picker_device") or {}
        _fallback_device = _pending.get("device") if _pending.get("agent") == self.name else None
 
        # Picker round-trips: the user already answered "which variant?" — strip the
        # marker and skip straight to fetching, no re-disambiguation.
        exact_pick = self._EXACT_MARKER in raw_q
        agg_pick   = any(m in raw_q for m in self._AGG_MARKERS)
        if exact_pick:
            stripped = _re.sub(r'\s*\(only\)\s*$', '', raw, flags=_re.IGNORECASE).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
        elif agg_pick:
            stripped = _re.sub(
                r'\s*(—\s*)?(all variants?|all variant|aggregate(?:d)?)\s*$', '', raw, flags=_re.IGNORECASE
            ).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
 
        if not exact_pick and not agg_pick:
            # NPI's metrics cache is keyed by the exact DeviceName the pipeline uses (e.g.
            # "iPhone 18 Pro", "iPhone 18 Pro Max") — there's no bare "iPhone 18" row, so a
            # generic family name must be disambiguated against the actual NPI device list
            # rather than silently queried as-is (which just returns an all-None/"N/A"
            # metrics payload — see get_npi_devices()).
            try:
                npi_devices = await asyncio.to_thread(npi_metrics.get_npi_devices)
            except Exception as e:
                logger.warning("[NpiLaunch] device list lookup failed for %r: %s", device, e)
                npi_devices = []
            device_lower = device.lower()
            matches = [d for d in npi_devices
                       if device_lower in d.lower() or d.lower() in device_lower]
            exact = next((d for d in matches if d.lower() == device_lower), None)
            if exact:
                device = exact
            elif len(matches) > 1:
                options = [{"label": f"All {device} (aggregate)", "query": f"{device} — all variants"}]
                options += [{"label": m, "query": f"{m} (only)"} for m in matches]
                prompt = (
                    f"**{device}** has {len(matches)} variants. Want the whole series "
                    "aggregated, or a specific model?"
                )
                yield {"type": "final", "result": {"agent": self.name, "output": {
                    "answer": prompt, "steps": [],
                    "device_options": {"device": device, "options": options, "agent": self.name}}}}
                return
            elif len(matches) == 1:
                device = matches[0]
            else:
                # No match in the curated NPI list. A genuinely non-existent/not-yet-
                # launched device is already caught earlier by router_node's device-
                # existence gate, before this agent ever runs (see _device_existence_gate),
                # so reaching here means either:
                if not _extracted:
                    # (a) nothing recognisable as a device at all — don't fabricate a
                    # 12-field N/A report titled with garbage input.
                    msg = (
                        "I couldn't identify a device in that question. Which device "
                        "would you like the launch performance report for?"
                    )
                    yield {"type": "content", "content": msg, "agent": self.name}
                    yield {"type": "final", "result": {"agent": self.name, "output": {
                        "answer": msg, "steps": []}}}
                    return
                # (b) a real, recognised device that simply isn't tracked in NPI launch
                # performance data — ask which of the other cards they want instead of
                # silently rerouting or building a fabricated N/A report. The reply is
                # detected deterministically by router_node from the button's own query
                # text ("Device overview for iPhone 16" / "Inventory health for iPhone
                # 16") — see the npi-scope-fallback check there.
                msg = (
                    f"**{device}** isn't tracked in NPI launch performance data. "
                    "Want the general **Device Overview** or **Inventory Health** for it instead?"
                )
                options = [
                    {"label": "Device Overview", "query": f"Device overview for {device}"},
                    {"label": "Inventory Health", "query": f"Inventory health for {device}"},
                ]
                yield {"type": "final", "result": {"agent": self.name, "output": {
                    "answer": msg, "steps": [],
                    "device_options": {"device": device, "options": options, "agent": self.name}}}}
                return
 
        yield {"type": "status",
               "content": f"Pulling NPI data for {device}…", "agent": self.name}
        try:
            if agg_pick:
                # No per-family row exists in the cache (or in get_npi_devices' own list) to
                # read, so aggregate live: every metric query binds `device` straight into an
                # ILIKE clause (see npi_metrics.py), so a %-wrapped pattern naturally sums/
                # ranges across every matching variant instead of matching one exact row —
                # same trick device_health.compute_device_health(exact=False) already relies on.
                metrics = await asyncio.to_thread(npi_metrics.compute_npi_metrics, f"%{device}%")
                metrics["device"] = f"{device} — all variants"
            else:
                # read_cached_metrics gives us ALL metrics in one call (served from cache)
                metrics = await asyncio.to_thread(npi_metrics.read_cached_metrics, device)
        except Exception as e:
            logger.error("[NpiLaunch] metrics fetch failed for %r: %s", device, e, exc_info=True)
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": f"Sorry — I couldn't retrieve NPI data for **{device}** ({e}).",
                "steps": []}}}
            return
 
        # Intent routing — check most specific first
        # `_resolved_device` (the plain, unsuffixed family/variant name) lets
        # _run_agent_node remember "last device this agent discussed" for a later
        # vague follow-up ("what about all variants aggregated?") — see its comment.
        if any(c in q for c in self._INVENTORY_CUES):
            md = npi_metrics.format_inventory_health_markdown(metrics)
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": md, "steps": [], "_resolved_device": device}}}
            return
        elif any(c in q for c in self._ACTIV_CUES):
            md = npi_metrics.format_activations_markdown(metrics)
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": md, "steps": [], "_resolved_device": device}}}
            return
        elif any(c in q for c in self._PROMO_CUES):
            md = npi_metrics.format_promos_markdown(metrics)
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": md, "steps": [], "_resolved_device": device}}}
            return
        elif any(c in q for c in self._FORECAST_CUES):
            md = npi_metrics.format_forecast_markdown(metrics)
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": md, "steps": [], "_resolved_device": device}}}
            return
 
        # Default: full launch performance — orders + YoY + actual vs forecast.
        # Attach the structured `metrics` dict as `launch_performance` so the frontend can
        # render the rich LaunchPerformanceCard; `answer` still carries the markdown as the
        # fallback (used in Teams and on history reload), same convention as device_health /
        # inventory_health above.
        md = npi_metrics.format_npi_full_markdown(metrics)
        yield {"type": "final", "result": {"agent": self.name, "output": {
            "answer": md, "steps": [], "launch_performance": metrics, "_resolved_device": device}}}
 
# ---------------------------------------------------------------------------
# Inventory-health agent (deterministic)
# ---------------------------------------------------------------------------
class InventoryHealthAgent(OrchestratorAgent):
    """Deterministic "inventory health of X" responder — per-SKU retail inventory from the
    curated inventory_health service, rendered as a standalone card. No LLM, same answer every
    time. Fires on inventory/stock/RSO/WOS questions about a device."""
 
    _CUES = ("inventory health", "inventory", "stock out", "stockout", "out of stock",
             "in stock", "rso", "weeks of supply", "weeks-of-supply", "wos", "stock level")
 
    # Metric words that fall OUTSIDE this card's scope (sales, forecast, activations,
    # launch performance, a named SKU/material number, comparisons, or a general device
    # overview ask) — on their own these signal the question belongs elsewhere (usually
    # SQL, or DeviceHealthAgent for the overview cues), per agent_metadata.json's
    # "not_for" rules for this agent. Deliberately excludes words that ARE part of the
    # Inventory Health Card itself (inventory, stock, wos, rso, in-transit, purchase order).
    _SPECIFIC = (
        "sku", "material number", "sales", "sold", "selling", "revenue", "forecast",
        "activation", "upgrade", "launch performance", "sell-through", "sell through",
        "channel mix", "promo", "promotion", "compare", " vs ",
        # general device-overview cues — mirrors DeviceHealthAgent._SPECIFIC including
        # "inventory" so the reverse direction (IHC → DHC) can also auto-route. Excludes
        # the bare word "health card" — it would false-positive on "inventory health card".
        "device overview", "device health", "device profile", "device summary",
        "overview card",
    )
    _SKU_NUMBER_RE = re.compile(r"\b\d{6,}\b")
 
    # Markers the variant-picker buttons round-trip back on (see run_stream) — same
    # convention as DeviceHealthAgent so the shared `device_options` UI component works.
    _AGG_MARKERS = ("all variants", "all variant", "aggregate")
    _EXACT_MARKER = "(only)"
 
    def __init__(self) -> None:
        super().__init__(name="inventory_health", priority=15)
 
    def score(self, context: AgentContext) -> float:
        from API.device_health import extract_device_query
        q = (context.query or "").lower()
        if not extract_device_query(q):
            return 0.0
        if any(c in q for c in self._CUES):
            return 0.95
        return 0.0
 
    async def _emit_card(self, device, exact, display):
        from API import inventory_health
        yield {"type": "status",
               "content": f"Pulling inventory health for {display}…", "agent": self.name}
        try:
            data = await asyncio.to_thread(inventory_health.compute_inventory_health, device, exact)
            data["device"] = display  # what the card title shows (e.g. "iPhone 17 — all variants")
            md = inventory_health.format_inventory_health_markdown(data)
        except Exception as e:
            logger.error("[InventoryHealth] failed for %r: %s", device, e, exc_info=True)
            md = f"Sorry — I couldn't pull inventory health for **{display}** ({e})."
            data = {}
        yield {"type": "final", "result": {"agent": self.name, "output": {
            "answer": md, "steps": [], "inventory_health": data,
            # Plain family/variant name, WITHOUT the "— all variants" display suffix —
            # see DeviceHealthAgent._emit_card's identical comment.
            "_resolved_device": device}}}
 
    async def run_stream(self, context: AgentContext):
        from API import device_health, inventory_health
        raw = context.query or ""
        ql = raw.lower()
        # Gate the picker-round-trip shortcut on the ORIGINAL, unrewritten text the
        # user/frontend actually sent (context.raw_query) — see DeviceHealthAgent's
        # identical guard for the full explanation. Without this, a fresh, independent
        # question that merely mentions a multi-variant family (e.g. "inventory card
        # for iPhone 18") can get silently answered as "all variants" — skipping the
        # picker entirely — if the follow-up rewriter carries that phrasing forward
        # from an earlier, unrelated turn.
        raw_ql = (context.raw_query or raw).lower()
 
        # Picker round-trip: explicit aggregate / single-variant selection → build that card.
        # IMPORTANT: do NOT use extract_device_query() here — it partially matches and
        # returns only the brand (e.g. "Motorola Razr+ 2025 (only)" → "Motorola").
        # Instead strip the marker directly to get the exact device name the frontend sent.
        # A typed (not clicked) reply to our own picker prompt may not repeat the
        # device name at all ("both series aggregated" / "all variants") — fall back
        # to the family name that prompt was actually about instead of treating the
        # literal reply text as the device (see _recover_device_name case 3).
        _pending = context.state.get("_pending_picker_device") or {}
        _fallback_device = _pending.get("device") if _pending.get("agent") == self.name else None
 
        if any(m in raw_ql for m in self._AGG_MARKERS):
            import re as _re
            stripped = _re.sub(r'\s*(—\s*)?(all variants?|all variant|aggregate(?:d)?)\s*$', '', raw, flags=_re.IGNORECASE).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
            async for c in self._emit_card(device, False, f"{device} — all variants"):
                yield c
            return
        if self._EXACT_MARKER in raw_ql:
            import re as _re
            stripped = _re.sub(r'\s*\(only\)\s*$', '', raw, flags=_re.IGNORECASE).strip()
            device = _recover_device_name(raw, stripped, device_health.extract_device_query, _fallback_device)
            async for c in self._emit_card(device, True, device):
                yield c
            return
 
        # Require a recognisable device name in the query.
        # If no device found → this query doesn't belong to InventoryHealthAgent.
        # Make the LLM mismatch call to figure out the right agent and suggest
        # the user switch tab — exactly like DeviceHealthAgent's equivalent flow.
        device = device_health.extract_device_query(raw)
        if not device:
            ql_check = ql.strip()
            is_what_is = any(k in ql_check for k in (
                "what is", "what's", "whats", "explain", "tell me about",
                "describe", "how does", "how do i use", "what does",
                "inventory health card", "health card",
            ))
            # "What is inventory health card?" — this IS about IHC, answer it
            if is_what_is and any(k in ql_check for k in ("inventory health", "inventory health card", "ihc")):
                msg = (
                    "**Inventory Health Card** gives you a per-SKU and family-level inventory "
                    "snapshot for any T-Mobile device:\n\n"
                    "- **RNI (retail network inventory)**\n"
                    "- **In-transit units**\n"
                    "- **Open purchase orders**\n"
                    "- **Weighted weeks of supply (WOS)**\n"
                    "- **Full SKU breakdown**\n\n"
                    "Which device would you like the inventory health card for? "
                    "*(e.g. iPhone 17 Pro, Galaxy S26 Ultra, Pixel 9)*"
                )
                yield {"type": "content", "content": msg, "agent": self.name}
                yield {"type": "final", "result": {"agent": self.name, "output": {
                    "answer": msg, "steps": []}}}
                return
 
            # Not an IHC question and no device name. The router already ran the LLM
            # mismatch check before entering this node — if the query belonged elsewhere,
            # it would have been routed to that agent's own node instead of here (see
            # router_node's device_health/inventory_health handling). So reaching this
            # point means it's a genuine IHC question missing a device name.
            msg = (
                "**Inventory Health Card** gives you a per-SKU and family-level inventory "
                "snapshot for any T-Mobile device — RNI, in-transit, open purchase orders, "
                "weighted weeks of supply.\n\n"
                "Which device would you like the inventory health card for? "
                "*(e.g. iPhone 17 Pro, Galaxy S26 Ultra, Pixel 9)*"
            )
            yield {"type": "content", "content": msg, "agent": self.name}
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": msg, "steps": []}}}
            return
 
        # A device name was found. The query may name a metric word OUTSIDE this card's
        # scope (a named SKU/material number, sales, forecast, etc.) that would normally
        # belong elsewhere per agent_metadata.json's "not_for" rules — but the router
        # already ran the LLM mismatch check before entering this node and would have
        # routed a genuine mismatch to that agent's own node instead. Reaching this point
        # means the query legitimately belongs here, so just build the inventory card.
 
        # Existence/lifecycle validation (non-existent / not-yet-launched devices) now
        # happens centrally in router_node's device-existence gate, BEFORE this node ever
        # runs — see _device_existence_gate(). A not-yet-launched device never reaches this
        # agent at all, so "no availability/purchase info before launch" is still guaranteed.
 
        # Fresh device question: if it names a multi-variant family, ask which view they want
        # (aggregate vs a specific model) instead of silently aggregating.
        yield {"type": "status", "content": f"Looking up {device}…", "agent": self.name}
        try:
            variants = await asyncio.to_thread(device_health.list_variants, device)
        except Exception as e:
            logger.warning("[InventoryHealth] variant lookup failed for %r: %s", device, e)
            variants = []
 
        if len(variants) > 1:
            # If the user named a specific model and it's an exact match in the list,
            # skip the picker and build the card directly.
            device_lower = device.lower()
            exact = next((v for v in variants if v.lower() == device_lower), None)
            if exact:
                async for c in self._emit_card(exact, True, exact):
                    yield c
                return
 
            options = [{"label": f"All {device} (aggregate)", "query": f"{device} — all variants"}]
            options += [{"label": v, "query": f"{v} (only)"} for v in variants]
            prompt = (
                f"**{device}** has {len(variants)} variants. Want the whole series aggregated, "
                "or a specific model?"
            )
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": prompt, "steps": [],
                "device_options": {"device": device, "options": options, "agent": self.name}}}}
            return
 
        # Exactly one variant resolved (e.g. the query already named a specific model) —
        # use the correctly-cased value from the DB for the display heading, not the
        # raw/mis-cased user text (e.g. "iphone 17 PRo Max" → "iPhone 17 Pro Max").
        if len(variants) == 1:
            v = variants[0]
            async for c in self._emit_card(v, True, v):
                yield c
            return
 
        # Unresolved device (no variants found) → build the card directly with raw text.
        async for c in self._emit_card(device, False, device):
            yield c
 
 
 
# ---------------------------------------------------------------------------
# Replenishment Information agent (deterministic) — Retail Allocation Planner
# ---------------------------------------------------------------------------
# Modes on which a SKU + store question is answered with the replenishment card inline.
# (Onboarding / OEM / o9 tabs are never hijacked — those tabs stay locked to their agent.)
_REPLENISHMENT_ELIGIBLE_MODES = frozenset(
    {"", "sql", "npi", "device_health", "inventory_health", "replenishment"}
)
 
 
class ReplenishmentAgent(OrchestratorAgent):
    """Deterministic "Replenishment Information" card for ONE SKU in ONE store.
 
    Selected (see _router_decide) when the user — with or without the Retail Allocation
    Planner role — asks for replenishment information for a SKU in a store, or just sends a
    SKU and a store number. The frontend prepends "Hi, I'm the Retail Allocation Planner,"
    when that role is selected; the prefix is stripped before parsing.
 
    No LLM: it runs the curated API/replenishment_info.py service and returns both the
    structured payload (`replenishment_info`, rendered by ReplenishmentCard.jsx) and a
    markdown fallback in `answer` (used in Teams and on history reload).
 
    If the SKU or store is missing it asks for the missing one and remembers the half it has
    in `_pending_replenishment`, so a bare "7920" reply completes the question.
    """
 
    def __init__(self) -> None:
        super().__init__(name="replenishment", priority=15)
 
    def score(self, context: AgentContext) -> float:
        raw = context.raw_query or context.query or ""
        return 0.95 if _repl.classify_query(raw) in ("explicit", "ids_both") else 0.0
 
    @staticmethod
    def _reply(agent_name: str, msg: str, pending: Optional[Dict[str, Any]], *, keep_pending: bool = True):
        output: Dict[str, Any] = {"answer": msg, "steps": []}
        if keep_pending:
            output["_pending_replenishment"] = pending
        return [
            {"type": "content", "content": msg, "agent": agent_name},
            {"type": "final", "result": {"agent": agent_name, "output": output}},
        ]
 
    async def run_stream(self, context: AgentContext):
        state = context.state or {}
        ref = _repl.parse_replenishment_query(context.query or "")
        sku, store = ref["sku"], ref["store"]
 
        # A short reply to our own "which store?" / "which SKU?" prompt (or "what about store
        # 7921") — the router flagged it; fill the missing half from what we remembered.
        pending = state.get("_pending_replenishment") or {}
        if state.get("_replenishment_followup") and pending:
            # The id comes from what the user actually typed (e.g. "1SMR"), not from a
            # paraphrase; whatever it doesn't supply is filled from what we remembered.
            reply = _repl.parse_pending_reply(context.raw_query or context.query or "", pending)
            sku = reply["sku"] or sku or pending.get("sku")
            store = reply["store"] or store or pending.get("store")
 
        if not sku and not store:
            msg = ("Replenishment information is shown for one SKU in one store. "
                   "Send both, for example: **SKU 610214681249 in Store 7920**.")
            for c in self._reply(self.name, msg, None, keep_pending=False):
                yield c
            return
        if not store:
            msg = (f"Which store should I check for SKU **{sku}**? "
                   "Reply with the store number, for example **7920**.")
            for c in self._reply(self.name, msg, {"sku": sku, "store": None, "awaiting": "store"}):
                yield c
            return
        if not sku:
            msg = (f"Which SKU should I check for Store **{store}**? "
                   "Reply with the SKU (material number), for example **610214681249**.")
            for c in self._reply(self.name, msg, {"sku": None, "store": store, "awaiting": "sku"}):
                yield c
            return
 
        yield {"type": "status",
               "content": f"Pulling replenishment information for SKU {sku} in Store {store}…",
               "agent": self.name}
        card = await asyncio.to_thread(_repl.compute_replenishment_info, sku, store)
        md = _repl.format_replenishment_markdown(card)
        remembered = {"sku": sku, "store": store, "awaiting": None}
 
        if card.get("status") == "ok":
            yield {"type": "final", "result": {"agent": self.name, "output": {
                "answer": md, "steps": [], "replenishment_info": card,
                "_pending_replenishment": remembered}}}
            return
        # not_found / error → plain text (no card). Remember the SKU+store on not_found so
        # "what about store 7921" still works; leave pending untouched on a hard error.
        for c in self._reply(self.name, md, remembered, keep_pending=card.get("status") == "not_found"):
            yield c
 
 
# ---------------------------------------------------------------------------
# Agent registry & routing tables
# ---------------------------------------------------------------------------
 
AGENT_REGISTRY: Dict[str, OrchestratorAgent] = {
    "rag": RAGAgent(),
    "contract": ContractManagementAgent(),
    "sql": SQLAgent(),
    "onboarding": OnboardingAgent(),
    "data": DataAgent(),
    "device_health": DeviceHealthAgent(),
    "npi_launch": NpiLaunchAgent(),
    "inventory_health": InventoryHealthAgent(),
    "replenishment": ReplenishmentAgent(),
}
 
DEFAULT_AGENT_NAME = "sql"
 
MODE_TO_AGENT: Dict[str, str] = {
    "sql":           "sql",          # "Ask questions on our Data" tab
    "oem":           "contract",     # "OEM TermSheets" tab (access-controlled)
    "o9":            "rag",          # "o9 User Manual" tab
    "npi":           "npi_launch",   # "NPI" tab → typed question, goes through LLM check
    "onboarding":    "onboarding",   # "Onboarding" tab (DataAgent if team=data-bi-solutions)
    "data":          "data",         # direct API / teams-bot access only
    "device_health": "device_health", # Device Overview Card (reached via redirect, no button)
    "inventory_health": "inventory_health", # Inventory Health Card (reached via redirect/toggle)
    "npi_launch":    "npi_launch",   # NPI variant-picker round-trip (Teams plain-text follow-up)
}
 
# Human-friendly tab names + the `mode` value the UI uses, keyed by the agent a
# tab maps to. Used to recommend a switch when a locked tab gets an out-of-context
# question (see _detect_tab_mismatch).
AGENT_TAB_LABELS: Dict[str, str] = {
    "contract":      "OEM TermSheets",
    "rag":           "o9 User Manual",
    "data":          "Onboarding · Data & BI Solutions",
    "onboarding":    "Onboarding",
    "sql":           "Ask questions on our Data",
    "npi_launch":    "NPI",            # mismatch messages show "NPI tab" correctly
    "device_health": "Device Overview", # Device Overview Card
    "replenishment": "Replenishment Information", # Replenishment Information Card
}
AGENT_TO_MODE: Dict[str, str] = {
    "contract":      "oem",
    "rag":           "o9",
    "data":          "onboarding",
    "onboarding":    "onboarding",
    "sql":           "sql",
    "npi_launch":    "npi",            # recommended_mode="npi" in switch-tab payloads
    "device_health": "device_health",  # maps back to its own mode
    "inventory_health": "inventory_health",  # maps back to its own mode
    "replenishment": "sql",  # the card has no tab of its own; it lives on the "sql" tab
}
 
# Label for the default "Ask questions on our Data" tab (the SQL agent).
_DEFAULT_TAB_LABEL = "Ask questions on our Data"
 
 
# all tabs. Special case: onboarding + team=data-bi-solutions → DataAgent.
def _current_tab_agent(state) -> str:
    mode = (state.get("mode") or "").strip().lower()
    team = (state.get("team") or "").strip().lower()
    # Onboarding tab + Data & BI Solutions team → DataAgent
    if mode == "onboarding" and team == "data-bi-solutions":
        return "data"
    # npi mode now maps to npi_launch via MODE_TO_AGENT — no special case needed
    if mode in MODE_TO_AGENT:
        return MODE_TO_AGENT[mode]
    if team:
        return "onboarding"
    return "sql"
 
 
# _detect_agent_mismatch removed — replaced by _llm_metadata_route().
 
# _rank_agents removed — routing is fully handled by MODE_TO_AGENT + LLM metadata check.
 
# ---------------------------------------------------------------------------
# RBAC constants
# ---------------------------------------------------------------------------
 
_SQL_O9_ROLES: frozenset = frozenset(
    {"Developers", "Business_Analysts", "Administrators"}
)
 
# Ambiguous keywords that could apply to both SQL and RAG agents
# _AMBIGUOUS_KEYWORDS removed — clarification flow removed, mode always set.
 
 
 
# ---------------------------------------------------------------------------
# Role extraction helper
# ---------------------------------------------------------------------------
 
 
_ROLE_INTRO_RE = r"\b(?:i\s*am|i\s*'m|i\s*\u2019m|im)\s+(?:the\s+|a\s+|an\s+)?"
_ROLE_PHRASES = {
    "retail_allocation_planner": r"retail\s+allocation\s+planner",
    "demand_planner": r"demand\s+planner",
}
 
 
def extract_role_from_text(text: str) -> str | None:
    """Detect "I am a <role>" / "I'm the <role>" (any of a/an/the, am/'m) in free text."""
    if not text:
        return None
    text_lower = text.lower()
    return next(
        (
            role
            for role, phrase in _ROLE_PHRASES.items()
            if re.search(_ROLE_INTRO_RE + phrase, text_lower)
        ),
        None,
    )
 
 
# detect_ambiguous_keyword removed.
 
# parse_clarification_response removed.
 
# _build_clarification_msg removed.
 
async def _yield_orchestrator_response(message: str, conversation_history: list):
    yield {
        "stream_chunk": {
            "type": "content",
            "content": message,
            "agent": "orchestrator",
        },
        "agent": "orchestrator",
    }
    yield {
        "result": {
            "agent": "orchestrator",
            "output": {
                "answer": message,
                "steps": [],
                "conversation_history": conversation_history,
            },
        },
    }
 
 
async def _yield_access_denied(message: str, conversation_history: list):
    async for chunk in _yield_orchestrator_response(message, conversation_history):
        yield chunk
 
 
# ---------------------------------------------------------------------------
# RBAC constants — module-level so router_node and access_denied_node can use them
# ---------------------------------------------------------------------------
 
_CONTRACT_ONLY_MSG = (
    "Your account only has access to contract-related questions. "
    "Please contact your administrator if you need access to data queries."
)
 
    # ---------------------------------------------------------------------------
# Agent metadata loading
# ---------------------------------------------------------------------------
 
_METADATA_PATH = _pathlib.Path(__file__).parent / "agent_metadata.json"
try:
    with open(_METADATA_PATH, "r") as _f:
        _AGENT_METADATA: Dict = json.load(_f)
    logger.info(
        "[Metadata] Loaded agent_metadata.json — %d agents",
        len(_AGENT_METADATA.get("agents", {})),
    )
except Exception as _meta_err:
    logger.warning("[Metadata] Could not load agent_metadata.json: %s", _meta_err)
    _AGENT_METADATA = {}
 
_DATA_BI_TEAM_VALUE = "data-bi-solutions"
 
 
# Agents the LLM tab-checker must never see or suggest. The replenishment card is chosen ONLY by
# the rule-based router block in _router_decide (see _repl.classify_query) — if the LLM could
# pick it, "sales for <sku>" typed on another tab got switched over and answered by the card.
# It stays DESCRIBED in agent_metadata.json (documentation) but is marked "llm_routable": false;
# "replenishment" is also hard-coded here so a copy of that file without the flag can't re-enable it.
_ALWAYS_HIDDEN_FROM_LLM = frozenset({"replenishment"})
 
 
def _llm_hidden_agents() -> frozenset:
    """Agent keys the tab-checker must not see: hard-coded ones + any with "llm_routable": false."""
    flagged = {
        k for k, v in _AGENT_METADATA.get("agents", {}).items()
        if isinstance(v, dict) and v.get("llm_routable") is False
    }
    return _ALWAYS_HIDDEN_FROM_LLM | flagged
 
 
def _build_metadata_prompt(query: str, active_mode: str, team: str = "") -> str:
    """Build the prompt for the tiny LLM routing call."""
    _hidden = _llm_hidden_agents()
    agents = {k: v for k, v in _AGENT_METADATA.get("agents", {}).items() if k not in _hidden}
    agent_descriptions = []
    for agent_key, info in agents.items():
        signals = "\n      - ".join(info.get("signals", []))
        agent_descriptions.append(
            f'  Agent: "{agent_key}" | Tab: "{info["tab"]}"\n'
            f'  Description: {info["description"]}\n'
            f'  Signals:\n      - {signals}'
        )
    agents_block = "\n\n".join(agent_descriptions)
 
    active_agent_key = _current_tab_agent({"mode": active_mode, "team": team})
    active_info = agents.get(active_agent_key, {})
    active_tab_label = active_info.get("tab", active_mode)
    if active_mode == "onboarding" and team == _DATA_BI_TEAM_VALUE:
        active_tab_label = "Onboarding · Data & BI Solutions"
 
    return f"""You are a query router for InsightBot, a T-Mobile internal chatbot.
 
The user is on the **"{active_tab_label}"** tab (mode="{active_mode}", team="{team}").
The user asked: "{query}"
 
Available agents:
 
{agents_block}
 
Decide if the user question MATCHES the current tab ("{active_tab_label}").
- If YES  → output exactly: MATCH
- If NO   → output exactly: MISMATCH:<agent_key>:<one sentence reason>
  where <agent_key> is one of: sql, npi_launch, contract, rag, onboarding, data, device_health, inventory_health
 
Rules:
- Be strict — if the question clearly belongs to a different agent say MISMATCH.
- NPI tab (mode=npi): Valid ONLY for a GENERIC, unqualified "launch performance" question
  about an NPI device (Apple iPhone 18 Pro, iPhone 18 Pro Max, Galaxy Z Fold8/Flip8/Fold8 Ultra, Google Pixel 11/Pro/Pro Fold/Pro XL)
  with no specific sub-metric named — e.g. "launch performance of Galaxy Z Fold8", "how is
  Pixel 11's launch going", "current launch performance of iPhone 18 Pro Max". This renders the
  full bundled Launch Performance card.
  EXCEPTION (inventory_health): if the question explicitly uses one of these exact phrases —
  "inventory health", "inventory health card", "inventory status", "inventory summary",
  "inventory overview", "inventory overview card" — with NO specific SKU/material number named
  (e.g. "inventory card for pixel 11 pro", "give me inventory overview for the same") →
  MISMATCH:inventory_health, NOT sql, because the Inventory Health Card is a separate agent
  from general sales/SQL data.
  EXCEPTION (device_health): if the question is a general device overview/status ask with NO
  specific sub-metric named — including exact phrases "device overview", "device overview
  card", "device health", "device health card", "device summary","device status", "device profile", or a bare
  "overview"/"overview for <device>" ask (e.g. "give me device overview for the same",
  "overview for Pixel 11 Pro Fold", "device summary for iPhone 18") →
  MISMATCH:device_health, NOT sql, because the Device Overview Card is a separate agent from
  general sales/SQL data and from the NPI tab's launch-performance card.
  Otherwise MISMATCH:sql if: device is not an NPI device, OR the question names ANY specific
  sub-metric about the NPI device — sales, orders to date, pre-orders, YoY vs previous gen,
  forecast, actual vs forecast, forward forecast, weekly orders curve,
  inventory/stock levels/WOS/RSO/in-transit (when NOT one of the exact inventory_health
  phrases above), activations, upgrades, promos/promo utilization — even though that same
  metric also appears inside the NPI tab's own launch-performance card. Also MISMATCH:sql for
  channel mix / SKU-level lookup / general OEM rankings unrelated to an NPI device.
- SQL tab (mode=sql): General data questions are valid here, INCLUDING questions about NPI
  devices that name a specific metric (YoY vs previous gen, orders to date, forecast, actual
  vs forecast, forward forecast, weekly orders curve, inventory, activations/upgrades, promos,
  general sales) → MATCH. Naming a specific metric about an NPI device does NOT trigger a
  mismatch just because that same metric happens to also be part of the NPI tab's bundled
  launch-performance card — see EXCEPTION 0 below for the one narrow case that does mismatch.
  EXCEPTION 0: a GENERIC "launch performance" question about an NPI device (Apple iPhone 18 Pro, iPhone 18 Pro Max, Galaxy Z Fold8 /
  Fold8 Ultra / Flip8, Google Pixel 11 / Pixel 11 Pro / Pixel 11 Pro Fold / Pixel 11 Pro XL)
  with NO specific sub-metric named — e.g. "launch performance of Galaxy Z Fold8", "how is
  Pixel 11's launch going/performing", "give me the launch performance card for Flip8" —
  is NOT a sql match → MISMATCH:npi_launch.
  This EXCEPTION 0 is narrow and does NOT apply to:
    (a) a NON-NPI device — e.g. "current launch performance of iPhone 17" stays a sql MATCH.
    (b) a SPECIFIC named sub-metric about an NPI device, even though that sub-metric is one
        of the things the launch-performance card bundles together — e.g. "YoY metric for
        Galaxy Z Fold8", "orders to date for Flip8", "actual vs forecast for Pixel 11 Pro",
        "weekly orders curve for Fold8 Ultra" all stay a sql MATCH, because the question
        names a specific metric rather than asking generically for "launch performance."
        Only the bare, unqualified "launch performance" ask triggers MISMATCH:npi_launch.
  EXCEPTION 1: a general overview question about ONE specific device with no metric named
  (e.g. "tell me about iPhone 17e", "give me overview of Samsung A37", "how is the Galaxy
  S26 doing", or just a bare device name) is NOT a sql match → MISMATCH:device_health.
  EXCEPTION 2: a general overview question about a whole BRAND with no specific model and no
  metric named (e.g. just "Samsung", "overview of Motorola", "iPhone") is NOT a sql match →
  MISMATCH:device_health — this brand case still applies even without a named model, because
  the Device Overview Card lists that brand's devices for the user to pick from.
  EXCEPTION 3: an inventory/stock question about ONE device or a whole BRAND — i.e. a question
  that explicitly uses one of these exact phrases: "inventory health", "inventory health card",
  "inventory status", "inventory summary" — with NO specific SKU/material number named (e.g.
  "inventory status for Samsung", "inventory health of iPhone 17", "give me the inventory
  summary for Galaxy S26 Ultra") is NOT a sql match → MISMATCH:inventory_health — applies with
  or without a named model, because the Inventory Health Card lists that brand's devices for
  the user to pick from.
  This EXCEPTION 3 is narrow and does NOT apply to any other inventory/stock wording, even
  without a SKU — e.g. "inventory for iPhone 17", "stock levels for iPhone 17", "is Motorola in
  stock", "weeks of supply for Samsung", "WOS for Galaxy S26", "RSO for Motorola", "is Samsung
  out of stock" — ALL of these stay a sql MATCH; only the four exact phrases above ("inventory
  health", "inventory health card", "inventory status", "inventory summary") trigger the card.
  A question naming a specific SKU/material number (e.g. "what is the inventory for SKU
  610214688811") also stays a sql MATCH.
- Check the EXCEPTIONS above and each agent's "not_for" list FIRST, before anything
  else — they take precedence even if a positive "signal" for the current tab also
  seems to match (e.g. a question using the exact phrase "inventory status" or
  "inventory health" with no SKU matches the SQL tab's general "inventory" signal
  too, but EXCEPTION 3 / inventory_health's not_for still wins — output
  MISMATCH:inventory_health. Any OTHER inventory/stock wording, e.g. "inventory for
  iPhone 17", "stock levels for iPhone 17", "weeks of supply for Samsung", "is
  Motorola in stock", does NOT trigger EXCEPTION 3 — it stays MATCH:sql. Similarly, a GENERIC launch
  performance question about an NPI device matches the SQL tab's general sales
  signals too, but EXCEPTION 0 still wins — output MISMATCH:npi_launch. Conversely,
  a NAMED sub-metric question like "YoY metric for Galaxy Z Fold8" or "orders to date
  for Flip8" is NOT covered by EXCEPTION 0 — it stays MATCH:sql even though it's
  about an NPI device and even though that metric also appears inside the launch
  performance card).
- Only if no EXCEPTION or "not_for" item applies: if it could loosely fit the
  current tab, say MATCH.
- Output ONLY the single line MATCH or MISMATCH:<key>:<reason>.
 
Worked examples for EXCEPTION 3 (inventory_health) — follow these exactly, especially
the ones with NO word "card" in them, since that wording alone does not decide it:
  "give me inventory health card for iPhone 17"   → MISMATCH:inventory_health
  "give me inventory status of pixel 11 pro"      → MISMATCH:inventory_health
  "what is the inventory status for iPhone 17"    → MISMATCH:inventory_health
  "inventory status for Samsung"                  → MISMATCH:inventory_health
  "inventory summary for Galaxy S26 Ultra"        → MISMATCH:inventory_health
  "inventory for iPhone 17"                       → MATCH   (no exact-phrase match)
  "stock levels for iPhone 17"                    → MATCH   (no exact-phrase match)
  "weeks of supply for Samsung"                   → MATCH   (no exact-phrase match)
  "what is the inventory for SKU 610214688811"    → MATCH   (SKU named)
 
IMPORTANT: The examples above are exact patterns, not a general template. If the user's
question does not closely resemble the wording/structure of one of these examples,
do NOT infer or guess a classification by analogy — treat it as if EXCEPTION 3 does not
apply, and fall through to the general MATCH/MISMATCH rule above. Never invent a
MISMATCH reason, device name, or exception that is not explicitly grounded in the
question text.
 
Response:"""
 
 
_NPI_PERFORMANCE_CUES = ("launch performance", "launch", "performing", "performance")
 
 
async def _npi_launch_hard_match(query: str) -> bool:
    """Deterministic (non-LLM) check: does `query` use launch/performance phrasing AND
    name a device actually tracked in the curated NPI catalog? Used by _router_decide
    to force-switch to npi_launch without depending on the (occasionally inconsistent)
    LLM metadata classifier for this one high-confidence case. Deliberately narrow:
    general "tell me about X"/"overview" phrasing does NOT match here, so it keeps
    going through the normal device_health/sql routing untouched."""
    from API.device_health import extract_device_query
    from API import npi_metrics
 
    q = (query or "").lower()
    if not any(c in q for c in _NPI_PERFORMANCE_CUES):
        return False
    device = extract_device_query(q)
    if not device:
        return False
    try:
        npi_devices = await asyncio.to_thread(npi_metrics.get_npi_devices)
    except Exception as e:
        logger.warning("[router_node] NPI device list lookup failed for %r: %s", device, e)
        return False
    device_lower = device.lower()
    return any(device_lower in d.lower() or d.lower() in device_lower for d in npi_devices)
 
 
async def _llm_metadata_route(
    query: str, active_mode: str, team: str = ""
) -> Dict[str, Any]:
    """Tiny async LLM call — validates whether user query fits the chosen tab."""
    if not active_mode or not _AGENT_METADATA:
        return {"match": True}
 
    prompt = _build_metadata_prompt(query, active_mode, team)
    try:
        response = await async_client.chat.completions.create(
            model=LLM_Model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_completion_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        logger.info(
            "[Metadata] LLM response for mode=%s team=%s: %s", active_mode, team, raw
        )
 
        if raw.upper().startswith("MATCH"):
            return {"match": True}
 
        if raw.upper().startswith("MISMATCH"):
            parts = raw.split(":", 2)
            if len(parts) >= 2:
                suggested_agent = parts[1].strip().lower()
                reason = parts[2].strip() if len(parts) > 2 else ""
                if suggested_agent in _llm_hidden_agents():
                    # the card is never an LLM-routed destination: already on the Data tab ->
                    # nothing to switch; anywhere else -> the Data tab's own (SQL) agent
                    if active_mode == "sql":
                        return {"match": True}
                    suggested_agent = "sql"
                suggested_mode = AGENT_TO_MODE.get(suggested_agent, "sql")
                return {
                    "match": False,
                    "suggested_agent": suggested_agent,
                    "suggested_mode": suggested_mode,
                    "reason": reason,
                }
 
        logger.warning("[Metadata] Unparseable LLM response: %s", raw)
        return {"match": True}
 
    except Exception as e:
        logger.error("[Metadata] LLM route call failed: %s", e)
        return {"match": True}
 
 
def _build_mismatch_suggestion(
    active_mode: str,
    team: str,
    suggested_agent: str,
    suggested_mode: str,
    reason: str,
    query: str,
) -> Dict[str, Any]:
    agents = _AGENT_METADATA.get("agents", {})
    active_agent_key = _current_tab_agent({"mode": active_mode, "team": team})
    active_info    = agents.get(active_agent_key, {})
    suggested_info = agents.get(suggested_agent, {})
    current_tab = active_info.get("tab", active_mode)
    if active_mode == "onboarding" and team == _DATA_BI_TEAM_VALUE:
        current_tab = "Onboarding · Data & BI Solutions"
    # device_health/inventory_health have no tab of their own — they're reached as a flag
    # on the "sql" tab, so the switch-tab message must point there, not at their internal
    # agent_metadata.json label ("Device Overview"/"Inventory Overview").
    if suggested_agent in ("device_health", "inventory_health", "replenishment"):
        target_tab = agents.get("sql", {}).get("tab", _DEFAULT_TAB_LABEL)
    else:
        target_tab = suggested_info.get("tab", suggested_agent)
    message = (
        f"This question doesn't quite fit the **{current_tab}** tab. "
        f"{reason} "
        f"For the best answer, switch to the **{target_tab}** tab and ask again."
    )
    return {
        "recommended_agent": suggested_agent,
        "recommended_mode":  suggested_mode,
        "recommended_tab":   target_tab,
        "current_tab":       current_tab,
        "message":           message,
        "original_query":    query,
    }
 
 
# ---------------------------------------------------------------------------
# F14: History sanitizer — used as the LangGraph messages reducer
# ---------------------------------------------------------------------------
 
_SANITIZE_FAILURE_MARKERS = (
    "error executing snowflake query:",
    "no response generated from sql agent",
    "i couldn't find that specific information",
    "the available documents don't contain",
)
_SANITIZE_SOFT_REFUSAL_MARKERS = (
    "couldn't find", "could not find", "don't contain", "doesn't contain",
    "does not contain", "do not contain", "i don't have", "i do not have",
    "i'm unable to", "i am unable to", "i cannot answer", "i can't answer",
    "not available in the",
)
_SOFT_REFUSAL_MAX_LEN = 400
 
 
def _is_failed_turn(content: str) -> bool:
    """Return True if the AI message content looks like a failure or refusal."""
    text = (content or "").strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    if not text:
        return True
    if any(m in text for m in _SANITIZE_FAILURE_MARKERS):
        return True
    if len(text) <= _SOFT_REFUSAL_MAX_LEN and any(m in text for m in _SANITIZE_SOFT_REFUSAL_MARKERS):
        return True
    return False
 
 
def _sanitize_messages(existing: list, new: list) -> list:
    """
    F14: Custom reducer for state["messages"].
    Drops failed/refusal AI turns (and their preceding human turns) before
    appending the new batch. Replaces the old _sanitize_history() + List[Dict] pattern.
    """
    keep = [True] * len(existing)
    for i, msg in enumerate(existing):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if _is_failed_turn(content):
            keep[i] = False
            for j in range(i - 1, -1, -1):
                if isinstance(existing[j], HumanMessage) and keep[j]:
                    keep[j] = False
                    break
                if isinstance(existing[j], AIMessage):
                    break
    cleaned = [m for m, k in zip(existing, keep) if k]
    return cleaned + list(new)
 
 
 # ---------------------------------------------------------------------------
# OrchestratorState / InsightBotState
# ---------------------------------------------------------------------------
 
 
class InsightBotState(_TypedDict, total=False):
    # F14: canonical history — single source of truth, reducer handles sanitization
    messages: Annotated[List[BaseMessage], _sanitize_messages]
 
    # Routing & session (same fields as before)
    next_agent:  str
    user_roles:  List[str]
    mode:        str
    role:        str
    team:        str
    session_id:  str
    hash_id:     str
    thread_id:   str
    query_id:    str
 
    # F12: request-scoped db type — replaces CURRENT_DB global
    database_type: str
 
    # Agent output written by each node
    agent_output: Dict[str, Any]
    agent_used:   str
 
    # Side-channel fields set by router_node, consumed by target nodes
    _access_denied_msg: str
    _suggestion_msg:    str
    _suggestion_meta:   Dict[str, Any]
    _auto_switch_msg:   str
    _auto_switch_meta:  Dict[str, Any]
 
    # Teams-bot follow-up threading fields (preserved from original)
    query: str
    result: Any
    conversation_history: List[Dict]
    _raw_history: List[Dict]
    _raw_query: str  # the literal text the user/frontend sent THIS turn, before any
                      # follow-up rewrite — agents must use this (not `query`) to detect
                      # a genuine picker-button round-trip; `query` may have been
                      # LLM-rewritten and can contain marker words the user never typed.
    _is_followup: bool
    _followup_target_agent: Optional[str]
    # Explicit/effective button-click flag (real button click, or a picker round-trip
    # query like "iPhone 18 — all variants" / "iPhone 18 Pro (only)") — read by
    # router_node's _skip_metadata check to bypass the LLM mismatch call. MUST be
    # declared here: StateGraph only propagates keys present in this schema, so an
    # undeclared key set in run_orchestrator_query's state_input is silently dropped
    # before router_node (or any other node) ever sees it, and state.get() falls back
    # to its default (False) regardless of what was actually sent.
    _skip_llm_check: bool
    stream_chunk: Dict[str, Any]
    inferred_role: str
 
    # Remembers which device a variant-picker prompt ({"device": ..., "agent": ...})
    # was about, set whenever DeviceHealthAgent/InventoryHealthAgent/NpiLaunchAgent emit
    # a `device_options` payload, cleared once resolved. Lets a typed (not clicked)
    # reply that never repeats the device name ("both series aggregated") still resolve
    # correctly instead of treating the reply text itself as the device — see
    # _recover_device_name. MUST be declared here for the same reason as _skip_llm_check
    # above: StateGraph only persists keys present in this schema across turns.
    _pending_picker_device: Optional[Dict[str, Any]]
 
    # Replenishment card context: {"sku", "store", "awaiting": "store"|"sku"|None}. Set by
    # ReplenishmentAgent when it asked "which store?"/"which SKU?" (or after showing a card),
    # so a short reply ("7920", "what about store 7921") can complete the question.
    # _replenishment_followup is set by router_node on every route to the replenishment node
    # and tells the agent whether to merge the pending half in. Both MUST be declared here
    # for the same reason as _pending_picker_device above.
    _pending_replenishment: Optional[Dict[str, Any]]
    _replenishment_followup: bool
    # {"agent": <agent that answered the previous turn>, "has_ids": <that question named a
    # SKU/store>}. Lets the router tell "7920" after "sales for SKU x" (a SQL follow-up)
    # from a fresh "7920" (the card asks for the SKU).
    _last_data_turn: Optional[Dict[str, Any]]
 
 
# Alias for backwards compatibility — orchestrator() and server.py use this name
OrchestratorState = InsightBotState
 
 
# ---------------------------------------------------------------------------
# F8: Helper + per-agent node functions
# ---------------------------------------------------------------------------
 
def _build_context_from_state(state: InsightBotState) -> "AgentContext":
    """Extract last human message + history dicts from InsightBotState messages list."""
    last_human = next(
        (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
        "",
    )
    history_dicts = [
        {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
        for m in state.get("messages", [])
        if isinstance(m, (HumanMessage, AIMessage)) and isinstance(m.content, str)
    ]
    return AgentContext(
        query=last_human, state=state, conversation_history=history_dicts,
        raw_query=state.get("_raw_query") or last_human,
    )
 
 
async def _run_agent_node(state: InsightBotState, agent_name: str) -> Dict:
    """Generic runner: calls agent.run_stream(), collects output into state."""
    agent   = AGENT_REGISTRY[agent_name]
    context = _build_context_from_state(state)
    output  = {}
    answer_parts = []
 
    async for chunk in agent.run_stream(context):
        if isinstance(chunk, dict) and chunk.get("type") == "final":
            output = chunk.get("result", {}).get("output", {})
        elif isinstance(chunk, dict) and chunk.get("type") == "content":
            answer_parts.append(chunk.get("content", ""))
 
    answer  = output.get("answer", "".join(answer_parts))
    db_type = output.get("database_type", state.get("database_type", "snowflake"))
 
    # Remember the device this agent last resolved (see the state schema comment on
    # _pending_picker_device) so a typed follow-up that doesn't repeat the device name
    # — whether it's answering an open variant picker ("both series aggregated") or a
    # vague later follow-up on the same topic ("what about all variants aggregated?")
    # — can still resolve it instead of treating the reply text itself as the device.
    # Deliberately only OVERWRITE on an actual resolution and never explicitly clear:
    # a turn that resolved nothing (e.g. NpiLaunchAgent's "I couldn't identify a
    # device" message) should leave the last real device in place rather than wiping
    # it, since that's still the best guess for what the user is continuing to discuss.
    _device_options = output.get("device_options")
    _resolved_device = (
        output.get("_resolved_device")
        or (_device_options.get("device") if _device_options else None)
    )
 
    result = {
        "agent_output": output,
        "agent_used":   agent_name,
        "database_type": db_type,
        "messages": [AIMessage(content=answer)] if answer else [],
    }
    if _resolved_device:
        result["_pending_picker_device"] = {"device": _resolved_device, "agent": agent_name}
    if "_pending_replenishment" in output:
        result["_pending_replenishment"] = output["_pending_replenishment"]
    elif agent_name != "replenishment":
        # Any other agent answering ends the card's conversation, so a later "7920" is not
        # mistaken for the answer to a stale "which store?".
        result["_pending_replenishment"] = None
    _ids = _repl.parse_replenishment_query(context.query or "")
    result["_last_data_turn"] = {"agent": agent_name, "has_ids": bool(_ids["sku"] or _ids["store"])}
    return result
 
 
async def sql_node(state: InsightBotState)           -> Dict: return await _run_agent_node(state, "sql")
async def rag_node(state: InsightBotState)           -> Dict: return await _run_agent_node(state, "rag")
async def contract_node(state: InsightBotState)      -> Dict: return await _run_agent_node(state, "contract")
async def onboarding_node(state: InsightBotState)    -> Dict: return await _run_agent_node(state, "onboarding")
async def data_node(state: InsightBotState)          -> Dict: return await _run_agent_node(state, "data")
async def device_health_node(state: InsightBotState) -> Dict: return await _run_agent_node(state, "device_health")
async def npi_launch_node(state: InsightBotState)    -> Dict: return await _run_agent_node(state, "npi_launch")
async def inventory_health_node(state: InsightBotState) -> Dict: return await _run_agent_node(state, "inventory_health")
async def replenishment_node(state: InsightBotState) -> Dict: return await _run_agent_node(state, "replenishment")
 
 
 
 
async def post_agent_check_node(state: InsightBotState) -> Dict:
    """Pass-through node after sql/rag — placeholder for future quality checks."""
    return {}
 
 
async def access_denied_node(state: InsightBotState) -> Dict:
    msg = state.get("_access_denied_msg", _CONTRACT_ONLY_MSG)
    return {
        "agent_output": {"answer": msg, "steps": [], "sources": []},
        "agent_used":   "orchestrator",
        "messages":     [AIMessage(content=msg)],
    }
 
 
# ---------------------------------------------------------------------------
# F8 + F9: ROUTER NODE — uses teams_bot routing (LLM metadata + MODE_TO_AGENT)
# ---------------------------------------------------------------------------
 
async def _router_decide(state: InsightBotState, query: str) -> Dict:
    """
    Determines next_agent using teams_bot routing logic. Called by router_node (below),
    which applies the device-existence gate to whatever this function decides.
    Preserves all teams_bot-specific routing:
      1. LLM metadata mismatch check (_llm_metadata_route) — runs against `query`,
         which by this point is always the resolved/rewritten self-contained version
         for follow-ups (see resolved_query in run_orchestrator_query), so it re-checks
         intent on every turn rather than blindly inheriting the parent turn's agent.
      2. Data+BI Solutions routing
      3. Onboarding routing
      4. Role-based routing (allocation_planner, demand_planner)
      5. MODE_TO_AGENT tab forcing (oem→contract, o9→rag, npi→npi_launch, device_health→device_health)
      6. RBAC checks for contract agent
      7. Default fallback to SQL
 
    On a mismatch (Step 1), the router no longer stops and asks the user to switch tabs —
    it re-runs steps 2-7 with the LLM-suggested mode/team so the graph continues straight
    into the right agent in the same turn. The mismatch message + recommended tab are
    attached to the result as _auto_switch_msg/_auto_switch_meta so run_orchestrator_query
    can surface a brief "switching tabs" status and tell the frontend which tab to select —
    no button, no second round-trip.
    """
    user_roles = state.get("user_roles") or []
    is_contract_only = (
        "InsightBot_Contract_Mngt" in user_roles
        and not any(r in _SQL_O9_ROLES for r in user_roles)
    )
    _active_mode = (state.get("mode") or "").strip().lower()
    _active_team = (state.get("team") or "").strip().lower()
 
    # NPI-scope-fallback redirect: NpiLaunchAgent, on hitting a real device that just
    # isn't tracked in NPI, asks the user to pick Device Overview or Inventory Health
    # instead of silently rerouting (see NpiLaunchAgent.run_stream). The button/typed
    # reply is a self-contained query ("Device overview for iPhone 16" / "Inventory
    # health for iPhone 16") that already carries the device name, so no state lookup
    # is needed here — just recognise the literal prefix and send it straight to the
    # right node instead of letting mode="npi" force it back to npi_launch.
    if _active_mode == "npi":
        _ql = query.strip().lower()
        _npi_scope_agent = None
        if _ql.startswith("device overview for "):
            _npi_scope_agent = "device_health"
        elif _ql.startswith("inventory health for "):
            _npi_scope_agent = "inventory_health"
        if _npi_scope_agent:
            logger.info("[router_node] npi-scope fallback reply → %s", _npi_scope_agent)
            # device_health/inventory_health have no tab of their own — they live on the
            # "sql" tab (see _BUTTON_ONLY_MODES comment below) — so surface the same
            # _auto_switch_msg/_auto_switch_meta the LLM-mismatch path uses elsewhere in
            # this function. Without this, run_orchestrator_query never emits a
            # mode_switch event, and the frontend's tab indicator is left stuck on "NPI"
            # even though the answer it just rendered came from the SQL tab's card.
            _mismatch = _build_mismatch_suggestion(
                active_mode="npi", team=_active_team, suggested_agent=_npi_scope_agent,
                suggested_mode="sql", reason="", query=query,
            )
            return {
                "next_agent": _npi_scope_agent,
                "_auto_switch_msg": f"Switching you to the **{_mismatch['recommended_tab']}** tab now...",
                "_auto_switch_meta": _mismatch,
            }
 
    # ── Replenishment Information card (rule-based — NO LLM is ever consulted) ──────────
    # Decided purely from the words of the message (see _repl.classify_query), with or
    # without the Retail Allocation Planner greeting / role:
    #   "explicit"  "replenishment info for SKU x [in store y]" / "... for SKU x" / "... for store y"
    #   "ids_both"  just a SKU and a store ("610214681249 7920")
    #   "id_only"   just ONE id ("610214681249" / "880C") -> the card asks for the other one
    # Anything naming another metric (sales, inventory, forecast, top N, ...) is NOT matched
    # and carries on to the normal SQL flow below, even with a SKU and/or store in it.
    # Follow-ups:
    #   * the previous turn was this card ("which store?" / a shown card): a short reply
    #     ("7920", "what about store 7921") continues it  -> is_pending_reply
    #   * the previous turn was a SQL question that named a SKU/store ("sales for SKU x"):
    #     a lone id ("7920") completes THAT question, so it stays with SQL  -> _sql_followup
    # Onboarding / OEM / o9 tabs are left alone (see _REPLENISHMENT_ELIGIBLE_MODES).
    if not (_active_mode == "onboarding" or _active_team) and (
        state.get("role") or _active_mode in _REPLENISHMENT_ELIGIBLE_MODES
    ):
        # The user's OWN words decide. The follow-up rewriter turns "sales for the same?" into
        # a full question and may paraphrase it however it likes (even echoing the previous
        # card's SKU/store or the word "replenishment") — that must never turn a metric
        # question into a card. So a card needs the RAW text to qualify, and if the text was
        # rewritten the rewrite has to still qualify too.
        _raw_text = state.get("_raw_query") or query
        _repl_kind = _repl.classify_query(_raw_text)
        if _repl_kind and state.get("_is_followup") and _repl.classify_query(query) is None:
            _repl_kind = None
        _repl_followup = _repl.is_pending_reply(_raw_text, state.get("_pending_replenishment"))
        _last_turn = state.get("_last_data_turn") or {}
        _sql_followup = (
            _repl_kind == "id_only"
            and not _repl_followup
            and _last_turn.get("agent") == "sql"
            and bool(_last_turn.get("has_ids"))
        )
        if (_repl_followup or _repl_kind) and not _sql_followup:
            if is_contract_only:
                return {"next_agent": "access_denied", "_access_denied_msg": _CONTRACT_ONLY_MSG}
            logger.info("[router_node] replenishment ask (kind=%s followup=%s) → replenishment",
                        _repl_kind, _repl_followup)
            _repl_route: Dict[str, Any] = {
                "next_agent": "replenishment",
                "_replenishment_followup": _repl_followup,
            }
            if _active_mode == "npi":
                # Same as the npi-scope fallback above: the card lives on the "sql" tab, so
                # tell the frontend to move the tab indicator off "NPI".
                _mismatch = _build_mismatch_suggestion(
                    active_mode="npi", team=_active_team, suggested_agent="replenishment",
                    suggested_mode="sql", reason="", query=query,
                )
                _repl_route["_auto_switch_msg"] = (
                    f"Switching you to the **{_mismatch['recommended_tab']}** tab now..."
                )
                _repl_route["_auto_switch_meta"] = _mismatch
            return _repl_route
 
    _onboarding_awaiting_team = _active_mode == "onboarding" and not _active_team
    _data_bi_solutions_active = _active_mode == "onboarding" and _active_team == _DATA_BI_TEAM_VALUE
 
    # device_health/inventory_health have no visible tab of their own — they're reached
    # inline on the "sql" tab (button/toggle click, or a router mismatch redirect below).
    # The LLM mismatch check still runs for them (a typed follow-up while viewing one of
    # these cards can drift onto an unrelated question, e.g. "sales for X" while looking
    # at the Inventory Health card), but any redirect out of them must go through the
    # graph's own "sql"/"contract"/etc. node — never call another agent's run_stream
    # directly from inside this one. Nodes carry per-agent behavior keyed off which node
    # is actually running (e.g. SQL's own token-stream suppression), so an in-process
    # call bypasses that bookkeeping and can leak raw intermediate output into the answer.
    _BUTTON_ONLY_MODES = {"device_health", "inventory_health", "replenishment"}
 
    # NOTE: follow-ups are NOT exempted from the mismatch check below. Before the
    # query-rewrite step existed, a terse follow-up like "what about Y" had no content
    # to classify, so skipping the check and trusting the parent's agent was the only
    # option. Now `query` here is always the rewritten, self-contained version (see
    # resolved_query in run_orchestrator_query) — e.g. "device health card for the same
    # device" becomes "Device health card for the Galaxy Flip 8" — so re-running the
    # mismatch check against it correctly re-detects an intent change (e.g. the parent
    # turn was answered by plain SQL, but the follow-up explicitly asks for the Device
    # Health card) instead of blindly inheriting whatever agent answered the parent.
    _skip_metadata = (
        not _active_mode
        or bool(state.get("role"))
        or _onboarding_awaiting_team
        or _data_bi_solutions_active
        or state.get("_skip_llm_check", False)  # explicit button-click flag from frontend
    )
 
    # ── Steps 2-7, factored out so a mismatch can fall through into them with the
    # LLM-suggested mode/team instead of stopping to ask the user to switch tabs. ──
    def _route_tail(active_mode: str, active_team: str) -> Dict:
        # Step 2: Data + BI Solutions
        if active_mode == "onboarding" and active_team == _DATA_BI_TEAM_VALUE:
            logger.info("[router_node] onboarding+data-bi-solutions → data")
            return {"next_agent": "data"}
 
        # Step 3: Onboarding
        if active_mode == "onboarding" or active_team:
            logger.info("[router_node] mode=onboarding or team set → onboarding")
            return {"next_agent": "onboarding"}
 
        # Step 4: Role-based routing
        if state.get("role") == "retail_allocation_planner":
            if is_contract_only:
                return {"next_agent": "access_denied", "_access_denied_msg": _CONTRACT_ONLY_MSG}
            logger.info("[router_node] role=retail_allocation_planner → sql")
            return {"next_agent": "sql"}
        if state.get("role") == "demand_planner":
            if is_contract_only:
                return {"next_agent": "access_denied", "_access_denied_msg": _CONTRACT_ONLY_MSG}
            logger.info("[router_node] role=demand_planner → sql")
            return {"next_agent": "sql"}
 
        # Step 5: MODE_TO_AGENT tab forcing
        # Covers: sql→sql, oem→contract, o9→rag, npi→npi_launch, data→data, device_health→device_health
        forced_key = MODE_TO_AGENT.get(active_mode)
        if forced_key:
            if forced_key == "contract":
                if "TEAMS_BOT" in user_roles:
                    return {"next_agent": "access_denied", "_access_denied_msg":
                            "This feature not available on Teams. Please use the Web version of InsightBot at https://insightbot.t-mobile.com/"}
                if "InsightBot_Contract_Mngt" not in user_roles:
                    return {"next_agent": "access_denied", "_access_denied_msg":
                            "Thank you for your interest in OEM Device Termsheets. You currently don't have access. Please reach out to Elle Wilson for further assistance."}
            if forced_key in AGENT_REGISTRY:
                logger.info("[router_node] mode='%s' → agent='%s'", active_mode, forced_key)
                return {"next_agent": forced_key}
 
        # Step 6: Contract-only RBAC fallback
        if is_contract_only:
            return {"next_agent": "access_denied", "_access_denied_msg": _CONTRACT_ONLY_MSG}
 
        # Step 7: Default — SQL
        logger.info("[router_node] no mode match → default sql")
        return {"next_agent": DEFAULT_AGENT_NAME}
 
    # ── Deterministic NPI auto-switch (bypasses the LLM metadata call for one narrow,
    #    high-confidence case) ──────────────────────────────────────────────────────
    # _llm_metadata_route below is a live LLM judgment call and isn't guaranteed stable
    # run-to-run on a borderline case like this. A genuine launch/performance question
    # naming a device that's actually tracked in NPI is unambiguous enough to decide in
    # code instead: always switch to npi_launch, regardless of which tab the user is on.
    # Deliberately narrow — general "tell me about X"/"overview" phrasing about the same
    # device is NOT included here, so it keeps going through the normal device_health/
    # sql routing exactly as before.
    if not _skip_metadata and _active_mode != "npi":
        if await _npi_launch_hard_match(query):
            logger.info("[router_node] deterministic NPI launch-performance match → npi_launch")
            _tail = _route_tail("npi", "")
            _tail["_auto_switch_msg"] = (
                "This looks like a **launch performance** question — switching you to NPI now..."
            )
            _tail["_auto_switch_meta"] = _build_mismatch_suggestion(
                active_mode=_active_mode, team=_active_team, suggested_agent="npi_launch",
                suggested_mode="npi", reason="", query=query,
            )
            return _tail
 
    # ── Step 1: LLM metadata mismatch check (teams_bot unique feature) ──────
    if not _skip_metadata:
        _route = await _llm_metadata_route(query, _active_mode, _active_team)
        if not _route.get("match"):
            _suggested_agent = _route.get("suggested_agent", "sql")
            # While in the NPI tab, don't let the LLM's semantic "this doesn't fit NPI"
            # judgment silently switch away to device_health/inventory_health on its own
            # (e.g. it can reason "iPhone 16 isn't a new/NPI product" from general
            # knowledge, independent of whether it's actually in the curated NPI device
            # list). That decision belongs to NpiLaunchAgent's own deterministic
            # get_npi_devices() check, which asks the user first instead of silently
            # rerouting — see its "not tracked in NPI" branch and the npi-scope-fallback
            # redirect above. Ignore this particular LLM suggestion and let the query
            # fall through to Step 5 (MODE_TO_AGENT forcing → npi_launch) as normal.
            if _active_mode == "npi" and _suggested_agent in ("device_health", "inventory_health"):
                logger.info(
                    "[router_node] LLM metadata mismatch suggested %s while mode=npi — "
                    "deferring to npi_launch's own scope check instead of auto-switching",
                    _suggested_agent,
                )
                return _route_tail(_active_mode, _active_team)
            # device_health/inventory_health have no visible tab of their own — they live
            # inline on the "Ask questions on our Data" (sql) tab, so there's no tab to
            # switch to. Route straight to the agent and let it answer inline.
            if _active_mode == "sql" and _suggested_agent == "device_health":
                logger.info(
                    "[router_node] LLM metadata mismatch: mode=%s team=%s → "
                    "device_health (direct, no suggestion)",
                    _active_mode, _active_team,
                )
                return {"next_agent": "device_health"}
            if _active_mode == "sql" and _suggested_agent == "inventory_health":
                logger.info(
                    "[router_node] LLM metadata mismatch: mode=%s team=%s → "
                    "inventory_health (direct, no suggestion)",
                    _active_mode, _active_team,
                )
                return {"next_agent": "inventory_health"}
 
            # Already inside device_health/inventory_health (no tab of their own) — if the
            # LLM suggests another tab-less agent (sql, or the sibling card), route straight
            # to that graph node (no tab to switch to) so it gets proper per-node handling
            # (token-stream suppression, status label, agent tracking) instead of being
            # answered in-process by the wrong node. If it suggests an agent that DOES have
            # its own visible tab (npi_launch, contract, rag, onboarding, data), fall through
            # to the normal auto-switch path below so the frontend actually switches tabs.
            if _active_mode in ("device_health", "inventory_health") and _suggested_agent in (
                "sql", "device_health", "inventory_health",
            ):
                _next = _suggested_agent if _suggested_agent in AGENT_REGISTRY else DEFAULT_AGENT_NAME
                logger.info(
                    "[router_node] LLM metadata mismatch: mode=%s team=%s → %s (direct, no suggestion)",
                    _active_mode, _active_team, _next,
                )
                return {"next_agent": _next}
 
            _mismatch = _build_mismatch_suggestion(
                active_mode=_active_mode,
                team=_active_team,
                suggested_agent=_suggested_agent,
                suggested_mode=_route.get("suggested_mode", "sql"),
                reason=_route.get("reason", ""),
                query=query,
            )
            logger.info(
                "[router_node] LLM metadata mismatch: mode=%s team=%s → auto-switching to %s",
                _active_mode, _active_team, _suggested_agent,
            )
            # "data" is only reachable via onboarding + the data-bi-solutions team; every
            # other suggested agent stands alone, so drop any stale team from the old tab.
            _new_team = _DATA_BI_TEAM_VALUE if _suggested_agent == "data" else ""
            _tail = _route_tail(_mismatch["recommended_mode"], _new_team)
            _tail["_auto_switch_msg"] = (
                f"This looks like a **{_mismatch['recommended_tab']}** question — "
                f"switching you there now..."
            )
            _tail["_auto_switch_meta"] = _mismatch
            return _tail
 
    return _route_tail(_active_mode, _active_team)
 
 
# Agents where a device-specific answer is expected — the device-existence gate below only
# applies when the router has decided on one of these. RAG/Contract/Onboarding/Data are never
# touched by it, even if the query happens to mention a device-shaped word — a device mention
# only means something for these four.
_DEVICE_AWARE_AGENTS = {"sql", "device_health", "npi_launch", "inventory_health"}
 
 
async def _device_existence_gate(query: str) -> Optional[Dict]:
    """Checks every device mentioned in `query` against the master catalog BEFORE the target
    agent runs. Returns a router short-circuit dict (next_agent="access_denied" + a message)
    if any mentioned device doesn't exist or hasn't launched yet; otherwise None, letting
    routing proceed exactly as _router_decide() decided.
 
    Centralized here (rather than duplicated inside each agent) so it can't be silently
    skipped for a device-aware agent added later — this is exactly the gap that let SQL
    questions through unchecked before — and so a non-existent device is caught before the
    expensive path (e.g. SQLAgent's full ReAct loop) ever starts, not discovered after it.
 
    Uses extract_all_device_queries() (not the single-match extract_device_query()) so a
    comparison question naming two devices can't hide a non-existent one behind a real one
    mentioned first."""
    from API.device_health import extract_all_device_queries
    from API.device_classifier import classify_device, format_not_found_message, format_upcoming_message
 
    devices = extract_all_device_queries(query)
    if not devices:
        return None
 
    not_found = None
    upcoming = None
    for d in devices:
        result = await asyncio.to_thread(classify_device, d)
        if result["status"] == "NOT_FOUND" and not_found is None:
            not_found = result
        elif result["status"] == "UPCOMING" and upcoming is None:
            upcoming = result
 
    # NOT_FOUND takes priority — a made-up device name is a stronger signal to stop on than
    # a real-but-unlaunched one mentioned alongside it in the same question.
    if not_found:
        return {"next_agent": "access_denied", "_access_denied_msg": format_not_found_message(not_found)}
    if upcoming:
        return {"next_agent": "access_denied", "_access_denied_msg": format_upcoming_message(upcoming)}
    return None
 
 
async def router_node(state: InsightBotState) -> Dict:
    """Graph entry point for the "router" node. Resolves the query text once, delegates the
    actual routing decision to _router_decide(), then applies the device-existence gate to
    whichever agent was decided — regardless of which internal path (mismatch auto-switch,
    MODE_TO_AGENT forcing, or the default fallback) produced it, since this wraps the single
    return point of _router_decide()."""
    query = state.get("query", "")
    if not query and state.get("messages"):
        query = next(
            (m.content for m in reversed(state.get("messages", [])) if isinstance(m, HumanMessage)),
            "",
        )
 
    result = await _router_decide(state, query)
    if result.get("next_agent") == "replenishment":
        result.setdefault("_replenishment_followup", False)
 
    next_agent = result.get("next_agent")
    if next_agent in _DEVICE_AWARE_AGENTS:
        gate = await _device_existence_gate(query)
        if gate:
            logger.info(
                "[router_node] device-existence gate blocked next_agent='%s' for query=%r",
                next_agent, query[:80],
            )
            return gate
    return result
 
 
# ---------------------------------------------------------------------------
# Streaming Orchestrator (kept intact — still used by legacy graph + standalone calls)
# ---------------------------------------------------------------------------
 
 
# ---------------------------------------------------------------------------
async def orchestrator(state):
    """Streaming orchestrator with follow-up context threading."""
    query = state["query"]
    logger.info("[Orchestrator Stream] Incoming query: %s", query)
    conversation_history = state.get("conversation_history", [])
 
    yield {
        "stream_chunk": {
            "type": "status",
            "content": "Analyzing your query...",
            "agent": "orchestrator",
        },
        "agent": "orchestrator",
    }
 
    _user_roles = state.get("user_roles") or []
    _is_contract_only = "InsightBot_Contract_Mngt" in _user_roles and not any(
        r in _SQL_O9_ROLES for r in _user_roles
    )
    logger.debug(
        "[Orchestrator] User roles: %s | contract_only: %s",
        _user_roles,
        _is_contract_only,
    )
    _CONTRACT_ONLY_MSG = (
        "Your account only has access to contract-related questions. "
        "Please contact your administrator if you need access to data queries."
    )
 
    _active_mode = (state.get("mode") or "").strip().lower()
    _active_team = (state.get("team") or "").strip().lower()
 
    _onboarding_awaiting_team = _active_mode == "onboarding" and not _active_team
    _data_bi_solutions_active = (
        _active_mode == "onboarding" and _active_team == _DATA_BI_TEAM_VALUE
    )
    # Follow-up flag set by run_orchestrator_query — skip mismatch routing for follow-up
    # fragments (e.g. "what about galaxy a37?") that look off-tab without context.
    _is_known_followup = state.get("_is_followup", False)
 
    # Modes activated exclusively by frontend button — never LLM-mismatch-check these.
    _BUTTON_ONLY_MODES = {"device_health", "inventory_health", "replenishment"}
 
 
    _skip_metadata = (
        not _active_mode
        or bool(state.get("role"))
        or _onboarding_awaiting_team
        or _data_bi_solutions_active
        or _is_known_followup              # skip mismatch check for confirmed follow-ups
        or _active_mode in _BUTTON_ONLY_MODES  # button-activated modes skip LLM check
        or state.get("_skip_llm_check", False)  # explicit button-click flag from frontend
    )
 
    # ── AgentContext — use effective (threaded) history from state ────────────
    context = AgentContext(
        query=query,
        state=state,
        conversation_history=state.get("conversation_history", conversation_history),
        raw_query=state.get("_raw_query") or query,
    )
 
    # ── inject_mode_tag closure — called after every successful agent answer ──
    def _tag_history_in_result(agent_result: dict) -> dict:
        """
        Append __mode__:<mode> + Q&A to _raw_history and store it in
        agent_result["output"]["conversation_history"] so the API layer
        persists the mode-tagged chain for future follow-up detection.
        """
        output = agent_result.get("output", {})
        answer = output.get("answer", "")
        # Use _raw_history (full chain) as the base — not the short effective_history
        raw_hist = state.get("_raw_history", state.get("conversation_history", []))
        if _active_mode and answer:
            tagged = _inject_mode_tag(raw_hist, _active_mode, query, answer)
            output = dict(output)
            output["conversation_history"] = tagged
            agent_result = dict(agent_result)
            agent_result["output"] = output
        return agent_result
 
    # ── Step 1: LLM metadata mismatch check ──────────────────────────────────
    if not _skip_metadata:
        _route = await _llm_metadata_route(query, _active_mode, _active_team)
        if not _route.get("match"):
            _suggested_agent = _route.get("suggested_agent", "sql")
            _suggested_mode  = _route.get("suggested_mode", "sql")
            _reason          = _route.get("reason", "")
            _mismatch = _build_mismatch_suggestion(
                active_mode=_active_mode,
                team=_active_team,
                suggested_agent=_suggested_agent,
                suggested_mode=_suggested_mode,
                reason=_reason,
                query=query,
            )
            logger.info(
                "[Orchestrator] LLM metadata mismatch: mode=%s team=%s → suggested=%s",
                _active_mode, _active_team, _suggested_agent,
            )
            _msg = _mismatch["message"]
            yield {
                "stream_chunk": {
                    "type": "agent_suggestion",
                    "content": _msg,
                    "agent": "orchestrator",
                    "suggestion": _mismatch,
                },
                "agent": "orchestrator",
            }
            yield {
                "stream_chunk": {
                    "type": "content",
                    "content": _msg,
                    "agent": "orchestrator",
                },
                "agent": "orchestrator",
            }
            yield {
                "result": {
                    "agent": "orchestrator",
                    "output": {
                        "answer": _msg,
                        "sources": [],
                        "steps": [],
                        "suggestion": _mismatch,
                        "conversation_history": conversation_history,
                    },
                }
            }
            return
 
    # ── Step 3: Data + BI Solutions ───────────────────────────────────────────
    # Onboarding tab + team="data-bi-solutions" → DataAgent.
    # This is NOT a standalone tab — it is a sub-selection within Onboarding.
    if _active_mode == "onboarding" and _active_team == _DATA_BI_TEAM_VALUE:
        data_agent = AGENT_REGISTRY.get("data")
        logger.info("[Orchestrator] onboarding+data-bi-solutions → DataAgent")
        yield {
            "stream_chunk": {
                "type": "status",
                "content": "Routing to DATA agent...",
                "agent": "orchestrator",
            },
            "agent": "orchestrator",
        }
        final_result = None
        async for chunk in data_agent.run_stream(context):
            if isinstance(chunk, dict) and chunk.get("type") == "final":
                final_result = chunk
            else:
                yield {"stream_chunk": chunk, "agent": data_agent.name}
        if final_result:
            yield {"result": _tag_history_in_result(final_result.get("result", {}))}
        return
 
    # ── Step 4: Onboarding ────────────────────────────────────────────────────
    # mode="onboarding" (any team except data-bi-solutions, or no team yet) OR
    # team set without a mode (web UI sends team= when chip is clicked).
    # When no team: OnboardingAgent itself asks the user to pick one (team_request).
    if _active_mode == "onboarding" or _active_team:
        onboarding_agent = AGENT_REGISTRY.get("onboarding")
        yield {
            "stream_chunk": {
                "type": "status",
                "content": "Routing to ONBOARDING agent...",
                "agent": "orchestrator",
            },
            "agent": "orchestrator",
        }
        final_result = None
        async for chunk in onboarding_agent.run_stream(context):
            if isinstance(chunk, dict) and chunk.get("type") == "final":
                final_result = chunk
            else:
                yield {"stream_chunk": chunk, "agent": onboarding_agent.name}
        if final_result:
            yield {"result": _tag_history_in_result(final_result.get("result", {}))}
        return
 
    # ── Step 5: Force agent via MODE_TO_AGENT ────────────────────────────────
    # mode is always set (Home.jsx sends mode="sql" for the default tab).
    # Unmapped mode falls back to SQLAgent (safe default).
    _forced_name = MODE_TO_AGENT.get(_active_mode, DEFAULT_AGENT_NAME)
    if _forced_name not in AGENT_REGISTRY:
        _forced_name = DEFAULT_AGENT_NAME
    logger.info("[Orchestrator] mode='%s' → agent='%s'", _active_mode, _forced_name)
    agent_candidates = [AGENT_REGISTRY[_forced_name]]
 
    if not agent_candidates:
        raise AgentExecutionError("No agents available for orchestration.")
 
    # Handle role-based queries with streaming
    if state.get("role") == "retail_allocation_planner":
        if _is_contract_only:
            async for chunk in _yield_access_denied(_CONTRACT_ONLY_MSG, conversation_history):
                yield chunk
            return
        state["inferred_role"] = "allocation"
        conversation_history = list(conversation_history) + [
            {"role": "system", "content": "__role__:allocation"}
        ]
        sql_agent = AGENT_REGISTRY.get("sql")
        yield {
            "stream_chunk": {
                "type": "status",
                "content": "Routing to SQL agent...",
                "agent": "orchestrator",
            },
            "agent": "orchestrator",
        }
        final_result = None
        async for chunk in sql_agent.run_stream(context):
            if isinstance(chunk, dict) and chunk.get("type") == "final":
                final_result = chunk
            else:
                yield {"stream_chunk": chunk, "agent": sql_agent.name}
        if final_result:
            yield {"result": _tag_history_in_result(final_result.get("result", {}))}
        return
 
    # Role-based: demand planner
    if state.get("role") == "demand_planner":
        if _is_contract_only:
            async for chunk in _yield_access_denied(_CONTRACT_ONLY_MSG, conversation_history):
                yield chunk
            return
        state["inferred_role"] = "demand"
        conversation_history = list(conversation_history) + [
            {"role": "system", "content": "__role__:demand"}
        ]
        sql_agent = AGENT_REGISTRY.get("sql")
        yield {
            "stream_chunk": {
                "type": "status",
                "content": "Routing to SQL agent...",
                "agent": "orchestrator",
            },
            "agent": "orchestrator",
        }
        final_result = None
        async for chunk in sql_agent.run_stream(context):
            if isinstance(chunk, dict) and chunk.get("type") == "final":
                final_result = chunk
            else:
                yield {"stream_chunk": chunk, "agent": sql_agent.name}
        if final_result:
            yield {"result": _tag_history_in_result(final_result.get("result", {}))}
        return
 
    # Main agent dispatch
    errors: List[str] = []
    for agent in agent_candidates:
        try:
            if _is_contract_only and agent.name in ("sql", "rag"):
                logger.warning(
                    "[Orchestrator] Access denied: contract-only user tried agent '%s'",
                    agent.name,
                )
                async for chunk in _yield_access_denied(_CONTRACT_ONLY_MSG, conversation_history):
                    yield chunk
                return
 
            if agent.name == "contract":
                user_roles = state.get("user_roles") or []
                if "TEAMS_BOT" in user_roles:
                    _teams_msg = (
                        "This feature not available on Teams. "
                        "Please use the Web version of InsightBot at https://insightbot.t-mobile.com/"
                    )
                    yield {
                        "stream_chunk": {
                            "type": "content",
                            "content": _teams_msg,
                            "agent": "orchestrator",
                        },
                        "agent": "orchestrator",
                    }
                    yield {
                        "result": {
                            "agent": "orchestrator",
                            "output": {
                                "answer": _teams_msg,
                                "steps": [],
                                "conversation_history": conversation_history,
                            },
                        }
                    }
                    return
                if "InsightBot_Contract_Mngt" not in user_roles:
                    logger.warning(
                        "[Orchestrator] Access denied to contract agent. User roles: %s",
                        user_roles,
                    )
                    _no_access = (
                        "Thank you for your interest in OEM Device Termsheets. "
                        "You currently don't have access to review OEM Termsheets. "
                        "Please reach out to Elle Wilson for further assistance."
                    )
                    yield {
                        "stream_chunk": {
                            "type": "content",
                            "content": _no_access,
                            "agent": "orchestrator",
                        },
                        "agent": "orchestrator",
                    }
                    yield {
                        "result": {
                            "agent": "orchestrator",
                            "output": {
                                "answer": _no_access,
                                "steps": [],
                                "conversation_history": conversation_history,
                            },
                        }
                    }
                    return
 
            logger.info("[Orchestrator Stream] Using streaming for agent: %s", agent.name)
            yield {
                "stream_chunk": {
                    "type": "status",
                    "content": f"Routing to {agent.name.upper()} agent...",
                    "agent": "orchestrator",
                },
                "agent": "orchestrator",
            }
            final_result = None
            async for chunk in agent.run_stream(context):
                if isinstance(chunk, dict) and chunk.get("type") == "final":
                    final_result = chunk
                elif isinstance(chunk, dict) and chunk.get("type") == "agent_suggestion":
                    # DeviceHealthAgent made an LLM mismatch call — forward as stream chunk
                    yield {"stream_chunk": chunk, "agent": agent.name}
                else:
                    yield {"stream_chunk": chunk, "agent": agent.name}
 
            if final_result:
                result_data = final_result.get("result", {})
                # If agent returned a mismatch suggestion, yield it as a result too
                if result_data.get("output", {}).get("_is_agent_suggestion"):
                    yield {"result": result_data}
                else:
                    yield {"result": _tag_history_in_result(result_data)}
            return
 
        except Exception as exc:
            logger.exception(
                "Agent '%s' failed to process query '%s'", agent.name, query
            )
            errors.append(f"{agent.name}: {exc}")
            continue
 
    yield {
        "result": {
            "agent": agent_candidates[-1].name,
            "output": {
                "answer": "All agents failed to process the request.",
                "error": "; ".join(errors),
                "steps": [],
                "conversation_history": conversation_history,
            },
        }
    }
 
 
 # ---------------------------------------------------------------------------
# LangGraph Workflow — refactored (F7/F8/F9/F14)
# ---------------------------------------------------------------------------
# Changes vs original single-node graph:
#   F7  — app.astream_events(version="v2") as entry point in run_orchestrator_query
#   F8  — Multi-node StateGraph: router → per-agent nodes via conditional edges
#   F9  — next_agent stored in InsightBotState, not decided inside monolithic orchestrator
#   F14 — messages: Annotated[..., _sanitize_messages] reducer replaces manual sanitization
# ---------------------------------------------------------------------------
 
def _route_fn(state: InsightBotState) -> str:
    return state.get("next_agent", DEFAULT_AGENT_NAME)
 
 
_graph_checkpointer = MemorySaver()
_graph = StateGraph(InsightBotState)
 
_graph.add_node("router",           router_node)
_graph.add_node("sql",              sql_node)
_graph.add_node("rag",              rag_node)
_graph.add_node("contract",         contract_node)
_graph.add_node("onboarding",       onboarding_node)
_graph.add_node("data",             data_node)
_graph.add_node("device_health",    device_health_node)
_graph.add_node("npi_launch",       npi_launch_node)
_graph.add_node("inventory_health", inventory_health_node)
_graph.add_node("replenishment",    replenishment_node)
_graph.add_node("access_denied",    access_denied_node)
_graph.add_node("post_agent_check", post_agent_check_node)
 
_graph.set_entry_point("router")
 
_graph.add_conditional_edges("router", _route_fn, {
    "sql":              "sql",
    "rag":              "rag",
    "contract":         "contract",
    "onboarding":       "onboarding",
    "data":             "data",
    "device_health":    "device_health",
    "npi_launch":       "npi_launch",
    "inventory_health": "inventory_health",
    "replenishment":    "replenishment",
    "access_denied":    "access_denied",
})
 
# sql and rag pass through post_agent_check (no-op, placeholder for future checks)
_graph.add_edge("sql",              "post_agent_check")
_graph.add_edge("rag",              "post_agent_check")
_graph.add_edge("post_agent_check", END)
 
# All other agents end directly
for _n in ("contract", "onboarding", "data", "device_health", "npi_launch", "inventory_health", "replenishment", "access_denied"):
    _graph.add_edge(_n, END)
 
# F7: compile with MemorySaver — enables thread_id based conversation persistence
app = _graph.compile(checkpointer=_graph_checkpointer)
 
# Keep old single-node graph compiled as legacy fallback (orchestrator() still works standalone)
graph = StateGraph(OrchestratorState)
graph.add_node("orchestrator", orchestrator)
graph.set_entry_point("orchestrator")
graph.set_finish_point("orchestrator")
_legacy_app = graph.compile()
 
logger.info("[multi_agent] LangGraph StateGraph compiled: %d nodes", len(_graph.nodes))
 
# ---------------------------------------------------------------------------
# History sanitization helpers
# ---------------------------------------------------------------------------
 
_HISTORY_FAILURE_MARKERS = (
    "error executing snowflake query:",
    "no response generated from sql agent",
    "i couldn't find that specific information",
    "the available documents don't contain",
)
 
_HISTORY_SOFT_REFUSAL_MARKERS = (
    "couldn't find",
    "could not find",
    "don't contain",
    "doesn't contain",
    "does not contain",
    "do not contain",
    "i don't have",
    "i do not have",
    "i'm unable to",
    "i am unable to",
    "i cannot answer",
    "i can't answer",
    "not available in the",
)
 
_HISTORY_SOFT_REFUSAL_MAX_LEN = 400
 
 
def _is_failed_assistant_turn(content: str) -> bool:
    text = (content or "").strip().lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'").replace("\u201d", '"').replace("\u201c", '"')
    if not text:
        return True
    if any(m in text for m in _HISTORY_FAILURE_MARKERS):
        return True
    if len(text) <= _HISTORY_SOFT_REFUSAL_MAX_LEN and any(
        m in text for m in _HISTORY_SOFT_REFUSAL_MARKERS
    ):
        return True
    return False
 
 
def _sanitize_history(conversation_history: List[Dict] | None) -> List[Dict]:
    """
    Drop failed/refusal exchanges.
    Preserves __mode__ system tags so follow-up chain detection still works
    after sanitization — mode boundaries must survive even when Q&A pairs are dropped.
    """
    history = conversation_history or []
    keep = [True] * len(history)
    dropped = 0
    for i, msg in enumerate(history):
        role = msg.get("role") if isinstance(msg, dict) else None
        if role not in ("assistant", "ai"):
            continue
        content = msg.get("content", "") if isinstance(msg, dict) else str(msg)
        if _is_failed_assistant_turn(content):
            keep[i] = False
            dropped += 1
            # Drop the preceding user turn too
            for j in range(i - 1, -1, -1):
                jrole = history[j].get("role") if isinstance(history[j], dict) else None
                jcontent = history[j].get("content", "") if isinstance(history[j], dict) else ""
                # Never drop __mode__ tags — they are boundary markers, not content
                if jrole == "system" and isinstance(jcontent, str) and jcontent.startswith("__mode__:"):
                    continue
                if jrole in ("user", "human"):
                    if keep[j]:
                        keep[j] = False
                        dropped += 1
                    break
                if jrole in ("assistant", "ai"):
                    break
    if dropped:
        logger.info(
            "[history] dropped %d message(s) from failed/refusal turns (%d -> %d)",
            dropped, len(history), sum(keep),
        )
    return [m for m, k in zip(history, keep) if k]
 
 
# ---------------------------------------------------------------------------
# Public API entry point
# ---------------------------------------------------------------------------
 
 
# Markers the device_health / inventory_health variant-picker buttons round-trip on
# (see DeviceHealthAgent/InventoryHealthAgent.run_stream). A query containing one of
# these is a fixed, self-contained button click — never a free-typed question — so it
# must never be sent through the follow-up LLM rewriter or the router's LLM mismatch
# check. Doing so paraphrases the marker away from the end of the string, which breaks
# the agents' end-anchored regex strip and leaves the *entire rewritten question* being
# treated as the device name (wrong card title, zero matching data, and the router
# re-classifying that garbled text on every subsequent click).
_PICKER_ROUNDTRIP_MARKERS = ("all variants", "all variant", "aggregate", "(only)")
 
 
def _is_picker_roundtrip_query(q: str) -> bool:
    ql = (q or "").lower()
    return any(m in ql for m in _PICKER_ROUNDTRIP_MARKERS)
 
 
async def run_orchestrator_query(
    query: str,
    conversation_history: List[Dict] | None = None,
    session_id: str | None = None,
    hash_id: str | None = None,
    thread_id: str | None = None,
    query_id: str | None = None,
    user_roles: List[str] | None = None,
    team: str | None = None,
    mode: str | None = None,
    role: str | None = None,
    skip_llm_check: bool = False,
):
    """
    Streaming orchestrator entry-point.
 
    Follow-up context threading:
    ─────────────────────────────
    Before invoking the orchestrator, we resolve whether this query is a follow-up
    of the immediate same-mode predecessor (-1 parent). The resolution uses:
 
      1. _get_same_mode_parent() — walks history backwards, stops at a __mode__ tag
         that differs from current mode (enforces Scenarios 2 & 3).
      2. _llm_detect_followup()  — tiny LLM call (max 5 tokens) classifies
         FOLLOWUP vs INDEPENDENT (enforces Scenario 4).
 
    Agents receive:
      - FOLLOWUP:    effective_history = [parent_user, parent_assistant] (2 msgs)
      - INDEPENDENT: effective_history = []  (fresh answer, no bleed)
 
    After each answer, _inject_mode_tag() appends __mode__:X + Q&A to _raw_history
    so future queries can detect same-mode chains correctly.
 
    The caller (API endpoint) must persist output["conversation_history"]
    (the tagged version) and pass it back as conversation_history= next time.
    """
    # Strip failed/refusal exchanges
    conversation_history = _sanitize_history(conversation_history or [])
 
    # ── Follow-up context threading ───────────────────────────────────────────
    _is_followup_flag = False
    _effective_history: List[Dict] = []
    # resolved_query is what the router and agents actually see for THIS turn.
    # `query` (the raw text the user typed) is kept untouched below for history
    # persistence/tagging so the chat log matches what was actually asked.
    resolved_query = query
 
    _followup_target_agent: Optional[str] = None
    # A device_health/inventory_health picker button click (e.g. "iPhone 18 — all
    # variants", "iPhone 18 Pro (only)") is already a fixed, self-contained query —
    # treat it exactly like an explicit skip_llm_check button click so it never gets
    # paraphrased by the follow-up rewriter or re-classified by the router's LLM
    # mismatch check. See _is_picker_roundtrip_query() above for why this matters.
    # A replenishment ask ("... SKU x in store y", or "<sku> <store>") is likewise a fixed,
    # self-contained question: never let the follow-up rewriter paraphrase the numbers.
    # A lone id ("7920") is deliberately NOT skipped: right after a SQL question it may be
    # a follow-up that the rewriter should complete ("sales for SKU x" -> "... in store 7920").
    _effective_skip_llm_check = (
        skip_llm_check
        or _is_picker_roundtrip_query(query)
        or _repl.classify_query(query) in ("explicit", "ids_both")
    )
    if mode and not role and not _effective_skip_llm_check:
        _is_followup_flag, _parent_ctx, resolved_query, _followup_target_agent = await _resolve_followup_context(
            current_query=query,
            current_mode=mode,
            conversation_history=conversation_history,
        )
        if _is_followup_flag:
            # Follow-up: agent sees ONLY the -1 parent Q&A (exactly 2 messages)
            _effective_history = _parent_ctx
            logger.info(
                "[run_orchestrator] FOLLOWUP detected for mode='%s' (parent_agent=%s) — "
                "injecting 1 parent turn as history, resolved query='%s...'",
                mode, _followup_target_agent, resolved_query[:80],
            )
        else:
            # Independent question: fresh start, no history bleed across unrelated Qs
            _effective_history = []
            logger.info(
                "[run_orchestrator] INDEPENDENT question for mode='%s' — "
                "starting with empty history",
                mode,
            )
    else:
        # Role-based, no mode, or an explicit/effective button click (skip_llm_check,
        # or a picker round-trip): pass full sanitized history as before. Button clicks
        # carry a fixed, self-contained question (e.g. NPI tab quick-actions, or the
        # device_health/inventory_health variant picker) — skipping follow-up rewriting
        # keeps that question intact instead of letting a same-mode parent turn's
        # metric bleed into it.
        _effective_history = conversation_history
 
    # ── Build state — F7/F8: now uses InsightBotState with messages list ────────
    # Convert _effective_history dicts → LangChain messages for InsightBotState
    history_messages: List[BaseMessage] = []
    for msg in (_effective_history or []):
        if isinstance(msg, dict):
            r = msg.get("role", "user")
            c = msg.get("content", "")
            if r in ("user", "human"):
                history_messages.append(HumanMessage(content=c))
            elif r == "system" and isinstance(c, str) and c.startswith("__mode__:"):
                pass  # skip __mode__ boundary tags — not valid LangChain messages
            else:
                history_messages.append(AIMessage(content=c))
        elif isinstance(msg, BaseMessage):
            history_messages.append(msg)
 
    # Append the resolved (rewritten-if-followup) query as the final HumanMessage —
    # this is what router_node's tab-mismatch check and every agent's context.query
    # actually see, so terse follow-ups ("give me same for iphone") reach routing/
    # generation as a self-contained question instead of a bare fragment.
    history_messages.append(HumanMessage(content=resolved_query))
 
    state_input: InsightBotState = {
        # F14: messages list with reducer — replaces flat query + conversation_history
        "messages": history_messages,
        # Keep teams-bot fields for orchestrator() + follow-up threading
        "query": resolved_query,
        "conversation_history": _effective_history,
        "_raw_history": conversation_history,
        "_raw_query": query,  # literal text before any follow-up rewrite — see InsightBotState
        "_is_followup": _is_followup_flag,
        "_followup_target_agent": _followup_target_agent,
        "_skip_llm_check": _effective_skip_llm_check,   # button clicks (explicit or picker round-trip) — skip LLM mismatch check
        "user_roles": user_roles or [],
        "mode": mode,
        "role": role,
        "team": team,
        "session_id":  session_id  or "api_session",
        "hash_id":     hash_id     or "api_hash",
        "thread_id":   thread_id   or "api_thread",
        "query_id":    query_id    or f"api_query_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
    }
 
    config = {
        "configurable":  {"thread_id": thread_id or "api_thread"},
        "recursion_limit": 25,
    }
 
    agent_used = "unknown"
 
    # SQL agent streams its own word-by-word answer inside run_stream().
    # Streaming on_chat_model_stream for SQL would expose raw ReAct tokens before
    # the final answer is ready (causing UI flicker). All other agents are fine.
    _AGENTS_WITH_OWN_STREAM = {"sql"}
 
    try:
        async for event in app.astream_events(state_input, config=config, version="v2"):
            kind = event.get("event", "")
            name = event.get("name",  "")
            data = event.get("data",  {})
 
            # ── Per-token streaming — suppressed for SQL agent ────────────────
            if kind == "on_chat_model_stream" and agent_used not in _AGENTS_WITH_OWN_STREAM:
                chunk = data.get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    yield {"type": "content", "content": chunk.content, "agent": agent_used}
 
            # ── Tool-end → progress status ────────────────────────────────────
            elif kind == "on_tool_end":
                if name == "execute_snowflake_sql":
                    yield {"type": "status", "content": "Formatting the results...", "agent": agent_used}
                elif name in ("validate_snowflake_sql", "query_database_schema"):
                    yield {"type": "status", "content": "Running the query...", "agent": agent_used}
 
            # ── Node start → status + track current agent ─────────────────────
            # ── Router auto-switched tabs on a mismatch → tell the frontend which tab
            # to select and show a brief "switching you there" status before the target
            # agent's own status/content chunks (below) take over. ─────────────────────
            elif kind == "on_chain_end" and name == "router":
                _router_out = data.get("output", {}) or {}
                _switch_meta = _router_out.get("_auto_switch_meta")
                if _switch_meta:
                    # Auto-switch: this turn is actually being answered under the
                    # recommended mode, not the mode the request started with — keep
                    # `mode` in sync so history tagging (_inject_mode_tag below) and
                    # the final output (read by server.py for the log table) reflect
                    # the tab we actually switched to, not the stale initial one.
                    mode = _switch_meta.get("recommended_mode") or mode
                    yield {"type": "status", "content": _router_out.get("_auto_switch_msg", ""),
                           "agent": "orchestrator"}
                    yield {"type": "mode_switch",
                           "mode":          _switch_meta.get("recommended_mode"),
                           "tab":           _switch_meta.get("recommended_tab"),
                           "target_agent":  _switch_meta.get("recommended_agent"),
                           "agent": "orchestrator"}
 
            elif kind == "on_chain_start" and name in AGENT_REGISTRY:
                agent_used = name
                status_map = {
                    "sql":           "Finding the right data...",
                    "rag":           "Searching O9 documents...",
                    "contract":      "Searching contract documents...",
                    "onboarding":    "Searching onboarding docs...",
                    "data":          "Searching data docs...",
                    "device_health": "Building device overview card...",
                    "npi_launch":    "Pulling launch performance...",
                    "inventory_health": "Pulling inventory health...",
                    "replenishment": "Pulling replenishment information...",
                }
                if name in status_map:
                    yield {"type": "status", "content": status_map[name], "agent": name}
 
            # ── Access denied node end → emit message immediately ─────────────
            elif kind == "on_chain_end" and name == "access_denied":
                output    = data.get("output", {})
                agent_out = output.get("agent_output", {})
                msg = agent_out.get("answer") if agent_out else None
                if not msg:
                    msg = state_input.get("_access_denied_msg", _CONTRACT_ONLY_MSG)
                yield {"type": "content", "content": msg, "agent": "orchestrator"}
 
                # Tag this turn into history exactly like a normal agent answer (see the
                # "Graph end" branch below) — this branch used to skip that entirely and
                # pass conversation_history through untouched, making an access_denied turn
                # (RBAC block, or a device-existence-gate "doesn't exist"/"not launched yet"
                # message) invisible to _get_same_mode_parent(). The next turn's follow-up
                # detection would then silently walk PAST it to an older, unrelated parent
                # turn instead of continuing from what the user just asked about.
                raw_hist = conversation_history or []
                tagged = (
                    _inject_mode_tag(raw_hist, mode, resolved_query, msg, agent_used="orchestrator")
                    if mode else
                    raw_hist + [
                        {"role": "user",      "content": resolved_query},
                        {"role": "assistant", "content": msg},
                    ]
                )
                yield {"type": "final", "result": {
                    "agent": "orchestrator",
                    "output": {
                        "answer": msg, "steps": [],
                        "conversation_history": tagged,
                    },
                }}
                return
 
            # ── Graph end → extract final output ──────────────────────────────
            elif kind == "on_chain_end" and name == "LangGraph":
                output     = data.get("output", {})
                agent_out  = output.get("agent_output", {})
                agent_used = output.get("agent_used", agent_used)
 
                # Normal agent output — inject __mode__ tag for follow-up threading.
                # Persist resolved_query (the rewritten, self-contained question that was
                # actually answered), not the raw query — otherwise the next turn's
                # follow-up parent lookup re-derives context from a vague fragment
                # (e.g. "same for S24") plus a truncated answer, and that loss compounds
                # over multiple follow-up hops.
                if agent_out:
                    raw_hist = conversation_history or []
                    if mode and agent_out.get("answer"):
                        tagged = _inject_mode_tag(raw_hist, mode, resolved_query, agent_out["answer"], agent_used=agent_used)
                    else:
                        tagged = raw_hist + [
                            {"role": "user",      "content": resolved_query},
                            {"role": "assistant", "content": agent_out.get("answer", "")},
                        ]
                    yield {"type": "final", "result": {
                        "agent": agent_used,
                        "output": {
                            **agent_out,
                            "conversation_history": tagged,
                            "final_result": agent_out.get("final_result"),
                        },
                    }}
 
    except Exception as e:
        logger.exception("[run_orchestrator_query] Unhandled error: %s", e)
        yield {"type": "error", "content": str(e)}
 