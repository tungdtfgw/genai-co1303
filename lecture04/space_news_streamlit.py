import streamlit as st
import requests
from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import get_key

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Space News",
    page_icon="🚀",
    layout="wide",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SNAPI_BASE = "https://api.spaceflightnewsapi.net/v4"

# ── Function declarations (Gemini tools) ──────────────────────────────────────
list_articles_function = {
    "name": "list_articles",
    "description": (
        "Fetches a list of spaceflight news articles from the Spaceflight News API. "
        "Use this to show recent news, search for a topic, or filter by news site."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Number of articles to return (default 10, max 50)."
            },
            "offset": {
                "type": "integer",
                "description": "Number of articles to skip for pagination (default 0)."
            },
            "search": {
                "type": "string",
                "description": "Keyword to search across article titles and summaries."
            },
            "news_site": {
                "type": "string",
                "description": "Filter by news site name, e.g. 'NASA', 'SpaceNews', 'NASASpaceflight'."
            },
            "ordering": {
                "type": "string",
                "description": (
                    "Field to sort by. Use '-published_at' for newest first (default), "
                    "'published_at' for oldest first."
                )
            },
        },
        "required": []
    }
}

get_article_function = {
    "name": "get_article",
    "description": (
        "Retrieves the full detail of a single spaceflight news article by its numeric ID."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "article_id": {
                "type": "integer",
                "description": "The numeric ID of the article to retrieve."
            }
        },
        "required": ["article_id"]
    }
}

# ── API implementation functions ──────────────────────────────────────────────
def list_articles(limit=10, offset=0, search=None, news_site=None, ordering="-published_at"):
    params = {"limit": limit, "offset": offset, "ordering": ordering}
    if search:
        params["search"] = search
    if news_site:
        params["news_site"] = news_site
    try:
        resp = requests.get(f"{SNAPI_BASE}/articles/", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return {
            "count": data.get("count", 0),
            "articles": data.get("results", [])
        }
    except Exception as e:
        return {"error": str(e)}


def get_article(article_id):
    try:
        resp = requests.get(f"{SNAPI_BASE}/articles/{article_id}/", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e)}


functions = {
    "list_articles": list_articles,
    "get_article": get_article,
}


def call_function(name, **kwargs):
    return functions[name](**kwargs)


# ── Rendering helpers ─────────────────────────────────────────────────────────
def render_article_card(article):
    if article.get("image_url"):
        st.image(article["image_url"], use_container_width=True)
    st.markdown(f"**[{article['title']}]({article['url']})**")
    site = article.get("news_site", "")
    pub = article.get("published_at", "")[:10] if article.get("published_at") else ""
    st.caption(f"{site}  ·  {pub}")
    summary = article.get("summary", "").strip()
    if summary:
        st.write(summary[:300] + ("…" if len(summary) > 300 else ""))


def render_articles_list(articles):
    """Render a list of articles as a 3-column grid of cards."""
    cols_per_row = 3
    for i in range(0, len(articles), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, article in zip(cols, articles[i:i + cols_per_row]):
            with col:
                with st.container(border=True):
                    render_article_card(article)


def render_article_detail(article):
    """Render a single article in full detail."""
    st.subheader(article.get("title", "Article"))
    col1, col2 = st.columns([2, 1])
    with col1:
        if article.get("image_url"):
            st.image(article["image_url"], use_container_width=True)
    with col2:
        st.markdown(f"**Source:** {article.get('news_site', 'N/A')}")
        st.markdown(f"**Published:** {article.get('published_at', '')[:10]}")
        authors = article.get("authors", [])
        if authors:
            names = ", ".join(a.get("name", "") for a in authors)
            st.markdown(f"**Authors:** {names}")
        st.link_button("Read full article →", article.get("url", "#"))
    summary = article.get("summary", "").strip()
    if summary:
        st.markdown("---")
        st.write(summary)


# ── Gemini setup ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=get_key())


def create_chat_session(client):
    config = GenerateContentConfig(
        system_instruction=(
            "You are a helpful space news assistant. "
            "When the user asks about space news, articles, missions, launches, or any spaceflight topics, "
            "use the list_articles function to fetch and show relevant articles. "
            "When the user asks for details about a specific article (by ID or title), "
            "use the get_article function. "
            "Always summarise what you found after fetching data. "
            "Keep your text responses concise. "
            "Respond in the same language the user uses."
        ),
        tools=[{"function_declarations": [list_articles_function, get_article_function]}]
    )
    return client.chats.create(model="gemini-2.5-flash", config=config)


# ── Agentic loop ──────────────────────────────────────────────────────────────
def answer_with_function_call(chat, prompt):
    """
    Send a message, handle any function calls the model requests,
    and return (text_response, tool_data).
    tool_data is a dict with keys 'articles' (list) or 'article' (dict) if a tool was called.
    """
    response = chat.send_message(message=prompt)
    tool_data = None

    # Keep processing until no more function calls
    while True:
        function_call_part = None
        for part in response.candidates[0].content.parts:
            if part.function_call:
                function_call_part = part.function_call
                break

        if not function_call_part:
            break

        fn_name = function_call_part.name
        fn_args = dict(function_call_part.args)

        result = call_function(fn_name, **fn_args)

        # Store structured data for rendering
        if fn_name == "list_articles" and "articles" in result:
            tool_data = {"type": "list", "data": result}
        elif fn_name == "get_article" and "id" in result:
            tool_data = {"type": "detail", "data": result}

        function_response_part = types.Part.from_function_response(
            name=fn_name,
            response={"result": result},
        )
        response = chat.send_message(function_response_part)

    text = ""
    if response and response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                text += part.text

    return text.strip(), tool_data


# ── Session state init ────────────────────────────────────────────────────────
client = get_gemini_client()

if "chat" not in st.session_state:
    st.session_state.chat = create_chat_session(client)

if "messages" not in st.session_state:
    st.session_state.messages = []          # list of {"role", "content", "tool_data"}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Space News")
    st.markdown("Ask the AI assistant anything about space news!")
    st.markdown("---")
    st.markdown("**Try asking:**")
    sample_questions = [
        "Show me the latest space news",
        "Find articles about Mars missions",
        "What's new from NASA?",
        "Show articles about SpaceX launches",
        "Find news about the James Webb telescope",
        "Get article with ID 37234",
    ]
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.pending_input = q

    st.markdown("---")
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat = create_chat_session(client)
        st.rerun()

# ── Main title ────────────────────────────────────────────────────────────────
st.title("🚀 Space News Assistant")
st.caption("Powered by Gemini + Spaceflight News API")

# ── Render chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("content"):
            st.write(msg["content"])
        td = msg.get("tool_data")
        if td:
            if td["type"] == "list":
                articles = td["data"].get("articles", [])
                count = td["data"].get("count", 0)
                if articles:
                    st.caption(f"Showing {len(articles)} of {count} articles")
                    render_articles_list(articles)
            elif td["type"] == "detail":
                render_article_detail(td["data"])

# ── Handle sidebar button ─────────────────────────────────────────────────────
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Fetching space news…"):
            text, tool_data = answer_with_function_call(st.session_state.chat, user_input)
        if text:
            st.write(text)
        if tool_data:
            if tool_data["type"] == "list":
                articles = tool_data["data"].get("articles", [])
                count = tool_data["data"].get("count", 0)
                if articles:
                    st.caption(f"Showing {len(articles)} of {count} articles")
                    render_articles_list(articles)
            elif tool_data["type"] == "detail":
                render_article_detail(tool_data["data"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "tool_data": tool_data,
    })
    st.rerun()

# ── Chat input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask about space news…"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Fetching space news…"):
            text, tool_data = answer_with_function_call(st.session_state.chat, user_input)
        if text:
            st.write(text)
        if tool_data:
            if tool_data["type"] == "list":
                articles = tool_data["data"].get("articles", [])
                count = tool_data["data"].get("count", 0)
                if articles:
                    st.caption(f"Showing {len(articles)} of {count} articles")
                    render_articles_list(articles)
            elif tool_data["type"] == "detail":
                render_article_detail(tool_data["data"])

    st.session_state.messages.append({
        "role": "assistant",
        "content": text,
        "tool_data": tool_data,
    })
    st.rerun()
