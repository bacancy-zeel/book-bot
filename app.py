"""Book Bot — a RAG chatbot over a book catalogue.

Pipeline: CSV -> one record per book -> local ONNX embeddings -> ChromaDB
cosine search -> Gemini writes a grounded answer citing the books it used.

Only generation touches a hosted API. Embedding runs locally, so indexing the
whole catalogue costs nothing and is not rate limited.

The UI is a plain chat: a sidebar of past conversations on the left, the thread
in the middle. History is persisted by `history.py`, not by session state, so
it survives a refresh.
"""
import html
from dataclasses import asdict, is_dataclass
from datetime import datetime

import streamlit as st

import history
from rag import store
from rag.generate import GenerationError, answer
from rag.retrieve import search

st.set_page_config(
    page_title="Book Bot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

history.init()

EXAMPLES = [
    "Recommend a dystopian novel about surveillance",
    "What are the highest rated fantasy books?",
    "A book about grief that isn't depressing",
    "Who wrote The Hunger Games?",
]

# Streamlit's own chrome (deploy button, hamburger menu, status widget, the
# coloured decoration bar) is removed here; `.streamlit/config.toml` stops most
# of it being served in the first place.
st.markdown("""
<style>
  :root {
    --bg: #faf9f5;
    --sidebar: #f0eee6;
    --ink: #2f2f2c;
    --muted: #82807a;
    --line: #e6e2d8;
    --accent: #c96442;
    --bubble: #ebe8df;
  }

  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"],
  [data-testid="stAppDeployButton"],
  #MainMenu,
  footer { display: none !important; }
  [data-testid="stHeader"] { background: transparent; }

  .stApp { background: var(--bg); }
  .block-container { max-width: 47rem; padding-top: 1.6rem; padding-bottom: 6rem; }
  [data-testid="stBottom"] > div { background: var(--bg); }

  /* ---------- sidebar ---------- */
  section[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--line);
    width: 278px !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0.25rem; }
  .brand {
    font-size: 15px; font-weight: 650; color: var(--ink);
    padding: 2px 4px 12px; letter-spacing: -0.01em;
  }
  .hist-group {
    font-size: 11px; font-weight: 650; color: var(--muted);
    text-transform: uppercase; letter-spacing: .06em;
    padding: 16px 10px 4px;
  }
  .hist-empty { font-size: 13px; color: var(--muted); padding: 10px; line-height: 1.5; }
  .confirm { font-size: 12.5px; color: var(--ink); padding: 8px 10px 2px; }

  /* Every level between the column and the label is a flex item, and a flex
     item defaults to min-width:auto — it refuses to shrink below its own text.
     So the chain widened to fit the whole title, which is what pushed the
     label out of its column and under the ✕. min-width:0 lets it shrink, and
     only then does text-overflow:ellipsis on the <p> have anything to clip. */
  section[data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
    flex-wrap: nowrap; align-items: center; gap: 4px;
  }
  section[data-testid="stSidebar"] [data-testid="stColumn"] {
    min-width: 0; overflow: hidden;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] {
    min-width: 0; width: 100%;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] button {
    background: transparent; border: none; color: var(--ink);
    display: flex; width: 100%; max-width: 100%; min-width: 0;
    overflow: hidden; text-align: left; justify-content: flex-start;
    font-size: 13.5px; font-weight: 450; padding: 7px 10px;
    border-radius: 8px; min-height: 0;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] button > div,
  section[data-testid="stSidebar"] [data-testid="stButton"] button [data-testid="stMarkdownContainer"] {
    min-width: 0; max-width: 100%; overflow: hidden;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] button p {
    margin: 0; max-width: 100%; display: block;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {
    background: rgba(0, 0, 0, .06); color: var(--ink);
  }
  /* the open conversation */
  section[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
    background: rgba(201, 100, 66, .12); color: #8d3d20; font-weight: 600;
  }
  /* Widgets are targeted by their st.* key, which Streamlit emits as a
     `st-key-<key>` class on the wrapper. Positional selectors (nth-of-type, a
     marker div plus `+`) break whenever the DOM is reshuffled. These use a
     descendant `button` rather than `.stButton > button` so they hold whether
     the class lands on the element container or on .stButton itself. */
  .st-key-new-chat button {
    background: #fff; border: 1px solid var(--line); font-weight: 550;
    box-shadow: 0 1px 2px rgba(0, 0, 0, .04); margin-bottom: 4px;
  }
  .st-key-new-chat button:hover {
    border-color: var(--accent); color: var(--accent); background: #fff;
  }
  /* the ✕ that deletes a chat, in each row's narrow second column */
  [class*="st-key-del-"] button {
    opacity: .5; padding: 7px 0; justify-content: center; text-align: center;
    font-size: 14px; line-height: 1; color: var(--muted);
  }
  [class*="st-key-del-"] button p {
    text-overflow: clip; text-align: center; width: 100%;
  }
  [data-testid="stHorizontalBlock"]:hover [class*="st-key-del-"] button {
    opacity: 1;
  }
  [class*="st-key-del-"] button:hover {
    opacity: 1; background: rgba(201, 100, 66, .15); color: var(--accent);
  }
  /* confirm row: destructive action reads as destructive */
  [class*="st-key-yes-"] button, .st-key-clear-yes button {
    background: var(--accent); color: #fff; justify-content: center;
    font-weight: 600; font-size: 12.5px;
  }
  [class*="st-key-yes-"] button p, .st-key-clear-yes button p,
  [class*="st-key-no-"] button p, .st-key-clear-no button p {
    text-align: center; width: 100%;
  }
  [class*="st-key-yes-"] button:hover, .st-key-clear-yes button:hover {
    background: #b0522f; color: #fff;
  }
  [class*="st-key-no-"] button, .st-key-clear-no button {
    border: 1px solid var(--line); background: #fff;
    justify-content: center; font-size: 12.5px;
  }
  .st-key-clear-all button {
    color: var(--muted); font-size: 12.5px; border-top: 1px solid var(--line);
    border-radius: 0; padding-top: 12px;
  }
  .st-key-clear-all button:hover {
    color: var(--accent); background: transparent;
  }

  /* ---------- messages ---------- */
  .msg-user { display: flex; justify-content: flex-end; margin: 22px 0 6px; }
  .msg-user .bubble {
    background: var(--bubble); color: var(--ink); max-width: 82%;
    padding: 10px 15px; border-radius: 16px 16px 4px 16px;
    font-size: 15px; line-height: 1.55;
  }
  .msg-label {
    font-size: 11.5px; font-weight: 650; color: var(--accent);
    letter-spacing: .04em; text-transform: uppercase; margin: 18px 0 -6px;
  }
  .block-container [data-testid="stMarkdownContainer"] p {
    font-size: 15.2px; line-height: 1.72;
  }

  /* ---------- welcome ---------- */
  .welcome { text-align: center; padding: 13vh 0 18px; }
  .welcome h1 {
    font-family: Georgia, "Times New Roman", serif; font-weight: 400;
    font-size: 32px; color: var(--ink); margin: 0 0 8px; letter-spacing: -.01em;
  }
  .welcome p { color: var(--muted); font-size: 14px; margin: 0; }
  [class*="st-key-ex-"] button {
    background: #fff; border: 1px solid var(--line); border-radius: 12px;
    color: var(--ink); font-size: 13.5px; text-align: left;
    justify-content: flex-start; padding: 13px 15px; height: 100%;
    white-space: normal; line-height: 1.45;
  }
  [class*="st-key-ex-"] button p { white-space: normal; }
  [class*="st-key-ex-"] button:hover {
    border-color: var(--accent); color: var(--accent); background: #fff;
  }

  /* ---------- sources ---------- */
  [data-testid="stExpander"] {
    border: 1px solid var(--line); border-radius: 12px;
    background: #fff; box-shadow: none;
  }
  [data-testid="stExpander"] summary { font-size: 13px; color: var(--muted); }
  .book-card {
    border: 1px solid var(--line); border-radius: 10px; padding: 11px 13px;
    background: #fdfcfa; margin-bottom: 8px;
  }
  .book-title { font-weight: 600; font-size: 13.5px; color: var(--ink); }
  .book-meta { font-size: 11.5px; color: var(--muted); margin-top: 3px; }
  .relevance-track {
    height: 4px; background: #eae7df; border-radius: 999px; margin-top: 8px;
  }
  .relevance-fill { height: 100%; background: var(--accent); border-radius: 999px; }
  .cite { font-size: 11px; color: var(--accent); font-weight: 700; }

  /* ---------- composer ---------- */
  [data-testid="stChatInput"] {
    border: 1px solid var(--line); border-radius: 16px; background: #fff;
    box-shadow: 0 2px 12px rgba(0, 0, 0, .06);
  }
  [data-testid="stChatInput"]:focus-within { border-color: var(--accent); }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
# Widest title the sidebar column fits at 13.5px before CSS has to clip it.
SIDEBAR_TITLE_CHARS = 26


def elide(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip(" ,.;:") + "…"


def as_dict(hit) -> dict:
    """Hits arrive as dataclasses live, and as plain dicts out of history."""
    return asdict(hit) if is_dataclass(hit) else hit


def render_sources(hits) -> None:
    hits = [as_dict(h) for h in hits]
    label = f"Sources · {len(hits)} book" + ("s" if len(hits) != 1 else "")
    with st.expander(label, expanded=False):
        for hit in hits:
            rating = hit.get("rating")
            meta = " · ".join(p for p in [
                hit.get("authors"), hit.get("year"), rating and f"★ {rating}",
            ] if p)
            similarity = float(hit.get("similarity") or 0.0)
            st.markdown(
                f"""<div class="book-card">
                      <span class="cite">[{hit.get('rank')}]</span>
                      <span class="book-title">{html.escape(str(hit.get('title', '')))}</span>
                      <div class="book-meta">{html.escape(meta)}</div>
                      <div class="relevance-track">
                        <div class="relevance-fill" style="width:{similarity * 100:.0f}%"></div>
                      </div>
                      <div class="book-meta">{similarity:.0%} match</div>
                    </div>""",
                unsafe_allow_html=True,
            )


def render_user(text: str) -> None:
    body = html.escape(text).replace("\n", "<br>")
    st.markdown(
        f'<div class="msg-user"><div class="bubble">{body}</div></div>',
        unsafe_allow_html=True,
    )


def render_assistant(text: str, hits) -> None:
    st.markdown('<div class="msg-label">Book Bot</div>', unsafe_allow_html=True)
    st.markdown(text)
    if hits:
        render_sources(hits)


def render_welcome() -> None:
    st.markdown(
        '<div class="welcome"><h1>What are you looking to read?</h1>'
        "<p>Ask about the catalogue — every answer cites the books it used.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    if not store.count():
        st.warning(
            "The catalogue is empty. Run `python -m rag.ingest` to index it."
        )
        return

    cols = st.columns(2)
    for index, example in enumerate(EXAMPLES):
        if cols[index % 2].button(example, key=f"ex-{index}",
                                  use_container_width=True):
            st.session_state.pending = example
            st.rerun()


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------
def group_of(timestamp: float) -> str:
    day = datetime.fromtimestamp(timestamp).date()
    days = (datetime.now().date() - day).days
    if days <= 0:
        return "Today"
    if days == 1:
        return "Yesterday"
    if days < 7:
        return "Previous 7 days"
    if days < 30:
        return "Previous 30 days"
    return "Older"


def open_chat(conversation_id: str | None) -> None:
    st.session_state.conversation_id = conversation_id
    st.session_state.pop("confirm_delete", None)
    st.session_state.pop("pending", None)
    st.rerun()


def render_row(conversation: dict, active: bool) -> None:
    conversation_id = conversation["id"]

    if st.session_state.get("confirm_delete") == conversation_id:
        st.markdown('<div class="confirm">Delete this chat?</div>',
                    unsafe_allow_html=True)
        yes, no = st.columns(2, gap="small")
        with yes:
            if st.button("Delete", key=f"yes-{conversation_id}",
                         use_container_width=True):
                history.delete(conversation_id)
                open_chat(None if active
                          else st.session_state.get("conversation_id"))
        with no:
            if st.button("Cancel", key=f"no-{conversation_id}",
                         use_container_width=True):
                st.session_state.pop("confirm_delete", None)
                st.rerun()
        return

    # The ✕ needs a column wide enough to actually be a click target.
    title, remove = st.columns([0.8, 0.2], gap="small")
    with title:
        # Elided in Python as well as in CSS: the stored title is up to 44
        # characters, which cannot fit a 278px sidebar at any font size, and a
        # label that overflows its button is what lands it under the ✕.
        if st.button(elide(conversation["title"], SIDEBAR_TITLE_CHARS),
                     key=f"open-{conversation_id}",
                     type="primary" if active else "secondary",
                     help=conversation["title"],
                     use_container_width=True):
            open_chat(conversation_id)
    with remove:
        if st.button("✕", key=f"del-{conversation_id}",
                     help="Delete this chat", use_container_width=True):
            st.session_state.confirm_delete = conversation_id
            st.rerun()


def render_sidebar(active_id: str | None) -> None:
    with st.sidebar:
        st.markdown('<div class="brand">📚 Book Bot</div>',
                    unsafe_allow_html=True)

        if st.button("＋  New chat", key="new-chat", use_container_width=True):
            open_chat(None)

        saved = history.conversations()
        if not saved:
            st.markdown(
                '<div class="hist-empty">No chats yet. Ask something and it '
                "will show up here.</div>",
                unsafe_allow_html=True,
            )
            return

        current_group = None
        for conversation in saved:
            group = group_of(conversation["updated_at"])
            if group != current_group:
                current_group = group
                st.markdown(f'<div class="hist-group">{group}</div>',
                            unsafe_allow_html=True)
            render_row(conversation, conversation["id"] == active_id)

        st.markdown('<div class="hist-group">&nbsp;</div>',
                    unsafe_allow_html=True)
        if st.session_state.get("confirm_clear"):
            st.markdown('<div class="confirm">Delete every chat?</div>',
                        unsafe_allow_html=True)
            yes, no = st.columns(2, gap="small")
            with yes:
                if st.button("Delete all", key="clear-yes",
                             use_container_width=True):
                    history.delete_all()
                    st.session_state.pop("confirm_clear", None)
                    open_chat(None)
            with no:
                if st.button("Cancel", key="clear-no",
                             use_container_width=True):
                    st.session_state.pop("confirm_clear", None)
                    st.rerun()
        elif st.button("Clear all history", key="clear-all",
                       use_container_width=True):
            st.session_state.confirm_clear = True
            st.rerun()


# --------------------------------------------------------------------------
def main() -> None:
    conversation_id = st.session_state.get("conversation_id")
    if conversation_id and not history.exists(conversation_id):
        conversation_id = st.session_state.conversation_id = None  # deleted

    chat = history.ConversationHistory(conversation_id)
    thread = chat.messages

    if thread:
        for message in thread:
            if message.type == "human":
                render_user(message.content)
            else:
                render_assistant(message.content,
                                 message.additional_kwargs.get("hits"))
    else:
        render_welcome()

    question = st.chat_input("Ask about books…") or st.session_state.pop(
        "pending", None)

    if question:
        if not conversation_id:
            conversation_id = history.create(history.title_from(question))
            st.session_state.conversation_id = conversation_id
            chat = history.ConversationHistory(conversation_id)

        chat.add_user_message(question)
        render_user(question)

        st.markdown('<div class="msg-label">Book Bot</div>',
                    unsafe_allow_html=True)
        with st.spinner("Searching the catalogue…"):
            hits = search(question)

        try:
            with st.spinner("Writing an answer…"):
                reply = answer(question, hits, thread)
        except GenerationError as error:
            reply = (
                f"**Could not generate an answer.** {error}\n\n"
                "Retrieval still worked — the matching books are listed below."
            )

        st.markdown(reply)
        if hits:
            render_sources(hits)

        chat.add_messages([
            history.answer_message(reply, [as_dict(h) for h in hits])
        ])

    # Rendered last so a chat started on this run is already in the list —
    # otherwise the new title would only appear on the following rerun.
    render_sidebar(conversation_id)


if __name__ == "__main__":
    main()
