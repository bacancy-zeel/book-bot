/* Book Bot front end.
 *
 * Every string that came from a user or from a book record is written with
 * textContent, never innerHTML. The one exception is the answer body, which
 * the server already rendered from Markdown with the model's own output
 * escaped first — see `render()` in main.py. */
"use strict";

const thread = document.getElementById("thread");
const welcome = document.getElementById("welcome");
const list = document.getElementById("conversations");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const send = document.getElementById("send");
const clearAll = document.getElementById("clear-all");
const dialog = document.getElementById("confirm");

let currentId = null;
let busy = false;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text() || response.statusText);
  return response.status === 204 ? null : response.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

/* ---------- messages ---------- */
function addUser(text) {
  const row = el("div", "msg-user");
  row.appendChild(el("div", "bubble", text));
  thread.appendChild(row);
  return row;
}

function sourcesNode(sources) {
  const details = el("details", "sources");
  const count = sources.length;
  details.appendChild(el("summary", null,
    `Sources · ${count} book${count === 1 ? "" : "s"}`));

  for (const source of sources) {
    const card = el("div", "book-card");
    const head = el("div");
    head.appendChild(el("span", "cite", `[${source.rank}]`));
    head.appendChild(el("span", "book-title", source.title));
    card.appendChild(head);

    const meta = [source.authors, source.year,
                  source.rating ? `★ ${source.rating}` : ""].filter(Boolean);
    card.appendChild(el("div", "book-meta", meta.join(" · ")));

    const track = el("div", "relevance-track");
    const fill = el("div", "relevance-fill");
    fill.style.width = `${Math.round(source.similarity * 100)}%`;
    track.appendChild(fill);
    card.appendChild(track);
    card.appendChild(el("div", "book-meta",
      `${Math.round(source.similarity * 100)}% match`));
    details.appendChild(card);
  }
  return details;
}

function addBot(html, sources) {
  thread.appendChild(el("div", "msg-label", "Book Bot"));
  const body = el("div", "msg-bot");
  body.innerHTML = html;  // server-rendered Markdown, model output pre-escaped
  thread.appendChild(body);
  if (sources && sources.length) thread.appendChild(sourcesNode(sources));
  return body;
}

function addThinking() {
  // Label and dots share a wrapper so the whole placeholder comes out in one
  // call once the answer lands.
  const holder = el("div");
  holder.appendChild(el("div", "msg-label", "Book Bot"));
  const dots = el("div", "thinking");
  dots.append(el("i"), el("i"), el("i"));
  holder.appendChild(dots);
  thread.appendChild(holder);
  scroll();
  return holder;
}

function scroll() { thread.scrollTop = thread.scrollHeight; }

function clearThread() {
  thread.textContent = "";
}

/* ---------- sidebar ---------- */
async function loadConversations() {
  const conversations = await api("/api/conversations");
  list.textContent = "";
  clearAll.hidden = conversations.length === 0;

  if (!conversations.length) {
    list.appendChild(el("div", "empty",
      "No chats yet. Ask something and it will show up here."));
    return;
  }

  let group = null;
  for (const conversation of conversations) {
    if (conversation.group !== group) {
      group = conversation.group;
      list.appendChild(el("div", "group", group));
    }

    const row = el("div", "row");
    const open = el("button", "open" + (conversation.id === currentId ? " active" : ""));
    open.appendChild(el("span", null, conversation.title));
    open.title = conversation.title;
    open.addEventListener("click", () => openConversation(conversation.id));

    const remove = el("button", "delete", "✕");
    remove.title = "Delete this chat";
    remove.addEventListener("click", async (event) => {
      event.stopPropagation();
      if (!await confirmed("Delete this chat?")) return;
      await api(`/api/conversations/${conversation.id}`, { method: "DELETE" });
      if (conversation.id === currentId) newChat();
      else await loadConversations();
    });

    row.append(open, remove);
    list.appendChild(row);
  }
}

/* ---------- navigation ---------- */
async function openConversation(id) {
  currentId = id;
  clearThread();
  const messages = await api(`/api/conversations/${id}`);
  for (const message of messages) {
    if (message.role === "user") addUser(message.content);
    else addBot(message.content_html, message.sources);
  }
  await loadConversations();
  scroll();
}

function newChat() {
  currentId = null;
  clearThread();
  thread.appendChild(welcome);
  loadConversations();
}

/* ---------- asking ---------- */
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  send.disabled = true;

  if (!currentId) clearThread();  // drops the welcome panel
  addUser(question);
  scroll();
  const dots = addThinking();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: currentId }),
    });
    dots.remove();
    currentId = data.conversation_id;
    addBot(data.reply_html, data.sources);
    await loadConversations();
  } catch (error) {
    dots.remove();
    thread.appendChild(el("div", "msg-bot", `Request failed: ${error.message}`));
  } finally {
    busy = false;
    send.disabled = false;
    scroll();
    input.focus();
  }
}

/* ---------- confirm ---------- */
function confirmed(question) {
  document.getElementById("confirm-text").textContent = question;
  dialog.showModal();
  return new Promise((resolve) => {
    dialog.addEventListener("close", () => resolve(dialog.returnValue === "ok"),
                            { once: true });
  });
}

document.getElementById("confirm-no").addEventListener("click", () => {
  dialog.close("cancel");
});
document.getElementById("confirm-yes").addEventListener("click", () => {
  dialog.close("ok");
});

/* ---------- wiring ---------- */
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value;
  input.value = "";
  ask(question);
});

document.getElementById("new-chat").addEventListener("click", newChat);

clearAll.addEventListener("click", async () => {
  if (!await confirmed("Delete every chat?")) return;
  await api("/api/conversations", { method: "DELETE" });
  newChat();
});

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => ask(button.dataset.question));
}

loadConversations();
input.focus();
