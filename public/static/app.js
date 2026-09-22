/* Book Bot front end.
 *
 * The thread lives here and nowhere else: `turns` is sent with each question
 * so a follow-up can be understood, and it is gone when the tab closes.
 *
 * Every string that came from a user or from a book record is written with
 * textContent, never innerHTML. The one exception is the answer body, which
 * the server already rendered from Markdown with the model's own output
 * escaped first — see `render()` in main.py. */
"use strict";

const thread = document.getElementById("thread");
const welcome = document.getElementById("welcome");
const form = document.getElementById("composer");
const input = document.getElementById("question");
const send = document.getElementById("send");
const newChat = document.getElementById("new-chat");

let turns = [];
let busy = false;

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text() || response.statusText);
  return response.json();
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function scroll() { thread.scrollTop = thread.scrollHeight; }

/* ---------- messages ---------- */
function addUser(text) {
  const row = el("div", "msg-user");
  row.appendChild(el("div", "bubble", text));
  thread.appendChild(row);
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

/* ---------- asking ---------- */
async function ask(question) {
  if (busy || !question.trim()) return;
  busy = true;
  send.disabled = true;

  if (!turns.length) thread.textContent = "";  // drops the welcome panel
  newChat.hidden = false;
  addUser(question);
  scroll();
  const pending = addThinking();

  try {
    const data = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: turns }),
    });
    pending.remove();
    addBot(data.reply_html, data.sources);
    turns.push({ role: "user", content: question },
               { role: "assistant", content: data.reply });
  } catch (error) {
    pending.remove();
    thread.appendChild(el("div", "msg-bot", `Request failed: ${error.message}`));
  } finally {
    busy = false;
    send.disabled = false;
    scroll();
    input.focus();
  }
}

function reset() {
  turns = [];
  thread.textContent = "";
  thread.appendChild(welcome);
  newChat.hidden = true;
  input.focus();
}

/* ---------- wiring ---------- */
form.addEventListener("submit", (event) => {
  event.preventDefault();
  const question = input.value;
  input.value = "";
  ask(question);
});

newChat.addEventListener("click", reset);

for (const button of document.querySelectorAll(".example")) {
  button.addEventListener("click", () => ask(button.dataset.question));
}

input.focus();
