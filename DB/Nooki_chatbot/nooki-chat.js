/*
 * Nooki 홈페이지 챗봇 위젯
 * 사용: <script src="DB/Nooki_chatbot/nooki-chat.js" defer></script>
 * 질문 → Supabase Edge Function(nooki-chat) → RAG 검색 + Gemini 답변
 * Shadow DOM 안에 그려서 홈페이지(Tailwind) 스타일과 서로 간섭하지 않음
 */
(() => {
  const CONFIG = {
    endpoint: "https://somiyetvungpiupzktav.supabase.co/functions/v1/nooki-chat",
    // 공개용 anon key (브라우저 노출 전제). 데이터 접근은 Edge Function 안에서만 이뤄짐
    anonKey:
      "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNvbWl5ZXR2dW5ncGl1cHprdGF2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3OTE4NDAxMzIsImV4cCI6MjEwNzQxNjEzMn0.mNST10jqL_tRf5ty6odHllIYpcf__7iDVcj6UXiG0AU",
    maxQuestion: 500,
    historyTurns: 6,
    storageKey: "nooki-chat-history",
    suggestions: [
      "Nooki는 어떤 로봇인가요?",
      "어떤 물건을 찾을 수 있나요?",
      "어두운 소파 밑에서도 찾을 수 있나요?",
      "어떤 하드웨어로 만들어졌나요?",
    ],
  };

  const WELCOME =
    "안녕하세요! **Nooki 도우미**예요.\nNooki의 사용 방법, 하드웨어·소프트웨어 구성, 성능 목표, 지원 범위 등을 물어보세요.";

  const ICONS = {
    chat: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 4h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-4 4V6a2 2 0 0 1 2-2z" fill="currentColor"/></svg>',
    close: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/></svg>',
    send: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 20l18-8L3 4v6l12 2-12 2z" fill="currentColor"/></svg>',
    reset: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v4h4" stroke="currentColor" stroke-width="2" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  };

  const STYLE = `
    :host { all: initial; }
    * { box-sizing: border-box; font-family: Inter, "Pretendard", "Malgun Gothic", system-ui, sans-serif; }
    svg { width: 20px; height: 20px; display: block; }
    button { font: inherit; cursor: pointer; border: none; background: none; color: inherit; }

    .launcher {
      position: fixed; right: 24px; bottom: 24px; z-index: 2147483000;
      width: 60px; height: 60px; border-radius: 50%;
      background: #00375e; color: #fff;
      display: grid; place-items: center;
      box-shadow: 0 8px 24px rgba(0, 55, 94, .35);
      transition: transform .2s, background .2s;
    }
    .launcher:hover { background: #1f4e79; transform: translateY(-2px); }
    .launcher svg { width: 26px; height: 26px; }
    .launcher::after {
      content: ""; position: absolute; top: 6px; right: 6px; width: 12px; height: 12px;
      border-radius: 50%; background: #feae2c; border: 2px solid #00375e;
    }
    .launcher.open::after { display: none; }

    .panel {
      position: fixed; right: 24px; bottom: 96px; z-index: 2147483000;
      width: 380px; height: min(600px, calc(100vh - 128px));
      background: #f7f9ff; color: #141c23;
      border: 1px solid #dbe3ed; border-radius: 16px; overflow: hidden;
      box-shadow: 0 16px 48px rgba(20, 28, 35, .18);
      display: flex; flex-direction: column;
      opacity: 0; transform: translateY(12px) scale(.98); pointer-events: none;
      transition: opacity .18s, transform .18s;
    }
    .panel.open { opacity: 1; transform: none; pointer-events: auto; }

    header {
      background: #00375e; color: #fff; padding: 14px 12px 14px 18px;
      display: flex; align-items: center; gap: 10px;
    }
    .avatar {
      width: 34px; height: 34px; border-radius: 50%; background: #feae2c; color: #291800;
      display: grid; place-items: center; font-weight: 700; font-size: 15px; flex: none;
    }
    .titles { flex: 1; min-width: 0; }
    .titles b { display: block; font-size: 15px; }
    .titles span { font-size: 12px; color: #a0cafc; }
    header button { width: 32px; height: 32px; border-radius: 8px; display: grid; place-items: center; color: #d1e4ff; }
    header button:hover { background: rgba(255,255,255,.12); color: #fff; }
    header button svg { width: 18px; height: 18px; }

    .log { flex: 1; overflow-y: auto; padding: 16px; display: flex; flex-direction: column; gap: 12px; }
    .msg { max-width: 85%; font-size: 14px; line-height: 1.6; word-break: keep-all; overflow-wrap: anywhere; }
    .msg.user {
      align-self: flex-end; background: #1f4e79; color: #fff;
      padding: 9px 13px; border-radius: 14px 14px 4px 14px; white-space: pre-wrap;
    }
    .msg.bot {
      align-self: flex-start; background: #fff; border: 1px solid #e1e9f3;
      padding: 10px 14px; border-radius: 14px 14px 14px 4px;
    }
    .msg.bot p { margin: 0; }
    .msg.bot p + p, .msg.bot p + ul, .msg.bot ul + p { margin-top: 6px; }
    .msg.bot ul { margin: 0; padding-left: 18px; }
    .msg.bot li + li { margin-top: 2px; }
    .msg.bot strong { color: #00375e; }
    .msg.error { border-color: #ffdad6; background: #fff8f7; color: #93000a; }
    .sources { margin-top: 8px; padding-top: 6px; border-top: 1px dashed #dbe3ed; font-size: 11.5px; color: #72777f; }

    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chips button {
      font-size: 12.5px; padding: 6px 11px; border-radius: 999px;
      background: #fff; border: 1px solid #c2c7d0; color: #42474f;
    }
    .chips button:hover { border-color: #feae2c; background: #fff8ee; color: #633f00; }

    .typing { display: inline-flex; gap: 4px; padding: 4px 0; }
    .typing i { width: 6px; height: 6px; border-radius: 50%; background: #72777f; animation: blink 1.2s infinite; }
    .typing i:nth-child(2) { animation-delay: .2s; }
    .typing i:nth-child(3) { animation-delay: .4s; }
    @keyframes blink { 0%, 80%, 100% { opacity: .25; } 40% { opacity: 1; } }

    form { display: flex; gap: 8px; align-items: flex-end; padding: 12px; border-top: 1px solid #e1e9f3; background: #fff; }
    textarea {
      flex: 1; resize: none; max-height: 96px; min-height: 40px;
      padding: 9px 12px; border: 1px solid #c2c7d0; border-radius: 12px;
      font: inherit; font-size: 14px; line-height: 1.45; color: #141c23; background: #fff;
    }
    textarea:focus { outline: 2px solid #1f4e79; outline-offset: -1px; border-color: transparent; }
    .send { width: 40px; height: 40px; border-radius: 12px; background: #feae2c; color: #291800; display: grid; place-items: center; flex: none; }
    .send:hover { background: #ffb955; }
    .send:disabled { background: #dbe3ed; color: #72777f; cursor: default; }
    .note { font-size: 11px; color: #72777f; text-align: center; padding: 0 12px 10px; background: #fff; }

    button:focus-visible, textarea:focus-visible { outline: 2px solid #feae2c; outline-offset: 2px; }

    @media (max-width: 480px) {
      .panel { right: 0; bottom: 0; width: 100vw; height: 100dvh; border-radius: 0; border: none; }
      .launcher { right: 16px; bottom: 16px; }
      .launcher.open { display: none; }
    }
    @media (prefers-reduced-motion: reduce) {
      .panel, .launcher { transition: none; }
      .typing i { animation: none; opacity: .6; }
    }
  `;

  // ---------- 상태 ----------
  let history = loadHistory(); // [{ role: "user" | "model", text, sources? }]
  let busy = false;

  function loadHistory() {
    try { return JSON.parse(sessionStorage.getItem(CONFIG.storageKey)) || []; } catch { return []; }
  }
  function saveHistory() {
    try { sessionStorage.setItem(CONFIG.storageKey, JSON.stringify(history.slice(-30))); } catch { /* 저장 불가 환경 무시 */ }
  }

  // ---------- DOM ----------
  const host = document.createElement("div");
  host.id = "nooki-chat";
  const root = host.attachShadow({ mode: "open" });
  root.innerHTML = `
    <style>${STYLE}</style>
    <button class="launcher" aria-label="Nooki 도우미 열기" aria-expanded="false">${ICONS.chat}</button>
    <section class="panel" role="dialog" aria-label="Nooki 도우미" aria-hidden="true">
      <header>
        <div class="avatar">N</div>
        <div class="titles"><b>Nooki 도우미</b><span>Nooki PRD 기반 AI 안내</span></div>
        <button class="reset" title="대화 새로 시작" aria-label="대화 새로 시작">${ICONS.reset}</button>
        <button class="close" title="닫기" aria-label="닫기">${ICONS.close}</button>
      </header>
      <div class="log" aria-live="polite"></div>
      <form>
        <textarea rows="1" maxlength="${CONFIG.maxQuestion}" placeholder="Nooki에 대해 물어보세요" aria-label="질문 입력"></textarea>
        <button class="send" type="submit" aria-label="보내기" disabled>${ICONS.send}</button>
      </form>
      <div class="note">AI 답변은 부정확할 수 있어요.</div>
    </section>
  `;

  const $ = (sel) => root.querySelector(sel);
  const launcher = $(".launcher");
  const panel = $(".panel");
  const log = $(".log");
  const form = $("form");
  const input = $("textarea");
  const sendBtn = $(".send");

  // ---------- 렌더링 ----------
  const escapeHtml = (s) =>
    s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  // 답변용 최소 마크다운: **굵게**, "- " 목록, 줄바꿈
  function renderMarkdown(text) {
    const inline = (s) => escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    const blocks = [];
    let list = null;
    for (const raw of text.split("\n")) {
      const line = raw.trim();
      const item = line.match(/^[-*•]\s+(.*)/);
      if (item) {
        (list ??= []).push(`<li>${inline(item[1])}</li>`);
        continue;
      }
      if (list) { blocks.push(`<ul>${list.join("")}</ul>`); list = null; }
      if (line) blocks.push(`<p>${inline(line)}</p>`);
    }
    if (list) blocks.push(`<ul>${list.join("")}</ul>`);
    return blocks.join("");
  }

  function addMessage(role, text, { sources = [], error = false } = {}) {
    const el = document.createElement("div");
    el.className = `msg ${role === "user" ? "user" : "bot"}${error ? " error" : ""}`;
    if (role === "user") {
      el.textContent = text;
    } else {
      el.innerHTML = renderMarkdown(text);
      if (sources.length) {
        const src = document.createElement("div");
        src.className = "sources";
        src.textContent = "참고: " + sources.slice(0, 2).join(" · ");
        el.appendChild(src);
      }
    }
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    return el;
  }

  function addSuggestions() {
    const wrap = document.createElement("div");
    wrap.className = "chips";
    for (const q of CONFIG.suggestions) {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = q;
      b.addEventListener("click", () => ask(q));
      wrap.appendChild(b);
    }
    log.appendChild(wrap);
  }

  function renderAll() {
    log.innerHTML = "";
    addMessage("bot", WELCOME);
    if (!history.length) addSuggestions();
    for (const t of history) addMessage(t.role, t.text, { sources: t.sources });
  }

  // ---------- 질문 ----------
  async function ask(question) {
    question = question.trim();
    if (!question || busy) return;
    busy = true;
    input.value = "";
    autosize();
    updateSendState();

    log.querySelector(".chips")?.remove();
    addMessage("user", question);

    const typing = document.createElement("div");
    typing.className = "msg bot";
    typing.innerHTML = '<span class="typing" aria-label="답변 작성 중"><i></i><i></i><i></i></span>';
    log.appendChild(typing);
    log.scrollTop = log.scrollHeight;

    const payload = {
      question,
      history: history.slice(-CONFIG.historyTurns).map(({ role, text }) => ({ role, text })),
    };

    try {
      const res = await fetch(CONFIG.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${CONFIG.anonKey}`,
          apikey: CONFIG.anonKey,
        },
        body: JSON.stringify(payload),
      });
      const data = await res.json().catch(() => ({}));
      typing.remove();

      if (!res.ok || data.error || !data.answer) {
        addMessage("bot", data.error || "답변을 가져오지 못했어요. 잠시 후 다시 시도해 주세요.", { error: true });
      } else {
        history.push({ role: "user", text: question });
        history.push({ role: "model", text: data.answer, sources: data.sources || [] });
        saveHistory();
        addMessage("bot", data.answer, { sources: data.sources || [] });
      }
    } catch {
      typing.remove();
      addMessage("bot", "네트워크 문제로 답변을 가져오지 못했어요. 연결을 확인해 주세요.", { error: true });
    } finally {
      busy = false;
      updateSendState();
      input.focus();
    }
  }

  // ---------- 입력 & 열기/닫기 ----------
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 96) + "px";
  }
  function updateSendState() {
    sendBtn.disabled = busy || !input.value.trim();
  }

  function setOpen(open) {
    panel.classList.toggle("open", open);
    launcher.classList.toggle("open", open);
    panel.setAttribute("aria-hidden", String(!open));
    launcher.setAttribute("aria-expanded", String(open));
    launcher.setAttribute("aria-label", open ? "Nooki 도우미 닫기" : "Nooki 도우미 열기");
    launcher.innerHTML = open ? ICONS.close : ICONS.chat;
    if (open) setTimeout(() => input.focus(), 50);
  }

  launcher.addEventListener("click", () => setOpen(!panel.classList.contains("open")));
  $(".close").addEventListener("click", () => setOpen(false));
  $(".reset").addEventListener("click", () => {
    history = [];
    saveHistory();
    renderAll();
    input.focus();
  });
  root.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { setOpen(false); launcher.focus(); }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    ask(input.value);
  });
  input.addEventListener("input", () => { autosize(); updateSendState(); });
  input.addEventListener("keydown", (e) => {
    // Enter 전송, Shift+Enter 줄바꿈, 한글 조합 중 Enter 는 무시
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      ask(input.value);
    }
  });

  renderAll();
  document.body.appendChild(host);
})();
