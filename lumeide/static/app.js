/* Lume Studio — front-end logic (no frameworks, no build step). */
'use strict';

const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

const KEYWORDS = new Set(['fn', 'let', 'const', 'if', 'else', 'for', 'in', 'while', 'return',
  'break', 'continue', 'import', 'from', 'as', 'true', 'false', 'null', 'and', 'or', 'not']);
const CONTEXTUAL = new Set(['model', 'layer', 'data', 'train', 'eval', 'evaluate', 'predict', 'save', 'load']);
const BUILTINS = new Set(['print', 'println', 'input', 'len', 'type', 'str', 'int', 'float', 'bool',
  'list', 'dict', 'range', 'sum', 'min', 'max', 'abs', 'round', 'floor', 'ceil', 'sqrt', 'exp',
  'log', 'log2', 'sin', 'cos', 'tan', 'pow', 'mean', 'median', 'argmax', 'softmax', 'sigmoid',
  'relu', 'seed', 'shuffle', 'sort', 'unique', 'count', 'keys', 'values', 'upper', 'lower',
  'strip', 'split', 'join', 'replace', 'startswith', 'endswith', 'contains', 'tensor', 'zeros',
  'ones', 'randn', 'eye', 'transpose', 'dataset', 'read_csv', 'write_csv', 'save_model', 'load_model']);

const state = {
  tabs: [],
  activeTab: null,
  chatHistory: [],
  model: 'openai-fast',
  models: [],
  lastError: null,
};

/* ------------------------------------------------------------------ utils */
function escapeHtml(text) {
  return text.replace(/[&<>"']/g, (c) => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
  ));
}

function toast(message, kind = '') {
  const node = $('#toast');
  node.textContent = message;
  node.className = 'toast show ' + kind;
  clearTimeout(node._timer);
  node._timer = setTimeout(() => { node.className = 'toast ' + kind; }, 3200);
}

async function api(path, payload) {
  const options = payload === undefined
    ? {}
    : { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) };
  const response = await fetch(path, options);
  const text = await response.text();
  try { return JSON.parse(text); } catch (err) { return { ok: false, error: text }; }
}

function setStatus(text) { $('#statusLeft').textContent = text; }

/* ------------------------------------------------------- syntax highlight */
function highlight(code) {
  let html = '';
  let i = 0;
  const n = code.length;
  while (i < n) {
    const ch = code[i];
    // comment
    if (ch === '#') {
      let end = code.indexOf('\n', i);
      if (end === -1) end = n;
      html += `<span class="tok-com">${escapeHtml(code.slice(i, end))}</span>`;
      i = end;
      continue;
    }
    // string (also f-strings)
    if (ch === '"' || ch === "'" || (ch === 'f' && (code[i + 1] === '"' || code[i + 1] === "'"))) {
      const prefix = ch === 'f' ? 'f' : '';
      const start = prefix ? i + 1 : i;
      const quote = code[start];
      let j = start + 1;
      while (j < n && code[j] !== quote) {
        if (code[j] === '\\') j += 1;
        j += 1;
      }
      j = Math.min(j + 1, n);
      html += `<span class="tok-str">${escapeHtml(code.slice(i, j))}</span>`;
      i = j;
      continue;
    }
    // number
    if (/[0-9]/.test(ch) || (ch === '.' && /[0-9]/.test(code[i + 1] || ''))) {
      let j = i;
      while (j < n && /[0-9._eE]/.test(code[j])) j += 1;
      html += `<span class="tok-num">${escapeHtml(code.slice(i, j))}</span>`;
      i = j;
      continue;
    }
    // identifier
    if (/[A-Za-z_]/.test(ch)) {
      let j = i;
      while (j < n && /[A-Za-z0-9_]/.test(code[j])) j += 1;
      const word = code.slice(i, j);
      let k = j;
      while (k < n && code[k] === ' ') k += 1;
      const isCall = code[k] === '(';
      let cls = '';
      if (KEYWORDS.has(word)) cls = 'tok-key';
      else if (CONTEXTUAL.has(word)) cls = 'tok-ctx';
      else if (isCall) cls = 'tok-fn';
      else if (BUILTINS.has(word)) cls = 'tok-builtin';
      html += cls ? `<span class="${cls}">${escapeHtml(word)}</span>` : escapeHtml(word);
      i = j;
      continue;
    }
    if (/[+\-*/%=<>!?:]/.test(ch)) {
      html += `<span class="tok-op">${escapeHtml(ch)}</span>`;
      i += 1;
      continue;
    }
    if (/[(){}[\],.@]/.test(ch)) {
      html += `<span class="tok-punc">${escapeHtml(ch)}</span>`;
      i += 1;
      continue;
    }
    html += escapeHtml(ch);
    i += 1;
  }
  return html;
}

/* ------------------------------------------------------------- the editor */
const editor = $('#editor');
const highlightBox = $('#highlight');
const gutter = $('#gutter');

function renderEditor() {
  const code = editor.value;
  highlightBox.querySelector('code').innerHTML = highlight(code) + '\n';
  const lines = code.split('\n').length;
  const errorLine = state.lastError && state.lastError.line ? state.lastError.line : 0;
  let html = '';
  for (let i = 1; i <= lines; i += 1) {
    html += `<span class="ln${i === errorLine ? ' err' : ''}">${i}</span>`;
  }
  gutter.innerHTML = html;
  positionErrorMarker();
}

function positionErrorMarker() {
  const marker = $('#errMarker');
  if (!state.lastError || !state.lastError.line) { marker.style.display = 'none'; return; }
  const lineHeight = 13.5 * 1.62;
  const top = 14 + (state.lastError.line - 1) * lineHeight - editor.scrollTop;
  marker.style.display = 'block';
  marker.style.top = `${top}px`;
  marker.style.left = '54px';
}

function syncScroll() {
  highlightBox.scrollTop = editor.scrollTop;
  highlightBox.scrollLeft = editor.scrollLeft;
  gutter.scrollTop = editor.scrollTop;
  positionErrorMarker();
}

editor.addEventListener('input', renderEditor);
editor.addEventListener('scroll', syncScroll);
editor.addEventListener('keydown', (event) => {
  if (event.key === 'Tab') {
    event.preventDefault();
    const start = editor.selectionStart;
    const end = editor.selectionEnd;
    editor.value = editor.value.slice(0, start) + '    ' + editor.value.slice(end);
    editor.selectionStart = editor.selectionEnd = start + 4;
    renderEditor();
  }
  if (event.key === 'Enter') {
    // keep the indentation of the current line
    const start = editor.selectionStart;
    const lineStart = editor.value.lastIndexOf('\n', start - 1) + 1;
    const line = editor.value.slice(lineStart, start);
    const indent = (line.match(/^\s*/) || [''])[0];
    const extra = /\{\s*$/.test(line) ? '    ' : '';
    if (indent || extra) {
      event.preventDefault();
      const insert = '\n' + indent + extra;
      editor.value = editor.value.slice(0, start) + insert + editor.value.slice(editor.selectionEnd);
      editor.selectionStart = editor.selectionEnd = start + insert.length;
      renderEditor();
    }
  }
});

/* ------------------------------------------------------------------ tabs */
function renderTabs() {
  const container = $('#tabs');
  container.innerHTML = '';
  state.tabs.forEach((tab) => {
    const node = el('div', 'tab' + (tab.name === state.activeTab ? ' active' : ''));
    node.appendChild(el('span', null, tab.name));
    const close = el('span', 'x', '✕');
    close.addEventListener('click', (event) => {
      event.stopPropagation();
      closeTab(tab.name);
    });
    node.appendChild(close);
    node.addEventListener('click', () => openTab(tab.name, false));
    container.appendChild(node);
  });
}

function openTab(name, focus = true) {
  if (!state.tabs.some((tab) => tab.name === name)) {
    state.tabs.push({ name, content: '' });
    api('/api/examples/' + encodeURIComponent(name)).then((data) => {
      const tab = state.tabs.find((t) => t.name === name);
      if (tab && data.content !== undefined) {
        tab.content = data.content;
        if (state.activeTab === name) { editor.value = data.content; renderEditor(); }
      }
    }).catch(() => {});
  }
  state.activeTab = name;
  const tab = state.tabs.find((t) => t.name === name);
  if (tab) { editor.value = tab.content; }
  renderTabs();
  renderEditor();
  if (focus) editor.focus();
}

function closeTab(name) {
  state.tabs = state.tabs.filter((tab) => tab.name !== name);
  if (state.activeTab === name) {
    state.activeTab = state.tabs.length ? state.tabs[0].name : null;
    editor.value = state.activeTab ? state.tabs[0].content : '';
  }
  renderTabs();
  renderEditor();
}

function currentCode() {
  const tab = state.tabs.find((t) => t.name === state.activeTab);
  if (tab) tab.content = editor.value;
  return editor.value;
}

function currentName() {
  return state.activeTab || 'main.lume';
}

/* ------------------------------------------------------------- file tree */
async function loadTree() {
  const data = await api('/api/tree');
  const list = $('#fileTree');
  list.innerHTML = '';
  const files = (data.files || []).filter((f) => f.path.endsWith('.lume'));
  if (!files.length) {
    list.appendChild(el('li', null, 'пока нет файлов .lume'));
  }
  files.forEach((file) => {
    const li = el('li');
    const button = el('button', file.path === state.activeTab ? 'active' : '');
    button.appendChild(el('span', 'ico', '◆'));
    button.appendChild(el('span', null, file.path));
    button.addEventListener('click', () => {
      openLocalFile(file.path);
    });
    li.appendChild(button);
    list.appendChild(li);
  });
  if (data.workspace) {
    $('#workspaceName').textContent = data.workspace.split(/[\\/]/).pop() || 'workspace';
  }
}

async function openLocalFile(path) {
  const data = await api('/api/file?path=' + encodeURIComponent(path));
  if (!data.ok) { toast(data.error || 'не удалось открыть файл', 'err'); return; }
  state.tabs = state.tabs.filter((tab) => tab.name !== path);
  state.tabs.push({ name: path, content: data.content });
  state.activeTab = path;
  editor.value = data.content;
  renderTabs();
  renderEditor();
}

async function loadExamples() {
  const data = await api('/api/examples');
  const list = $('#exampleList');
  list.innerHTML = '';
  (data.files || []).forEach((file) => {
    const li = el('li');
    const button = el('button');
    button.appendChild(el('span', 'ico', '◆'));
    button.appendChild(el('span', null, file.title || file.name));
    button.appendChild(el('span', 'dim', file.name));
    button.addEventListener('click', () => openTab(file.name));
    li.appendChild(button);
    list.appendChild(li);
  });
}

/* -------------------------------------------------------------- running */
async function runCode() {
  const code = currentCode();
  if (!code.trim()) { toast('Нет кода для запуска', 'err'); return; }
  setStatus('запуск…');
  const output = $('#consoleOutput');
  output.innerHTML = '';
  output.appendChild(el('span', 'sys', '▶ запуск…\n'));
  showConsole('output');

  const result = await api('/api/run', { code, filename: currentName() });
  output.innerHTML = '';
  (result.output || []).forEach((line) => {
    output.appendChild(document.createTextNode(line + '\n'));
  });

  state.lastError = result.ok ? null : { line: result.line, col: result.col, message: result.error };
  renderEditor();

  if (result.ok) {
    setStatus(`готово · ${result.ms} мс`);
    $('#runTime').textContent = result.ms + ' мс';
    addSystemLine(output, `✓ выполнено за ${result.ms} мс`);
    renderModels(result.models || []);
    showConsole(result.models && result.models.length ? 'models' : 'output');
  } else {
    setStatus('ошибка');
    addSystemLine(output, '✖ ' + result.error);
    showConsole('output');
    showProblem(result.error, result.line);
  }
}

function addSystemLine(container, text) {
  const span = el('span', 'sys', '\n' + text + '\n');
  container.appendChild(span);
}

async function checkCode() {
  const result = await api('/api/check', { code: currentCode(), filename: currentName() });
  state.lastError = result.ok ? null : { line: result.line, col: result.col, message: result.error };
  renderEditor();
  if (result.ok) {
    toast('Синтаксис в порядке ✓', 'ok');
    setStatus('синтаксис ок');
    $('#problemCount').textContent = '0';
    $('#problemCount').className = 'badge zero';
    $('#problems').innerHTML = '<div class="empty">Нет проблем</div>';
  } else {
    showProblem(result.error, result.line);
    setStatus('ошибка в синтаксисе');
  }
}

function showProblem(message, line) {
  const badge = $('#problemCount');
  badge.textContent = '1';
  badge.className = 'badge';
  const box = $('#problems');
  box.innerHTML = '';
  const item = el('div', 'p-item');
  item.appendChild(el('span', 'loc', line ? `строка ${line}` : 'ошибка'));
  item.appendChild(el('span', 'msg', message));
  item.addEventListener('click', () => {
    if (line) {
      const lines = editor.value.split('\n');
      let pos = 0;
      for (let i = 0; i < line - 1 && i < lines.length; i += 1) pos += lines[i].length + 1;
      editor.focus();
      editor.setSelectionRange(pos, pos + (lines[line - 1] || '').length);
    }
    showConsole('problems');
  });
  box.appendChild(item);
  showConsole('problems');
}

function renderModels(models) {
  const box = $('#modelsPanel');
  box.innerHTML = '';
  if (!models.length) {
    box.appendChild(el('div', 'empty', 'В коде нет моделей — добавьте блок model Name { ... }'));
    return;
  }
  models.forEach((model) => {
    const card = el('div', 'model');
    card.appendChild(el('h4', null, model.name));
    card.appendChild(el('div', 'layers', (model.layers || []).join(' → ')));
    if (model.metrics && model.metrics.accuracy !== undefined) {
      card.appendChild(el('div', 'metrics',
        `accuracy ${Number(model.metrics.accuracy).toFixed(3)} · f1 ${Number(model.metrics.f1 || 0).toFixed(3)}`));
    }
    box.appendChild(card);
  });
}

/* ------------------------------------------------------------- console UI */
function showConsole(which) {
  $('#console').classList.remove('collapsed');
  document.querySelectorAll('.console .ptab').forEach((tab) => {
    tab.classList.toggle('active', tab.dataset.console === which);
  });
  $('#consoleOutput').style.display = which === 'output' ? 'block' : 'none';
  $('#problems').style.display = which === 'problems' ? 'block' : 'none';
  $('#modelsPanel').style.display = which === 'models' ? 'block' : 'none';
}

/* ---------------------------------------------------------- AI assistant */
function renderChat() {
  const chat = $('#chat');
  chat.innerHTML = '';
  state.chatHistory.forEach((message) => {
    const bubble = el('div', 'msg ' + (message.role === 'user' ? 'user' : 'bot'));
    bubble.appendChild(el('div', 'who', message.role === 'user' ? 'вы' : 'Lume Assistant'));
    if (message.image) {
      const img = document.createElement('img');
      img.src = message.image;
      bubble.appendChild(img);
    } else {
      bubble.appendChild(el('div', 'md', renderMarkdown(message.content || '')));
    }
    chat.appendChild(bubble);
  });
  chat.scrollTop = chat.scrollHeight;
}

function renderMarkdown(text) {
  const blocks = [];
  let html = escapeHtml(text);
  html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
    blocks.push(`<pre><code>${code.replace(/\n$/, '')}</code></pre>`);
    return `@@BLOCK${blocks.length - 1}@@`;
  });
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  html = html.replace(/\*\*([^*]+)\*\*/g, '<b>$1</b>');
  html = html.replace(/^\s*[-*]\s+/gm, '• ');
  html = html.replace(/\n{2,}/g, '</div><div class="p">');
  html = html.replace(/\n/g, '<br>');
  html = `<div class="p">${html}</div>`;
  html = html.replace(/@@BLOCK(\d+)@@/g, (match, index) => blocks[Number(index)]);
  return html;
}

async function askAssistant(message) {
  if (!message.trim()) return;
  state.chatHistory.push({ role: 'user', content: message });
  renderChat();
  const thinking = el('div', 'msg bot');
  thinking.appendChild(el('div', 'who', 'Lume Assistant'));
  thinking.appendChild(el('div', 'typing', 'печатает…'));
  $('#chat').appendChild(thinking);
  $('#chat').scrollTop = $('#chat').scrollHeight;

  const result = await api('/api/chat', {
    message,
    code: currentCode(),
    history: state.chatHistory.slice(-6),
    model: state.model,
  });
  thinking.remove();
  if (result.ok) {
    state.chatHistory.push({ role: 'assistant', content: result.reply });
  } else {
    state.chatHistory.push({ role: 'assistant', content: '⚠️ ' + (result.error || 'ассистент недоступен') });
  }
  renderChat();
}

async function generateImage() {
  const prompt = window.prompt('Опишите картинку, которую должен нарисовать ассистент:');
  if (!prompt) return;
  const result = await api('/api/image', { prompt });
  if (!result.ok) { toast(result.error || 'не удалось сгенерировать', 'err'); return; }
  state.chatHistory.push({ role: 'assistant', content: 'Готово!', image: result.url });
  renderChat();
}

/* --------------------------------------------------------------- building */
async function build(target) {
  const code = currentCode();
  if (!code.trim()) { toast('Нет кода для сборки', 'err'); return; }
  const modal = $('#modal');
  const body = $('#modalBody');
  $('#modalTitle').textContent = 'Сборка…';
  body.innerHTML = '<p class="muted">Собираем проект…</p>';
  modal.classList.add('open');

  const result = await api('/api/build/' + target, { code, name: currentName().replace(/\.lume$/, '') });
  if (!result.ok) {
    body.innerHTML = `<p style="color:var(--err)">${escapeHtml(result.error || 'ошибка сборки')}</p>`;
    return;
  }
  const files = (result.files || []).map((f) => `<li>${escapeHtml(f)}</li>`).join('');
  body.innerHTML = `
    <p>✓ Готово: <b>${escapeHtml(result.name || '')}</b></p>
    <p class="path">${escapeHtml(result.path || '')}</p>
    ${result.archive ? `<p class="path">архив: ${escapeHtml(result.archive)}</p>` : ''}
    <ul>${files}</ul>
    <p class="muted">${target === 'app'
      ? 'Запустите app.py — это готовое приложение с веб-интерфейсом.'
      : 'Откройте папку и запустите app.py (в режиме приложения) или model.py (как библиотеку).'}</p>`;
  toast('Сборка готова ✓', 'ok');
  loadTree();
}

/* ------------------------------------------------------------------ boot */
async function loadModels() {
  const data = await api('/api/models');
  state.models = data.models || [];
  const select = $('#modelSelect');
  select.innerHTML = '';
  const seen = new Set();
  state.models.forEach((model) => {
    const id = model.id || model.name;
    if (!id || seen.has(id)) return;
    seen.add(id);
    const option = document.createElement('option');
    option.value = id;
    option.textContent = model.name ? `${id} — ${model.name}` : id;
    if (id === state.model) option.selected = true;
    select.appendChild(option);
  });
  if (!seen.size) {
    const option = document.createElement('option');
    option.value = 'openai-fast';
    option.textContent = 'openai-fast';
    select.appendChild(option);
  }
}

function bindEvents() {
  $('#btnRun').addEventListener('click', runCode);
  $('#btnCheck').addEventListener('click', checkCode);
  $('#btnNewFile').addEventListener('click', async () => {
    const name = window.prompt('Имя нового файла (например main.lume):', 'main.lume');
    if (!name) return;
    await api('/api/save', { path: name, content: editor.value });
    loadTree();
    toast('Файл создан: ' + name, 'ok');
  });
  $('#btnReloadTree').addEventListener('click', loadTree);
  $('#btnClearConsole').addEventListener('click', () => { $('#consoleOutput').innerHTML = ''; });
  $('#btnToggleConsole').addEventListener('click', () => $('#console').classList.toggle('collapsed'));
  $('#btnModalClose').addEventListener('click', () => $('#modal').classList.remove('open'));
  $('#modal').addEventListener('click', (event) => {
    if (event.target.id === 'modal') $('#modal').classList.remove('open');
  });

  $('#btnBuild').addEventListener('click', (event) => {
    event.stopPropagation();
    $('#buildMenu').classList.toggle('open');
  });
  document.addEventListener('click', () => $('#buildMenu').classList.remove('open'));
  document.querySelectorAll('#buildMenu button').forEach((button) => {
    button.addEventListener('click', () => {
      $('#buildMenu').classList.remove('open');
      build(button.dataset.build);
    });
  });

  document.querySelectorAll('.ptab[data-panel]').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.ptab[data-panel]').forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      document.querySelectorAll('.sidebar .panel').forEach((panel) => panel.classList.remove('active'));
      $('#panel' + tab.dataset.panel.charAt(0).toUpperCase() + tab.dataset.panel.slice(1))
        .classList.add('active');
    });
  });
  document.querySelectorAll('.ptab[data-console]').forEach((tab) => {
    tab.addEventListener('click', () => showConsole(tab.dataset.console));
  });

  $('#btnSend').addEventListener('click', () => {
    const input = $('#chatInput');
    const text = input.value;
    input.value = '';
    askAssistant(text);
  });
  $('#chatInput').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault();
      $('#btnSend').click();
    }
  });
  $('#btnImage').addEventListener('click', generateImage);
  document.querySelectorAll('.quick button').forEach((button) => {
    button.addEventListener('click', () => askAssistant(button.dataset.prompt));
  });
  $('#modelSelect').addEventListener('change', (event) => { state.model = event.target.value; });

  $('#btnTheme').addEventListener('click', () => {
    const root = document.documentElement;
    const next = root.dataset.theme === 'dark' ? 'light' : 'dark';
    root.dataset.theme = next;
    localStorage.setItem('lume-theme', next);
  });

  document.addEventListener('keydown', (event) => {
    const ctrl = event.ctrlKey || event.metaKey;
    if (ctrl && event.key === 'Enter') { event.preventDefault(); runCode(); }
    if (ctrl && event.shiftKey && event.key.toLowerCase() === 'c') { event.preventDefault(); checkCode(); }
    if (ctrl && event.key.toLowerCase() === 's') {
      event.preventDefault();
      const tab = state.tabs.find((t) => t.name === state.activeTab);
      if (tab) {
        tab.content = editor.value;
        api('/api/save', { path: tab.name, content: editor.value }).then(() => toast('Сохранено ✓', 'ok'));
      }
    }
  });

  editor.addEventListener('keyup', updateCursor);
  editor.addEventListener('click', updateCursor);
}

function updateCursor() {
  const upto = editor.value.slice(0, editor.selectionStart);
  const lines = upto.split('\n');
  $('#statusPos').textContent = `стр ${lines.length}, кол ${lines[lines.length - 1].length + 1}`;
}

async function boot() {
  const savedTheme = localStorage.getItem('lume-theme');
  if (savedTheme) document.documentElement.dataset.theme = savedTheme;
  bindEvents();
  await loadModels();
  await loadExamples();
  await loadTree();
  const health = await api('/api/health');
  if (health.ok) {
    $('#runtimeStatus').classList.add('on');
    $('#aiStatus').classList.add('on');
  }
  openTab('hello.lume');
  setStatus('готово');
}

boot();
