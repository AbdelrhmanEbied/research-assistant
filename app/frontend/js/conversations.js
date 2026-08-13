import { dom, state } from './state.js';
import { api, ICON_X, showToast } from './utils.js';
import { createMessageRow, addRow, attachMessageExtras, clearMessages, showEmptyState } from './render.js';

function updateHeader() {
  if (state.currentConversationId === null) {
    dom.mainHeader.textContent = 'New chat';
    dom.headerActions.hidden = true;
    return;
  }
  const conv = state.conversationsCache.find(c => c.id === state.currentConversationId);
  dom.mainHeader.textContent = (conv && conv.title) || 'New chat';
  dom.headerActions.hidden = false;
}

// TODO: this refetches the whole list after every send, fine for now but
// it'll need paginating once there are a few hundred chats
export async function loadConversations() {
  const q = convSearchEl.value.trim();
  const res = await api('/chat/list' + (q ? '?q=' + encodeURIComponent(q) : ''));
  state.conversationsCache = await res.json();

  dom.convList.innerHTML = '';
  if (state.conversationsCache.length === 0) {
    dom.convList.innerHTML = '<div class="conv-empty">No chats found</div>';
  }
  for (const conv of state.conversationsCache) {
    const item = document.createElement('div');
    item.className = 'conv-item' + (conv.id === state.currentConversationId ? ' active' : '');
    // it's a div, so it needs this to be reachable by keyboard at all
    item.tabIndex = 0;
    item.setAttribute('role', 'button');

    const title = document.createElement('div');
    title.className = 'title';
    title.textContent = conv.title || 'New chat';
    title.title = 'Double-click to rename';
    title.addEventListener('dblclick', (e) => {
      e.stopPropagation();
      renameConversationInline(conv.id, title);
    });
    item.appendChild(title);

    const delBtn = document.createElement('button');
    delBtn.className = 'del-btn';
    delBtn.innerHTML = ICON_X;
    delBtn.title = 'Delete conversation';
    delBtn.setAttribute('aria-label', `Delete "${conv.title || 'New chat'}"`);
    delBtn.onclick = (e) => { e.stopPropagation(); deleteConversation(conv.id); };
    item.appendChild(delBtn);

    item.onclick = () => selectConversation(conv.id);
    item.onkeydown = (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectConversation(conv.id); }
    };
    dom.convList.appendChild(item);
  }
  updateHeader();
}

async function fetchMessagesPage(id) {
  const res = await api(`/chat/${id}/messages?limit=${state.messagesPage.limit}&offset=${state.messagesPage.loaded}`);
  const page = await res.json();
  state.messagesPage.total = page.total;
  state.messagesPage.loaded += page.messages.length;
  return page.messages;
}

function addLoadEarlierButton() {
  if (dom.messages.querySelector('.load-earlier')) return;
  const btn = document.createElement('button');
  btn.className = 'load-earlier';
  btn.textContent = 'Load earlier messages';
  btn.addEventListener('click', loadEarlierMessages);
  dom.messages.insertBefore(btn, dom.messages.firstChild);
}

async function loadEarlierMessages() {
  const btn = dom.messages.querySelector('.load-earlier');
  if (btn) btn.remove();
  const scrollGap = dom.messages.scrollHeight - dom.messages.scrollTop;
  const older = await fetchMessagesPage(state.currentConversationId);
  // page is newest-first; walk backwards so the oldest lands on top
  for (let i = older.length - 1; i >= 0; i--) {
    const row = createMessageRow(older[i].role, older[i].content);
    dom.messages.insertBefore(row, dom.messages.firstChild);
    if (older[i].role === 'assistant') attachMessageExtras(row.querySelector('.bubble-content'), older[i].extra, { last: false });
  }
  dom.messages.scrollTop = dom.messages.scrollHeight - scrollGap;
  if (state.messagesPage.loaded < state.messagesPage.total) addLoadEarlierButton();
}

export async function selectConversation(id) {
  if (window.innerWidth <= 768) dom.sidebar.classList.remove('open');
  if (id === state.currentConversationId) return;
  state.currentConversationId = id;
  clearMessages();
  await loadConversations();

  state.messagesPage = { total: 0, loaded: 0, limit: 200 };
  const messages = await fetchMessagesPage(id);

  if (messages.length === 0) showEmptyState();
  else {
    // newest-first page; walk backwards so the oldest renders first
    for (let i = messages.length - 1; i >= 0; i--) {
      const contentEl = addRow(messages[i].role, messages[i].content);
      if (messages[i].role === 'assistant') {
        attachMessageExtras(contentEl, messages[i].extra, { last: i === 0 });
      }
    }
    if (state.messagesPage.loaded < state.messagesPage.total) addLoadEarlierButton();
  }
}

export async function newConversation() {
  const res = await api('/chat/conversations', { method: 'POST' });
  const conv = await res.json();
  state.currentConversationId = conv.id;
  clearMessages();
  showEmptyState();
  await loadConversations();
  dom.input.focus();
  return conv.id;
}

export async function deleteConversation(id) {
  await api(`/chat/${id}`, { method: 'DELETE' });
  if (id === state.currentConversationId) {
    state.currentConversationId = null;
    clearMessages();
    showEmptyState();
  }
  await loadConversations();
}

export async function ensureConversation() {
  if (state.currentConversationId === null) await newConversation();
  return state.currentConversationId;
}

async function renameConversationInline(id, titleEl) {
  const original = titleEl.textContent;
  const input = document.createElement('input');
  input.className = 'conv-title-edit';
  input.style.width = '100%';
  input.value = original;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (title && title !== original) {
      try {
        await api(`/chat/${id}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
      } catch (_) {}
    }
    await loadConversations();
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.blur();
    else if (e.key === 'Escape') { input.value = original; input.blur(); }
  });
  input.addEventListener('blur', commit);
}

async function exportConversation(format) {
  if (!state.currentConversationId) return;
  try {
    const res = await api(`/chat/${state.currentConversationId}/export?format=${format}`);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `conversation-${state.currentConversationId}.${format === 'markdown' ? 'md' : 'json'}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  } catch (_) {
    showToast('Export failed');
  }
}

/* ---- wiring ---- */

const convSearchEl = document.getElementById('convSearch');
let convSearchTimer = null;
convSearchEl.addEventListener('input', () => {
  clearTimeout(convSearchTimer);
  convSearchTimer = setTimeout(loadConversations, 250);
});

const renameBtn = document.getElementById('renameBtn');
const exportMdBtn = document.getElementById('exportMdBtn');
const exportJsonBtn = document.getElementById('exportJsonBtn');

renameBtn.addEventListener('click', () => {
  if (!state.currentConversationId) return;
  const span = document.getElementById('mainHeader');
  const original = span.textContent;
  const input = document.createElement('input');
  input.className = 'conv-title-edit';
  input.value = original;
  span.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const commit = async () => {
    if (done) return;
    done = true;
    const title = input.value.trim();
    if (title && title !== original) {
      try {
        await api(`/chat/${state.currentConversationId}`, {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title }),
        });
      } catch (_) {}
    }
    await loadConversations();
  };

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') input.blur();
    else if (e.key === 'Escape') { input.value = original; input.blur(); }
  });
  input.addEventListener('blur', commit);
});

exportMdBtn.addEventListener('click', () => exportConversation('markdown'));
exportJsonBtn.addEventListener('click', () => exportConversation('json'));

dom.newChatBtn.addEventListener('click', newConversation);