import { dom, state, EMPTY_STATE_HTML } from './state.js';
import { escapeHtml, fmtMs, fmtNum, hasGSAP, reduceMotion, SOURCES_MARKER, DETAILS_MARKER, ERROR_MARKER, THINKING_MARKER } from './utils.js';
import { setContent, typesetMath } from './markdown.js';
import { initHeroAnimation, stopHeroBeatLoop } from './motion.js';

function splitThinking(text) {
  const parts = String(text).split(THINKING_MARKER);
  if (parts.length <= 1) return { text, thinking: null };
  const thinking = parts.slice(0, -1)
    .map(s => s.replace(/^\n+/, '').replace(/\n+$/, ''))
    .filter(Boolean)
    .join('\n\n');
  return {
    text: parts[parts.length - 1].replace(/^\n+/, ''),
    thinking: thinking || null,
  };
}

export function parseTail(raw) {
  let text = raw;
  let sources = null;
  let details = null;
  let error = null;

  const errIdx = raw.lastIndexOf(ERROR_MARKER);
  if (errIdx !== -1) {
    text = raw.slice(0, errIdx);
    try { error = JSON.parse(raw.slice(errIdx + ERROR_MARKER.length).trim()); } catch (_) {}
    const split = splitThinking(text);
    return {
      text: split.text.replace(/[\s\n]+$/, ''),
      thinking: split.thinking,
      sources: null,
      details: null,
      error: (error && typeof error === 'object') ? error : null,
    };
  }

  const detIdx = text.lastIndexOf(DETAILS_MARKER);
  if (detIdx !== -1) {
    text = text.slice(0, detIdx);
    try { details = JSON.parse(raw.slice(detIdx + DETAILS_MARKER.length).trim()); } catch (_) {}
  }
  const srcIdx = text.lastIndexOf(SOURCES_MARKER);
  if (srcIdx !== -1) {
    const after = text.slice(srcIdx + SOURCES_MARKER.length);
    try { sources = JSON.parse(after.trim()); } catch (_) {}
    text = text.slice(0, srcIdx);
  }

  const split = splitThinking(text);
  return {
    text: split.text.replace(/[\s\n]+$/, ''),
    thinking: split.thinking,
    sources: Array.isArray(sources) ? sources : null,
    details: (details && typeof details === 'object') ? details : null,
    error: null,
  };
}

function sourceMetaText(s) {
  const bits = [];
  if (s.source) bits.push(s.source === 'web' ? 'web' : 'doc');
  if (s.chunk_id) bits.push(s.chunk_id.slice(0, 8));
  if (s.page != null) bits.push(`p.${s.page}`);
  return bits.join(' · ');
}

export function renderSources(el, sources) {
  if (!sources || !sources.length) return;

  const count = document.createElement('div');
  count.className = 'sources-count';
  count.textContent = `${sources.length} source${sources.length === 1 ? '' : 's'}`;
  el.appendChild(count);

  const list = document.createElement('div');
  list.className = 'source-list';

  for (const s of sources) {
    const item = document.createElement('div');
    item.className = 'source-item';

    const head = document.createElement('button');
    head.className = 'source-head';
    head.type = 'button';
    head.setAttribute('aria-expanded', 'false');

    const kind = document.createElement('span');
    kind.className = 'source-kind';
    kind.textContent = s.source === 'web' ? 'web' : 'doc';
    head.appendChild(kind);

    const name = document.createElement('span');
    name.className = 'source-name';
    name.textContent = s.label || 'Source';
    name.title = s.label || '';
    head.appendChild(name);

    const meta = document.createElement('span');
    meta.className = 'source-meta';
    meta.textContent = sourceMetaText(s);
    head.appendChild(meta);

    const chev = document.createElement('span');
    chev.className = 'source-chev';
    chev.textContent = '›';
    head.appendChild(chev);

    const detail = document.createElement('div');
    detail.className = 'source-detail';
    if (s.snippet) {
      const snip = document.createElement('div');
      snip.className = 'snip';
      snip.textContent = s.snippet;
      detail.appendChild(snip);
    }
    const metaLine = document.createElement('div');
    metaLine.className = 'meta';
    if (s.url) {
      const a = document.createElement('a');
      a.href = s.url;
      a.target = '_blank';
      a.rel = 'noopener noreferrer';
      a.textContent = s.url;
      metaLine.appendChild(a);
    }
    if (s.source === 'web' && s.domain) {
      const d = document.createElement('span');
      d.textContent = `domain: ${s.domain}`;
      metaLine.appendChild(d);
    }
    if (s.document_id) {
      const d = document.createElement('span');
      d.textContent = `document id: ${s.document_id}`;
      metaLine.appendChild(d);
    }
    if (s.chunk_id) {
      const d = document.createElement('span');
      d.textContent = `chunk: ${s.chunk_id}`;
      metaLine.appendChild(d);
    }
    if (s.page != null) {
      const d = document.createElement('span');
      d.textContent = `page: ${s.page}`;
      metaLine.appendChild(d);
    }
    if (metaLine.childNodes.length) detail.appendChild(metaLine);

    head.addEventListener('click', () => {
      const open = item.classList.toggle('open');
      head.setAttribute('aria-expanded', String(open));
    });
    item.appendChild(head);
    item.appendChild(detail);
    list.appendChild(item);
  }

  el.appendChild(list);
}

export function renderDetails(el, details) {
  if (!details) return;
  const panel = document.createElement('div');
  panel.className = 'details-panel';

  const kv = (k, v) => `<div class="kv-row"><div class="k">${k}</div><div class="v">${v}</div></div>`;
  const rows = [
    kv('Model', escapeHtml(details.model || '—')),
    kv('Provider', escapeHtml(details.provider || '—')),
    kv('Mode', escapeHtml(details.mode || '—')),
    kv('Source', escapeHtml(details.source || '—')),
    kv('Search type', escapeHtml(details.search_type || '—')),
    kv('Rerank', details.rerank == null ? '—' : String(details.rerank)),
    kv('Retrieval limit', details.retrieval_limit == null ? '—' : details.retrieval_limit),
    kv('Web depth', escapeHtml(details.search_depth || '—')),
    kv('Retrieved docs', details.retrieved_documents == null ? '—' : details.retrieved_documents),
    kv('Reranked docs', details.reranked_documents == null ? '—' : details.reranked_documents),
    kv('Sources', details.source_count == null ? '—' : details.source_count),
  ];
  let html = `<div class="details-grid">${rows.join('')}</div>`;

  const lat = details.latencies || {};
  const latRows = Object.entries(lat).filter(([, v]) => v != null)
    .map(([k, v]) => kv(k, fmtMs(v))).join('');
  if (latRows) html += `<div class="details-sec"><div class="sec-title">Latencies</div>${latRows}</div>`;

  const tok = details.tokens || {};
  const tokRows = Object.entries(tok).filter(([, v]) => v != null)
    .map(([k, v]) => kv(k, fmtNum(v))).join('');
  if (tokRows) html += `<div class="details-sec"><div class="sec-title">Tokens</div>${tokRows}</div>`;

  panel.innerHTML = html;
  el.appendChild(panel);
}

export function createThinkingPanel(row) {
  let panel = row.querySelector('.thinking-panel');
  if (panel) return panel;

  panel = document.createElement('div');
  panel.className = 'thinking-panel open';

  const head = document.createElement('button');
  head.className = 'thinking-head';
  head.type = 'button';
  head.setAttribute('aria-expanded', 'true');
  head.innerHTML = '<span class="thinking-spark" aria-hidden="true"></span><span class="thinking-label">Thinking</span><span class="thinking-chev" aria-hidden="true">›</span>';

  const body = document.createElement('div');
  body.className = 'thinking-body';
  body.textContent = 'Thinking…';

  head.addEventListener('click', () => {
    const open = panel.classList.toggle('open');
    body.hidden = !open;
    head.setAttribute('aria-expanded', String(open));
  });

  panel.appendChild(head);
  panel.appendChild(body);

  const inner = row.querySelector('.row-inner') || row;
  row.insertBefore(panel, inner);
  return panel;
}

export function setThinkingText(panel, text) {
  if (!panel) return;
  const body = panel.querySelector('.thinking-body');
  body.textContent = text && text.trim() ? text : 'Thinking…';
}

export function createMessageRow(role, text) {
  const row = document.createElement('div');
  row.className = 'row ' + role;

  const inner = document.createElement('div');
  inner.className = 'row-inner';

  const avatar = document.createElement('div');
  avatar.className = 'avatar ' + role;
  avatar.textContent = role === 'user' ? 'U' : 'A';

  const content = document.createElement('div');
  content.className = 'bubble-content';
  if (text) setContent(content, text);

  inner.appendChild(avatar);
  inner.appendChild(content);
  row.appendChild(inner);
  return row;
}

export function clearMessages() { dom.messages.innerHTML = ''; }

export function showEmptyState() {
  dom.messages.innerHTML = EMPTY_STATE_HTML;
  initHeroAnimation();
}

export function scrollToBottom() { dom.messages.scrollTop = dom.messages.scrollHeight; }

export function addRow(role, text) {
  const emptyState = dom.messages.querySelector('.empty-state');
  if (emptyState) { emptyState.remove(); stopHeroBeatLoop(); }

  const row = createMessageRow(role, text);
  dom.messages.appendChild(row);

  // let gsap do it instead of the css fadein when it's around
  if (hasGSAP && !reduceMotion) {
    row.style.animation = 'none';
    gsap.from(row, { opacity: 0, y: 22, scale: 0.97, duration: 0.5, ease: 'back.out(1.5)' });
  }

  scrollToBottom();
  return row.querySelector('.bubble-content');
}

export function showTypingIndicator(el) {
  el.innerHTML = '<div class="typing-dots"><span></span><span></span><span></span></div>';
}

// drop the caret inside the last block, otherwise it lands on its own line
function appendCaret(el) {
  let host = el;
  while (host.lastElementChild && /^(UL|OL|BLOCKQUOTE)$/.test(host.lastElementChild.tagName)) {
    host = host.lastElementChild;
  }
  const last = host.lastElementChild;
  if (last && /^(P|LI|H1|H2|H3)$/.test(last.tagName)) host = last;
  const caret = document.createElement('span');
  caret.className = 'caret';
  host.appendChild(caret);
}

// types the response out instead of dumping whole chunks in. rate scales
// with how far behind we are so it can't lag a fast stream.
export function createTypewriter(el) {
  let target = '';
  let shown = 0;
  let ended = false;
  let rafId = null;
  let resolveDone;
  const done = new Promise((r) => { resolveDone = r; });

  const paint = (withCaret) => {
    setContent(el, target.slice(0, shown), { math: false });
    if (withCaret) appendCaret(el);
  };

  const tick = () => {
    if (shown < target.length) {
      const backlog = target.length - shown;
      shown = Math.min(target.length, shown + Math.max(2, Math.ceil(backlog / 8)));
      paint(true);
      scrollToBottom();
    }
    if (ended && shown >= target.length) {
      rafId = null;
      paint(false);
      resolveDone();
      return;
    }
    rafId = requestAnimationFrame(tick);
  };

  const start = () => { if (rafId === null) rafId = requestAnimationFrame(tick); };

  return {
    push(fullText) {
      target = fullText;
      if (reduceMotion) { shown = target.length; paint(false); scrollToBottom(); return; }
      start();
    },
    finish() {
      ended = true;
      if (reduceMotion) { shown = target.length; paint(false); return Promise.resolve(); }
      start();
      // rAF is throttled in background tabs and dead in some webviews.
      // without this the message can stall half-written with the composer
      // stuck disabled.
      const guard = setTimeout(() => {
        shown = target.length;
        paint(false);
        resolveDone();
      }, 4000);
      return done.then(() => clearTimeout(guard));
    },
  };
}

export function addSystemNote(text) {
  const emptyState = dom.messages.querySelector('.empty-state');
  if (emptyState) { emptyState.remove(); stopHeroBeatLoop(); }

  const note = document.createElement('div');
  note.className = 'system-note';
  note.textContent = text;
  dom.messages.appendChild(note);
  scrollToBottom();
}

// message actions (regenerate / details) belong to the LAST assistant message
// only, like chat apps do. rendering them for a message clears any that are
// showing elsewhere so they can never pile up per-message.
export function renderMessageActions(row, { details, stopped }) {
  if (!row) return;
  dom.messages.querySelectorAll('.msg-actions').forEach(n => n.remove());
  dom.messages.querySelectorAll('.details-panel').forEach(n => n.remove());

  const actions = document.createElement('div');
  actions.className = 'msg-actions';

  const regen = document.createElement('button');
  regen.className = 'msg-action';
  regen.type = 'button';
  regen.innerHTML = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 21v-5h5"/></svg> Regenerate';
  regen.disabled = state.isStreaming;
  regen.addEventListener('click', () => {
    import('./chat.js').then(({ regenerateLast }) => regenerateLast());
  });
  actions.appendChild(regen);

  if (details) {
    const det = document.createElement('button');
    det.className = 'msg-action';
    det.type = 'button';
    det.textContent = 'Details';
    det.addEventListener('click', () => {
      const existing = row.querySelector('.details-panel');
      if (existing) { existing.remove(); det.textContent = 'Details'; }
      else { renderDetails(row.querySelector('.bubble-content'), details); det.textContent = 'Hide details'; }
    });
    actions.appendChild(det);
  }

  if (stopped) {
    const note = document.createElement('span');
    note.style.cssText = 'font-size:12px;color:var(--text-faint)';
    note.textContent = 'Generation stopped — the partial answer was not saved.';
    actions.appendChild(note);
  }

  row.appendChild(actions);
}

export function attachMessageExtras(contentEl, extra, { last = false } = {}) {
  if (!contentEl) return;
  const row = contentEl.closest('.row');
  const sources = extra && extra.sources;
  const details = extra && extra.details;
  const thinking = extra && extra.thinking;
  if (thinking) {
    const panel = createThinkingPanel(row);
    setThinkingText(panel, thinking);
  }
  if (sources) renderSources(contentEl, sources);
  if (last) renderMessageActions(row, { details, stopped: false });
}