import { dom, state } from './state.js';
import { api, showToast, escapeHtml, ICON_FILE, ICON_SUMMARIZE, ICON_X } from './utils.js';
import { addSystemNote } from './render.js';
import { ensureConversation } from './conversations.js';
import { sendMessage, autoResize, updateSendState } from './chat.js';

const docsBtn = document.getElementById('docsBtn');
const docsModal = document.getElementById('docsModal');
const docsModalBody = document.getElementById('docsModalBody');
const docsModalClose = document.getElementById('docsModalClose');
const compareBtn = document.getElementById('compareBtn');
const compareModal = document.getElementById('compareModal');
const compareModalClose = document.getElementById('compareModalClose');
const compareSelect = document.getElementById('compareSelect');
const compareNote = document.getElementById('compareNote');
const compareGo = document.getElementById('compareGo');
const attachBtn = document.getElementById('attachBtn');
const fileInput = document.getElementById('fileInput');

// every document across every conversation, with a delete button per row.
// separate from the per-conversation attachment list already shown inline.
async function renderAllDocuments() {
  docsModalBody.innerHTML = '<div class="modal-empty">Loading…</div>';
  let docs;
  try {
    const res = await api('/documents/');
    docs = await res.json();
  } catch (err) {
    docsModalBody.innerHTML = `<div class="modal-empty">Failed to load: ${escapeHtml(err.message)}</div>`;
    return;
  }

  if (docs.length === 0) {
    docsModalBody.innerHTML = '<div class="modal-empty">No documents uploaded yet.</div>';
    return;
  }

  docsModalBody.innerHTML = '';
  for (const doc of docs) {
    const convTitles = (doc.conversations || []).map(c => c.title || 'New chat');
    const convText = convTitles.length
      ? `in ${convTitles.join(', ')}`
      : 'no conversations';
    const row = document.createElement('div');
    row.className = 'doc-row';
    row.innerHTML = `
      ${ICON_FILE}
      <div class="doc-name" title="${escapeHtml(doc.name)}">
        <span class="doc-name-text">${escapeHtml(doc.name)}</span>
        <span class="doc-convs" title="${escapeHtml(convText)}">${escapeHtml(convText)}</span>
      </div>
      <div class="modal-actions">
        <button class="icon-btn" title="Summarize this document" aria-label="Summarize ${escapeHtml(doc.name)}">${ICON_SUMMARIZE}</button>
        <button class="icon-btn del-btn" title="Delete document" aria-label="Delete ${escapeHtml(doc.name)}">${ICON_X}</button>
      </div>
    `;
    row.querySelector('.modal-actions .del-btn').addEventListener('click', async () => {
      try {
        await api(`/documents/${doc.id}`, { method: 'DELETE' });
        row.remove();
        if (!docsModalBody.querySelector('.doc-row')) {
          docsModalBody.innerHTML = '<div class="modal-empty">No documents uploaded yet.</div>';
        }
        showToast(`"${doc.name}" deleted`);
      } catch (err) {
        showToast(`Failed to delete "${doc.name}"`);
      }
    });
    row.querySelector('.modal-actions button:not(.del-btn)').addEventListener('click', () => summarizeDocument(doc));
    docsModalBody.appendChild(row);
  }
}

function openDocsModal() {
  docsModal.hidden = false;
  renderAllDocuments();
}

export function closeDocsModal() { docsModal.hidden = true; }

async function openCompareModal() {
  compareModal.hidden = false;
  compareNote.value = '';
  compareSelect.innerHTML = '<div class="modal-empty">Loading…</div>';
  try {
    const res = await api('/documents/');
    const docs = await res.json();
    if (!docs.length) {
      compareSelect.innerHTML = '<div class="modal-empty">No documents uploaded yet. Attach a document first.</div>';
      return;
    }
    compareSelect.innerHTML = '';
    for (const doc of docs) {
      const label = document.createElement('label');
      label.className = 'compare-opt';
      const cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.value = doc.id;
      const span = document.createElement('span');
      span.className = 'doc-name';
      span.textContent = doc.name;
      span.title = doc.name;
      label.appendChild(cb);
      label.appendChild(span);
      compareSelect.appendChild(label);
    }
  } catch (err) {
    compareSelect.innerHTML = `<div class="modal-empty">${escapeHtml(err.message)}</div>`;
  }
}

export function closeCompareModal() { compareModal.hidden = true; }

async function summarizeDocument(doc) {
  const conversationId = await ensureConversation();
  try {
    await api('/documents/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, document_ids: [doc.id] }),
    });
  } catch (_) {
    showToast('Failed to link document');
    return;
  }
  closeDocsModal();
  dom.input.value = 'Summarize the key points of this document.';
  state.pendingMode = 'summarize';
  dom.sourceSelect.value = 'documents';
  autoResize();
  updateSendState();
  await sendMessage();
}

export async function uploadDocument(file) {
  const conversationId = await ensureConversation();
  showToast(`Uploading "${file.name}"...`);

  const formData = new FormData();
  formData.append('conversation_id', conversationId);
  formData.append('file', file);

  try {
    await api('/documents/upload', { method: 'POST', body: formData });
    addSystemNote(`Uploaded and indexed "${file.name}"`);
    showToast(`"${file.name}" indexed`);
  } catch (err) {
    addSystemNote(`Failed to upload "${file.name}": ${err.message}`);
    showToast('Upload failed');
  }
}

/* ---- wiring ---- */

docsBtn.addEventListener('click', openDocsModal);
docsModalClose.addEventListener('click', closeDocsModal);
docsModal.addEventListener('click', (e) => { if (e.target === docsModal) closeDocsModal(); });

compareBtn.addEventListener('click', openCompareModal);
compareModalClose.addEventListener('click', closeCompareModal);
compareModal.addEventListener('click', (e) => { if (e.target === compareModal) closeCompareModal(); });

compareGo.addEventListener('click', async () => {
  const ids = [...compareSelect.querySelectorAll('input:checked')].map(i => Number(i.value));
  if (!ids.length) { showToast('Select at least one document'); return; }
  const note = compareNote.value.trim();
  const conversationId = await ensureConversation();
  try {
    await api('/documents/link', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conversation_id: conversationId, document_ids: ids }),
    });
  } catch (_) {
    showToast('Failed to link documents');
    return;
  }
  closeCompareModal();
  dom.input.value = note || 'Compare these documents, highlighting the similarities and differences.';
  state.pendingMode = 'compare';
  dom.sourceSelect.value = 'documents';
  autoResize();
  updateSendState();
  await sendMessage();
});

attachBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) {
    uploadDocument(fileInput.files[0]);
    fileInput.value = '';
  }
});