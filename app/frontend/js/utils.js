import { dom } from './state.js';

// optional CDN libraries, see the has* flags below. if a CDN is blocked the
// chat still works, it just looks plainer
export const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
export const hasGSAP = typeof gsap !== 'undefined';
export const hasMarked = typeof marked !== 'undefined';
export const hasPurify = typeof DOMPurify !== 'undefined';
export const hasKatex = typeof renderMathInElement !== 'undefined';
export const hasCharts = typeof Chart !== 'undefined';

// backend appends these terminators followed by JSON payloads (the citation
// list, the response details, or a generation error), which the streaming
// response splits on so the markers never reach the markdown
export const SOURCES_MARKER = '@@RESEARCH_SOURCES@@';
export const DETAILS_MARKER = '@@RESEARCH_DETAILS@@';
export const ERROR_MARKER = '@@RESEARCH_ERROR@@';

// icons built in JS, the rest are inline in the markup
export const ICON_X = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M18 6 6 18"/><path d="m6 6 12 12"/></svg>';
export const ICON_FILE = '<svg class="ico doc-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6"/></svg>';
export const ICON_SUMMARIZE = '<svg class="ico" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/></svg>';

export function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export async function api(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res;
}

export function showToast(text) {
  dom.toast.textContent = text;
  dom.toast.classList.add('show');
  clearTimeout(showToast._t);
  showToast._t = setTimeout(() => dom.toast.classList.remove('show'), 2500);
}

export function fmtClock(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function fmtMs(ms) {
  if (ms == null || Number.isNaN(ms)) return '—';
  if (ms < 1000) return Math.round(ms) + ' ms';
  return (ms / 1000).toFixed(2) + ' s';
}

export function fmtNum(n) {
  if (typeof n !== 'number') return n ?? '—';
  if (Number.isInteger(n)) return n.toLocaleString();
  return n.toLocaleString(undefined, { maximumFractionDigits: 2 });
}