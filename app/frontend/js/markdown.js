import { escapeHtml, hasMarked, hasPurify, hasKatex } from './utils.js';

if (hasMarked) marked.setOptions({ breaks: true, gfm: true });

// only used if marked/dompurify didn't load
function renderMarkdownBasic(raw) {
  const parts = raw.split(/```([\s\S]*?)```/g);
  let html = '';
  for (let i = 0; i < parts.length; i++) {
    if (i % 2 === 1) {
      html += `<pre><code>${escapeHtml(parts[i].replace(/^\w*\n/, ''))}</code></pre>`;
      continue;
    }
    const escaped = escapeHtml(parts[i]);
    for (const block of escaped.split(/\n{2,}/)) {
      if (!block.trim()) continue;
      const lines = block.split('\n');
      const isList = lines.every(l => /^\s*[-*]\s+/.test(l));
      const inline = (s) => s
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
      if (isList) {
        html += '<ul>' + lines.map(l => `<li>${inline(l.replace(/^\s*[-*]\s+/, ''))}</li>`).join('') + '</ul>';
      } else {
        html += `<p>${inline(block).replace(/\n/g, '<br>')}</p>`;
      }
    }
  }
  return html || escapeHtml(raw);
}

// marked treats \( and \[ as escaped punctuation and eats the backslash, so
// those delimiters never survive to KaTeX. Stash math behind placeholders
// first, put it back after parsing. Code spans are matched first in the same
// pass so a $ inside `code` doesn't get grabbed.
const MATH_PATTERN = new RegExp([
  '(```[\\s\\S]*?```|`[^`\\n]*`)|',
  '(\\$\\$[\\s\\S]*?\\$\\$',
  '|\\\\\\[[\\s\\S]*?\\\\\\]',
  '|\\\\\\([\\s\\S]*?\\\\\\)',
  '|\\$(?!\\s)(?:\\\\.|[^\\\\$\\n])*?[^\\s\\\\$]\\$)',
].join(''), 'g');

function protectMath(raw) {
  const stash = [];
  const text = raw.replace(MATH_PATTERN, (match, code, math) => {
    if (code !== undefined) return code;
    stash.push(math);
    return `@@MATH${stash.length - 1}@@`;
  });
  return { text, stash };
}

// escaped on the way back in, so "$a<b$" survives and we don't hand
// innerHTML anything raw
function restoreMath(html, stash) {
  if (!stash.length) return html;
  return html.replace(/@@MATH(\d+)@@/g, (m, i) => {
    const src = stash[Number(i)];
    return src === undefined ? m : escapeHtml(src);
  });
}

// sanitise before innerHTML, model output is untrusted
function renderRich(raw) {
  const { text, stash } = protectMath(raw);
  const html = (hasMarked && hasPurify)
    ? DOMPurify.sanitize(marked.parse(text))
    : renderMarkdownBasic(text);
  return restoreMath(html, stash);
}

export function typesetMath(el) {
  if (!hasKatex || !el) return;
  try {
    renderMathInElement(el, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '\\[', right: '\\]', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
      ],
      ignoredTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code'],
      throwOnError: false,
    });
  } catch (_) { /* a bad expression shouldn't take out the whole message */ }
}

// math: false while streaming. half-finished expressions are wrong to
// typeset and it's wasted work on every frame anyway
export function setContent(el, raw, { math = true } = {}) {
  el.innerHTML = renderRich(raw);
  if (math) typesetMath(el);
}