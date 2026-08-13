import { dom } from './state.js';
import { loadConversations } from './conversations.js';
import { applyMagnetic, initHeroAnimation, spawnParticles } from './motion.js';
import { closeSettingsModal } from './settings.js';
import { closeDocsModal, closeCompareModal } from './documents.js';
import { closeTelemetryModal } from './analytics.js';

/* ---- mobile sidebar drawer ---- */

dom.sidebarToggle.addEventListener('click', () => dom.sidebar.classList.toggle('open'));

// tap outside to close the drawer on mobile
document.addEventListener('click', (e) => {
  if (window.innerWidth > 768) return;
  if (!dom.sidebar.classList.contains('open')) return;
  if (dom.sidebar.contains(e.target) || dom.sidebarToggle.contains(e.target)) return;
  dom.sidebar.classList.remove('open');
});

/* ---- escape closes any open modal ---- */

document.addEventListener('keydown', (e) => {
  if (e.key !== 'Escape') return;
  closeDocsModal();
  closeTelemetryModal();
  closeSettingsModal();
  closeCompareModal();
});

/* ---- magnetic buttons (44px hit area is handled in css) ---- */

applyMagnetic(dom.sendBtn);
applyMagnetic(document.getElementById('attachBtn'));
applyMagnetic(dom.newChatBtn);

/* ---- boot ---- */

spawnParticles();
initHeroAnimation();
loadConversations();
dom.input.focus();