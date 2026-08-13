import { hasGSAP, reduceMotion } from './utils.js';

document.addEventListener('visibilitychange', () => {
  document.body.classList.toggle('anim-paused', document.hidden);
  if (document.hidden) {
    stopHeroBeatLoop();
    if (hasGSAP) gsap.globalTimeline.pause();
  } else {
    if (hasGSAP) gsap.globalTimeline.resume();
    startHeroBeatLoop();
  }
});

// 14 is about the limit before it starts looking like dust on the screen
export function spawnParticles() {
  if (reduceMotion) return;
  const host = document.getElementById('particles');
  const frag = document.createDocumentFragment();
  for (let i = 0; i < 14; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = 2 + Math.random() * 3;
    p.style.width = size + 'px';
    p.style.height = size + 'px';
    p.style.left = Math.random() * 100 + 'vw';
    p.style.top = Math.random() * 100 + 'vh';
    p.style.setProperty('--dur', (12 + Math.random() * 14) + 's');
    p.style.setProperty('--delay', (Math.random() * 8) + 's');
    p.style.setProperty('--dx', (Math.random() * 70 - 35) + 'px');
    p.style.setProperty('--dy', (-25 - Math.random() * 55) + 'px');
    frag.appendChild(p);
  }
  host.appendChild(frag);
}

// wrap each word so it can't break mid-word, then one span per char
function splitHeroLetters() {
  const el = document.getElementById('heroText');
  if (!el) return null;
  const text = el.textContent;
  el.textContent = '';
  let idx = 0;
  const words = text.split(' ');
  words.forEach((word, wi) => {
    const wordSpan = document.createElement('span');
    wordSpan.style.display = 'inline-block';
    wordSpan.style.whiteSpace = 'nowrap';
    [...word].forEach((ch) => {
      const letter = document.createElement('span');
      letter.className = 'letter';
      letter.style.setProperty('--i', idx++);
      letter.textContent = ch;
      wordSpan.appendChild(letter);
    });
    el.appendChild(wordSpan);
    if (wi < words.length - 1) el.appendChild(document.createTextNode(' '));
  });
  return el;
}

let beatTimer = null;

// Beats stick to y/rotation, the breathing tween owns yPercent/scale.
// Different transform components, so GSAP composes them instead of one
// overwriting the other. Don't move either onto the other's props.
function beatWave(letters) {
  return gsap.timeline()
    .to(letters, { y: -12, rotation: 4, duration: 0.28, ease: 'power2.out', stagger: 0.018 })
    .to(letters, { y: 0, rotation: 0, duration: 0.5, ease: 'elastic.out(1, 0.55)', stagger: 0.018 }, '-=0.08');
}
function beatBounce(letters) {
  return gsap.timeline()
    .to(letters, { y: -18, duration: 0.24, ease: 'power2.out', stagger: 0.014 })
    .to(letters, { y: 0, duration: 0.55, ease: 'bounce.out', stagger: 0.014 }, '-=0.05');
}
function beatTilt(letters) {
  return gsap.timeline()
    .to(letters, { rotation: (i) => (i % 2 ? 7 : -7), duration: 0.3, ease: 'power2.out', stagger: 0.016 })
    .to(letters, { rotation: 0, duration: 0.6, ease: 'elastic.out(1, 0.5)', stagger: 0.016 }, '-=0.1');
}

// picked at random each time so it doesn't feel like a loop
const heroBeats = [beatWave, beatBounce, beatTilt];

function playHeroBeat() {
  const heroText = document.getElementById('heroText');
  if (!heroText || document.hidden) return;
  const letters = heroText.querySelectorAll('.letter');
  if (!letters.length) return;
  heroBeats[Math.floor(Math.random() * heroBeats.length)](letters);
}

function startHeroBeatLoop() {
  if (!hasGSAP || reduceMotion || document.hidden) return;
  stopHeroBeatLoop();
  if (!document.getElementById('heroText')) return;
  beatTimer = setInterval(playHeroBeat, 4000);
}

export function stopHeroBeatLoop() {
  if (beatTimer) { clearInterval(beatTimer); beatTimer = null; }
}

// re-run every time the empty state is remounted. matchMedia gives us the
// reduced-motion branch for free and reverts itself if the setting flips.
let heroMM = null;
export function initHeroAnimation() {
  if (heroMM) { heroMM.revert(); heroMM = null; }

  const el = splitHeroLetters();
  if (!el) { stopHeroBeatLoop(); return; }
  if (!hasGSAP) { startHeroBeatLoop(); return; }

  const letters = el.querySelectorAll('.letter');

  heroMM = gsap.matchMedia();
  heroMM.add({
    motionOk: '(prefers-reduced-motion: no-preference)',
    reduced: '(prefers-reduced-motion: reduce)',
  }, (ctx) => {
    if (ctx.conditions.reduced) return;

    // autoAlpha, not opacity, so an interrupted tween can't leave them
    // invisible but still hit-testable
    gsap.from(letters, {
      autoAlpha: 0, y: 28,
      duration: 0.6, stagger: 0.025, ease: 'power3.out',
    });

    gsap.to(letters, {
      yPercent: -5, scale: 1.03,
      duration: 1.8, ease: 'sine.inOut',
      repeat: -1, yoyo: true,
      stagger: { each: 0.05, from: 'start' },
    });
  });

  startHeroBeatLoop();
}

// one listener on document rather than rebinding on every remount
if (hasGSAP && !reduceMotion) {
  const px = gsap.quickTo('#heroWrap', 'x', { duration: 0.7, ease: 'power3' });
  const py = gsap.quickTo('#heroWrap', 'y', { duration: 0.7, ease: 'power3' });
  document.addEventListener('mousemove', (e) => {
    if (!document.getElementById('heroWrap')) return;
    px((e.clientX / window.innerWidth - 0.5) * 16);
    py((e.clientY / window.innerHeight - 0.5) * 11);
  });
}

export function applyMagnetic(btn, strength = 12) {
  if (!hasGSAP || reduceMotion) return;
  const moveX = gsap.quickTo(btn, 'x', { duration: 0.3, ease: 'power3' });
  const moveY = gsap.quickTo(btn, 'y', { duration: 0.3, ease: 'power3' });
  btn.addEventListener('mousemove', (e) => {
    const r = btn.getBoundingClientRect();
    moveX((e.clientX - (r.left + r.width / 2)) * (strength / 100));
    moveY((e.clientY - (r.top + r.height / 2)) * (strength / 100));
  });
  btn.addEventListener('mouseleave', () => { moveX(0); moveY(0); });
}