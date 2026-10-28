'use strict';

// Keep navigation usable without changing the desktop layout.
const navToggle = document.querySelector('#navToggle');
const nav = document.querySelector('#navLinks');
navToggle.addEventListener('click', () => {
  const expanded = navToggle.getAttribute('aria-expanded') !== 'true';
  navToggle.setAttribute('aria-expanded', String(expanded));
  nav.classList.toggle('is-open', expanded);
});
nav.addEventListener('click', event => {
  if (!event.target.closest('a')) return;
  nav.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
});
nav.addEventListener('keydown', event => {
  if (event.key !== 'Escape') return;
  nav.classList.remove('is-open');
  navToggle.setAttribute('aria-expanded', 'false');
  navToggle.focus();
});

// Native video controls keep every comparison available directly.
const videoStatus = document.querySelector('#videoStatus');
for (const video of document.querySelectorAll('video')) {
  video.addEventListener('error', () => {
    videoStatus.textContent = 'A rollout could not load. Reload the page or open its video file.';
  });
}

// Image links still open the full figure when JavaScript is unavailable.
const dialog = document.querySelector('#figureDialog');
const expandedFigure = document.querySelector('#expandedFigure');
let figureTrigger;
for (const link of document.querySelectorAll('[data-lightbox]')) {
  link.addEventListener('click', event => {
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    if (typeof dialog.showModal !== 'function') return;
    event.preventDefault();
    figureTrigger = link;
    const thumbnail = link.querySelector('img');
    expandedFigure.src = link.href;
    expandedFigure.alt = thumbnail.alt;
    document.querySelector('#figureCaption').textContent = link.closest('figure').querySelector('figcaption')?.textContent || thumbnail.alt;
    dialog.showModal();
  });
}
document.querySelector('#closeFigure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target !== dialog) return;
  const bounds = dialog.getBoundingClientRect();
  if (event.clientX < bounds.left || event.clientX > bounds.right || event.clientY < bounds.top || event.clientY > bounds.bottom) dialog.close();
});
dialog.addEventListener('close', () => {
  expandedFigure.removeAttribute('src');
  figureTrigger?.focus({ preventScroll: true });
});
