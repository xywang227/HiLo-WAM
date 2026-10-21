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

// Progressive enhancement: all panels remain readable when JavaScript is disabled.
const tabGroups = [];
for (const group of document.querySelectorAll('[data-tabs]')) {
  const list = group.querySelector('[data-tab-list]');
  const buttons = [...list.querySelectorAll('[data-tab]')];
  const panels = buttons.map(button => document.getElementById(button.dataset.tab));
  const activate = (index, focus = false) => {
    buttons.forEach((button, i) => {
      button.setAttribute('aria-selected', String(i === index));
      button.tabIndex = i === index ? 0 : -1;
      panels[i].hidden = i !== index;
      if (i !== index) panels[i].querySelectorAll('video').forEach(video => video.pause());
    });
    if (focus) buttons[index].focus();
  };
  list.setAttribute('role', 'tablist');
  buttons.forEach((button, index) => {
    button.setAttribute('role', 'tab');
    button.setAttribute('aria-controls', panels[index].id);
    panels[index].setAttribute('role', 'tabpanel');
    panels[index].setAttribute('aria-labelledby', button.id);
    panels[index].tabIndex = 0;
    button.addEventListener('click', () => activate(index));
    button.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % buttons.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + buttons.length) % buttons.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = buttons.length - 1;
      if (next === undefined) return;
      event.preventDefault();
      activate(next, true);
    });
  });
  list.hidden = false;
  activate(0);
  tabGroups.push({ panels, activate });
}
function revealAnchor(hash, scroll = false) {
  let id;
  try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
  const target = document.getElementById(id);
  if (!target) return;
  for (const { panels, activate } of tabGroups) {
    const index = panels.findIndex(panel => panel === target || panel.contains(target));
    if (index >= 0) activate(index);
  }
  if (scroll) requestAnimationFrame(() => target.scrollIntoView({ block: 'start' }));
}
document.addEventListener('click', event => {
  const link = event.target.closest('a[href^="#"]');
  if (link) revealAnchor(link.hash);
});
window.addEventListener('hashchange', () => revealAnchor(location.hash, true));
if (location.hash) revealAnchor(location.hash, true);

// Native controls remain available if a row cannot start all of its videos.
const videos = [...document.querySelectorAll('video')];
const videoStatus = document.querySelector('#videoStatus');
let playbackSpeed = 1;
for (const video of videos) {
  video.addEventListener('loadedmetadata', () => { video.playbackRate = playbackSpeed; });
  video.addEventListener('error', () => {
    videoStatus.textContent = 'A rollout could not load. Please reload the page or open its video file.';
  });
}
for (const button of document.querySelectorAll('[data-speed]')) {
  button.addEventListener('click', () => {
    playbackSpeed = Number(button.dataset.speed);
    videos.forEach(video => { video.playbackRate = playbackSpeed; });
    document.querySelectorAll('[data-speed]').forEach(control => {
      control.setAttribute('aria-pressed', String(control === button));
    });
    videoStatus.textContent = `All rollouts set to ${playbackSpeed}× speed.`;
  });
}
for (const row of document.querySelectorAll('.video-task-row')) {
  const rowVideos = [...row.querySelectorAll('video')];
  row.querySelector('[data-play-row]').addEventListener('click', async () => {
    const results = await Promise.allSettled(rowVideos.map(video => {
      video.currentTime = 0;
      video.playbackRate = playbackSpeed;
      return video.play();
    }));
    const failed = results.some(result => result.status === 'rejected');
    videoStatus.textContent = failed
      ? 'Some videos could not start. Use the individual video controls to retry.'
      : `${row.querySelector('h4').textContent}: playing from the beginning at ${playbackSpeed}×.`;
  });
  row.querySelector('[data-pause-row]').addEventListener('click', () => {
    rowVideos.forEach(video => video.pause());
    videoStatus.textContent = `${row.querySelector('h4').textContent}: paused.`;
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
