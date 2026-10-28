'use strict';

// Animate the paper's bottom-input / top-output framework on one coordinate grid.
(() => {
  const figure = document.querySelector('#method-framework');
  if (!figure) return;
  const svg = figure.querySelector('.paper-framework');
  const layer = svg.querySelector('.paper-flight-layer');
  const timeline = new Map([...svg.querySelectorAll('[data-timeline-frame]')].map(frame => [frame.dataset.timelineFrame, frame]));
  const streams = new Map([...svg.querySelectorAll('[data-model-stream]')].map(stream => [stream.dataset.modelStream, stream]));
  const feedbacks = new Map([...svg.querySelectorAll('[data-feedback]')].map(path => [path.dataset.feedback, path]));
  const cacheLinks = new Map([...svg.querySelectorAll('[data-cache-link]')].map(link => [link.dataset.cacheLink, link]));
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
  const namespace = 'http://www.w3.org/2000/svg';
  const clamp = value => Math.max(0, Math.min(1, value));
  const startDenoising = 3.1;
  const rounds = {h: 2, l: 3, a: 4};
  const periods = {h: 1.15, l: 1.3, a: 1.7};
  const stops = Object.fromEntries(Object.keys(rounds).map(name => [name, startDenoising + rounds[name] * periods[name]]));
  const cycleLength = 12.5;
  const blurAtStop = {h: .55, l: .15};
  const residualNoise = {h: .18, l: .07};
  const positions = {past2:55, past:150, now:340, one:387.5, two:435, coarse:530, end:625, far:880};
  const flights = [];
  const packets = [];
  let intersecting = false;
  let request = 0;
  let previousTime = 0;
  let elapsed = 0;
  let lastDraw = 0;
  let previousStage = '';
  const previousNoiseSteps = {h: -1, l: -1};

  function element(tag, attributes = {}) {
    const node = document.createElementNS(namespace, tag);
    for (const [key, value] of Object.entries(attributes)) node.setAttribute(key, value);
    return node;
  }

  function flight(stream, key, source, target, start, returning = false, destination = null) {
    const group = element('g', {'class':'paper-frame-flight', 'data-flight-stream':stream, 'data-flight-frame':key, 'data-flight-direction':returning ? 'return' : 'sample'});
    const photo = timeline.get(key).querySelector('image').cloneNode();
    photo.setAttribute('x', 0); photo.setAttribute('y', 0);
    if (returning) photo.style.filter = `blur(${blurAtStop[stream]}px)`;
    const border = element('rect', {rx:3, 'class':'paper-flight-border'});
    const grain = returning ? element('rect', {rx:3, 'class':'paper-flight-noise',filter:`url(#paper-noise-${stream})`}) : null;
    group.append(photo);
    if (grain) { grain.style.opacity=residualNoise[stream]; group.append(grain); }
    group.append(border);
    group.style.setProperty('--flight-color', {h:'#be824d',l:'#659357'}[stream]);
    layer.append(group);
    const route = element('path');
    if (returning) route.setAttribute('d', `M${source.x} ${source.y}Q${target.x} 112 ${target.x} ${target.y}`);
    else {
      const side = stream === 'h' ? 45 : 716;
      route.setAttribute('d', `M${source.x} ${source.y}Q${side} 112 ${side} 145V448Q${side} 462 ${side + (stream === 'h' ? 14 : -14)} 462H${target.x}V${target.y}`);
    }
    flights.push({stream,key,source,target,start,returning,destination,group,photo,grain,border,route,length:route.getTotalLength(),duration:returning?1.05:1.2});
  }

  [...svg.querySelectorAll('[data-context]')].forEach((context, index) => {
    const key = context.dataset.context;
    const stream = context.dataset.contextStream;
    const x = stream === 'l' ? 390 : {past2:110,past:156,now:202}[key];
    flight(stream,key,{x:positions[key],y:66,width:42,height:28},{x,y:408.5,width:36,height:25},stream === 'h' ? .1+index*.14 : 1.7,false,context);
  });
  for (const [stream, keys, xs, begin] of [['h',['coarse','far'],[175,245],stops.h+.12],['l',['one','two','end'],[430,500,570],stops.l+.12]]) {
    keys.forEach((key,index) => flight(stream,key,{x:xs[index],y:158,width:48,height:32},{x:positions[key],y:66,width:42,height:28},begin+index*.12,true));
  }

  for (const stream of ['h','l','a']) {
    const group = element('g', {'class':`paper-feedback-packet paper-color-${stream}`, 'data-feedback-packet':stream});
    group.append(element('rect',{width:36,height:24,rx:3,fill:'#f6f8f2'}));
    let photo, grain, curve;
    if (stream === 'a') {
      curve = element('path',{'class':'paper-action-curve',transform:'scale(.24 .364)'});
      group.append(curve);
    } else {
      photo = timeline.get(stream === 'h' ? 'coarse' : 'one').querySelector('image').cloneNode();
      for (const [key,value] of Object.entries({x:0,y:0,width:36,height:24})) photo.setAttribute(key,value);
      grain = element('rect',{width:36,height:24,rx:3,filter:`url(#paper-noise-${stream})`});
      group.append(photo,grain);
    }
    group.append(element('rect',{width:36,height:24,rx:3,'class':'paper-packet-border'}));
    layer.append(group);
    packets.push({stream,group,photo,grain,curve,path:feedbacks.get(stream),length:feedbacks.get(stream).getTotalLength()});
  }

  function phase(time) {
    if (time < 1.7) return ['sample-h','Sparse history conditions the long-horizon stream.'];
    if (time < startDenoising) return ['sample-l','Current observation conditions dense local dynamics.'];
    if (time < stops.h) return ['joint-denoise','Noise enters below; all three streams denoise together.'];
    if (time < stops.l) return ['return-h','Long-horizon predictions are cached; local dynamics continue.'];
    if (time < stops.l+1.6) return ['return-l','Local dynamics are cached; action denoising continues.'];
    if (time < stops.a) return ['refine-action','Action denoising continues with cached visual features.'];
    return ['complete','One completed action chunk, guided by both visual scales.'];
  }

  function trajectory(progress, time) {
    const points = [];
    const amplitude = Math.pow(1 - progress, 1.45);
    for (let index = 0; index < 19; index++) {
      const u = index / 18;
      const x = 7 + u * 136;
      const y = 35 - 14 * Math.sin(u * Math.PI * 2.05) + 6 * Math.sin(u * Math.PI * 3.2);
      const perturbX = 8 * Math.sin(index * 4.3 + 1.7) + 2 * Math.sin(time * 3 + index);
      const perturbY = 22 * Math.sin(index * 2.7 + .9) + 5 * Math.cos(time * 4 + index * 1.5);
      points.push([
        Math.max(4, Math.min(146, x + amplitude * perturbX)),
        Math.max(5, Math.min(61, y + amplitude * perturbY)),
      ]);
    }
    const number = value => value.toFixed(2);
    let path = `M${number(points[0][0])} ${number(points[0][1])}`;
    for (let index = 0; index < points.length - 1; index++) {
      const before = points[Math.max(0, index - 1)];
      const current = points[index];
      const next = points[index + 1];
      const after = points[Math.min(points.length - 1, index + 2)];
      const first = [current[0] + (next[0] - before[0]) / 6, current[1] + (next[1] - before[1]) / 6];
      const second = [next[0] - (after[0] - current[0]) / 6, next[1] - (after[1] - current[1]) / 6];
      path += `C${number(first[0])} ${number(first[1])} ${number(second[0])} ${number(second[1])} ${number(next[0])} ${number(next[1])}`;
    }
    return path;
  }

  function draw(time) {
    const [stage,text] = phase(time);
    if (stage !== previousStage) {
      figure.dataset.stage = stage;
      figure.querySelector('.fw-stage-text').textContent = text;
      previousStage = stage;
    }
    const progress = Object.fromEntries(['h','l','a'].map(name => [name,clamp((time-startDenoising)/(stops[name]-startDenoising))]));
    const reset = time > cycleLength-.45 ? clamp((cycleLength-time)/.45) : 1;
    figure.style.setProperty('--output-visible',clamp((time-startDenoising)/.12));
    for (const [name,stream] of streams) {
      const state = time < startDenoising ? 'waiting' : progress[name] < 1 ? 'denoising' : name === 'a' ? 'ready' : 'cached';
      stream.dataset.streamState = state;
      stream.dataset.completedSteps = Math.min(rounds[name], Math.floor(Math.max(0,time-startDenoising)/periods[name] + 1e-7));
      stream.querySelector('.paper-stream-status').textContent = {waiting:'Input noise',denoising:'Denoising',cached:'Cached',ready:'Ready'}[state];
      feedbacks.get(name).dataset.state = state;
    }
    for (const [name,link] of cacheLinks) {
      link.style.opacity = clamp((time-stops[name])/.18);
      link.dataset.cacheActive = String(time>=stops[name]);
    }
    for (const tile of svg.querySelectorAll('[data-tile-stream]')) {
      const name = tile.dataset.tileStream;
      const amount = clamp(progress[name] - (tile.hasAttribute('data-input') && progress[name]<1 ? .05 : 0));
      const photo = tile.querySelector('.paper-noise-image');
      photo.style.opacity = amount;
      photo.style.filter = `blur(${blurAtStop[name]+(1-progress[name])*1.5}px)`;
      tile.querySelector('.paper-noise-grain').style.opacity = 1-amount*(1-residualNoise[name]);
    }
    for (const frame of timeline.values()) {
      if (!frame.classList.contains('paper-target')) continue;
      let amount = 0;
      let owner = 'h';
      for (const item of flights.filter(flight => flight.returning && flight.key === frame.dataset.timelineFrame)) {
        const returned = clamp((time-item.start-item.duration+.15)/.15);
        if (returned>0) owner = item.stream;
        amount = Math.max(amount,returned);
      }
      frame.querySelector('image').style.filter = `blur(${blurAtStop[owner]}px)`;
      const grain = frame.querySelector('.paper-timeline-noise');
      grain.setAttribute('filter',`url(#paper-noise-${owner})`);
      grain.style.opacity = residualNoise[owner];
      frame.style.opacity = amount*reset;
      frame.style.setProperty('--returned',amount*reset);
    }
    svg.querySelector('.paper-long-continuation').style.opacity = timeline.get('coarse').style.opacity;
    for (const item of flights) {
      const ratio = (time-item.start)/item.duration;
      if (item.destination) item.destination.style.setProperty('--arrived',clamp((ratio-.85)/.15)*reset);
      if (ratio<0 || ratio>1) { item.group.style.opacity=0; continue; }
      const smooth = ratio*ratio*(3-2*ratio);
      const point = item.route.getPointAtLength(smooth*item.length);
      const width = item.source.width+(item.target.width-item.source.width)*smooth;
      const height = item.source.height+(item.target.height-item.source.height)*smooth;
      item.group.setAttribute('transform',`translate(${point.x-width/2} ${point.y-height/2})`);
      item.photo.setAttribute('width',width); item.photo.setAttribute('height',height);
      item.border.setAttribute('width',width); item.border.setAttribute('height',height);
      if (item.grain) { item.grain.setAttribute('width',width); item.grain.setAttribute('height',height); }
      item.group.style.opacity = Math.min(1,ratio*8,(1-ratio)*18);
    }
    const actionPath = trajectory(progress.a,time);
    svg.querySelector('[data-action-output]').setAttribute('d',actionPath);
    svg.querySelector('[data-action-input]').setAttribute('d',trajectory(clamp(progress.a-.04),time));
    for (const packet of packets) {
      if (time<startDenoising || time>=stops[packet.stream]) {packet.group.style.opacity=0;continue;}
      const round = Math.floor((time-startDenoising)/periods[packet.stream]);
      packet.group.dataset.feedbackRound = round+1;
      const ratio = ((time-startDenoising)/periods[packet.stream])%1;
      const point = packet.path.getPointAtLength(ratio*packet.length);
      packet.group.setAttribute('transform',`translate(${point.x-18} ${point.y-12})`);
      packet.group.style.opacity = Math.min(1,ratio*8,(1-ratio)*12);
      if (packet.curve) packet.curve.setAttribute('d',actionPath);
      else {
        packet.photo.style.opacity=progress[packet.stream];
        packet.photo.style.filter=`blur(${blurAtStop[packet.stream]}px)`;
        packet.grain.style.opacity=1-progress[packet.stream]*(1-residualNoise[packet.stream]);
      }
    }
    for (const name of ['h','l']) {
      const step = Math.min(rounds[name],Math.floor(Math.max(0,time-startDenoising)/periods[name]));
      if (step!==previousNoiseSteps[name]) {
        svg.querySelector(`#paper-noise-${name} feTurbulence`).setAttribute('seed',(name==='h'?13:29)+step);
        previousNoiseSteps[name]=step;
      }
    }
  }

  function tick(now) {
    request=0;
    if (!intersecting || document.hidden || reducedMotion.matches) return;
    if (previousTime) elapsed+=(now-previousTime)/1000;
    previousTime=now;
    if (now-lastDraw>36) { draw(elapsed%cycleLength);lastDraw=now; }
    request=requestAnimationFrame(tick);
  }

  function updatePlayback() {
    if (request) cancelAnimationFrame(request);
    request=0; previousTime=0;
    figure.classList.toggle('is-animated',!reducedMotion.matches);
    figure.classList.toggle('is-static',reducedMotion.matches);
    if (reducedMotion.matches) {
      layer.style.display='none'; figure.style.setProperty('--output-visible',1);
      for (const frame of svg.querySelectorAll('.paper-target')) {
        frame.style.opacity=1;
        const owner=['coarse','far'].includes(frame.dataset.timelineFrame)?'h':'l';
        frame.querySelector('image').style.filter=`blur(${blurAtStop[owner]}px)`;
        const grain=frame.querySelector('.paper-timeline-noise');
        grain.setAttribute('filter',`url(#paper-noise-${owner})`);grain.style.opacity=residualNoise[owner];
      }
      svg.querySelector('.paper-long-continuation').style.opacity=1;
      for (const tile of svg.querySelectorAll('[data-tile-stream]')) {
        const photo=tile.querySelector('.paper-noise-image');
        photo.style.opacity=1;photo.style.filter=`blur(${blurAtStop[tile.dataset.tileStream]}px)`;
        tile.querySelector('.paper-noise-grain').style.opacity=residualNoise[tile.dataset.tileStream];
      }
      for (const link of cacheLinks.values()) {link.style.opacity=1;link.dataset.cacheActive='true';}
      for (const context of svg.querySelectorAll('[data-context]')) context.style.setProperty('--arrived',1);
      for (const [name,stream] of streams) {stream.dataset.streamState=name==='a'?'ready':'cached';stream.querySelector('.paper-stream-status').textContent=name==='a'?'Ready':'Cached';}
      svg.querySelector('[data-action-output]').setAttribute('d',trajectory(1,0));
      figure.querySelector('.fw-stage-text').textContent='Visual streams stop and cache before action denoising finishes.';
      return;
    }
    layer.style.display='';
    if (intersecting && !document.hidden) request=requestAnimationFrame(tick);
  }
  document.addEventListener('visibilitychange',updatePlayback);
  reducedMotion.addEventListener('change',updatePlayback);
  if ('IntersectionObserver' in window) new IntersectionObserver(entries=>{intersecting=entries[0].isIntersecting;updatePlayback();},{threshold:.1}).observe(figure);
  else intersecting=true;
  updatePlayback();
})();
