// Nerfies' static/js/index.js, rewritten without jQuery, plus three
// additions: repo links resolved at runtime, a reduced-motion guard for
// the autoplaying clips, and a copy button for the BibTeX block.
(function () {
  'use strict';

  // ---- repo links ---------------------------------------------------------
  // The HTML holds paths relative to this folder, so a local clone works
  // with no server and no JavaScript. On GitHub Pages
  // (<owner>.github.io/<repo>/...) the same links are pointed at github.com,
  // which renders the Markdown and lists the folders. The owner is read
  // from the address bar, so no account name is written into this repo.
  function githubRepo() {
    var host = location.hostname.match(/^([A-Za-z0-9-]+)\.github\.io$/);
    if (!host) return null;
    var first = location.pathname.split('/').filter(Boolean)[0];
    if (!first || /\.html?$/i.test(first)) return null;
    return 'https://github.com/' + host[1] + '/' + first;
  }

  function wireRepoLinks() {
    var base = githubRepo();
    if (!base) return;
    var links = document.querySelectorAll('a[data-repo-path]');
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var path = a.getAttribute('data-repo-path');
      var kind = a.getAttribute('data-repo-kind') || 'blob';
      a.href = path ? base + '/' + kind + '/HEAD/' + path : base;
    }
    var line = document.getElementById('bibtex-url');
    if (line) {
      line.textContent = '  url       = {' + base + '},\n';
      line.hidden = false;
    }
  }

  // ---- navbar burger (Nerfies) -------------------------------------------
  // The burger is a real <button>, so Enter and Space already click it.
  function wireBurger() {
    var burger = document.querySelector('.navbar-burger');
    var menu = document.querySelector('.navbar-menu');
    if (!burger || !menu) return;
    function setOpen(open) {
      burger.classList.toggle('is-active', open);
      menu.classList.toggle('is-active', open);
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
    }
    burger.addEventListener('click', function () {
      setOpen(!burger.classList.contains('is-active'));
    });
    // Close the opened menu after a jump to a section of this page.
    var links = menu.querySelectorAll('a[href^="#"]');
    for (var i = 0; i < links.length; i++) {
      links[i].addEventListener('click', function () { setOpen(false); });
    }
  }

  // ---- reduced motion ----------------------------------------------------
  var reduceMotion = window.matchMedia &&
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function calmVideos() {
    if (!reduceMotion) return;
    var vids = document.querySelectorAll('video[autoplay]');
    for (var i = 0; i < vids.length; i++) {
      vids[i].removeAttribute('autoplay');
      vids[i].pause();
      vids[i].controls = true;
    }
  }

  // ---- clips play only while on screen -----------------------------------
  // The carousel and the side-by-side clips carry preload="none" and no
  // autoplay. Each one loads and plays when a quarter of it is visible and
  // pauses when it leaves, so off-screen and cloned slides cost nothing.
  function playWhenVisible(videos) {
    if (!videos.length) return;
    if (!('IntersectionObserver' in window)) {
      if (!reduceMotion) {
        for (var i = 0; i < videos.length; i++) videos[i].autoplay = true;
      }
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      for (var e = 0; e < entries.length; e++) {
        var v = entries[e].target;
        if (entries[e].isIntersecting) {
          if (!reduceMotion) {
            var p = v.play();
            if (p && p.catch) p.catch(function () {});
          }
        } else {
          v.pause();
        }
      }
    }, { threshold: 0.25 });
    for (var j = 0; j < videos.length; j++) io.observe(videos[j]);
  }

  // ---- results carousel (Nerfies options) --------------------------------
  function wireCarousel() {
    if (typeof bulmaCarousel === 'undefined') return;
    bulmaCarousel.attach('#results-carousel', {
      slidesToScroll: 1,
      slidesToShow: 3,
      loop: true,
      infinite: true,
      autoplay: false,
      autoplaySpeed: 3000
    });

    var root = document.querySelector('#results-carousel .slider');
    if (!root) return;
    root.setAttribute('role', 'region');
    root.setAttribute('aria-roledescription', 'carousel');
    root.setAttribute('aria-label', 'Clips by chapter. Use the arrow keys to move.');

    // The library draws its arrows as plain divs: make them buttons.
    var arrows = [
      ['.slider-navigation-previous', 'Previous clip'],
      ['.slider-navigation-next', 'Next clip']
    ];
    for (var a = 0; a < arrows.length; a++) {
      var el = root.querySelector(arrows[a][0]);
      if (!el) continue;
      el.setAttribute('role', 'button');
      el.setAttribute('tabindex', '0');
      el.setAttribute('aria-label', arrows[a][1]);
      el.addEventListener('keydown', function (e) {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          this.click();
        }
      });
    }
    var dots = root.querySelector('.slider-pagination');
    if (dots) dots.setAttribute('aria-hidden', 'true');

    // Clones exist only for the endless loop; hide them from assistive tech
    // and from the tab order.
    var clones = root.querySelectorAll('.slider-item[data-cloned="true"]');
    for (var c = 0; c < clones.length; c++) {
      clones[c].setAttribute('aria-hidden', 'true');
      clones[c].inert = true;
    }

    // Arrow keys inside a clip seek the clip; do not also move the carousel.
    var vids = root.querySelectorAll('video');
    for (var v = 0; v < vids.length; v++) {
      vids[v].addEventListener('keyup', function (e) { e.stopPropagation(); });
    }

    // Slides scrolled out of view leave the tab order too.
    if ('IntersectionObserver' in window) {
      var view = new IntersectionObserver(function (entries) {
        for (var i = 0; i < entries.length; i++) {
          var item = entries[i].target;
          if (item.getAttribute('data-cloned') === 'true') continue;
          item.inert = entries[i].intersectionRatio < 0.5;
        }
      }, { root: root, threshold: [0, 0.5, 1] });
      var items = root.querySelectorAll('.slider-item');
      for (var k = 0; k < items.length; k++) view.observe(items[k]);
    }
  }

  // ---- wavefront scrub (Nerfies "interpolating states") ------------------
  // Each .scrub panel has one range input and one or more .scrub-frames
  // boxes. Every box holds the same number of frames, so one slider moves
  // them in lockstep. Frame 0 is in the HTML; the rest load when the panel
  // nears the viewport, so a phone does not fetch them up front.
  function wireScrubPanel(panel) {
    var slider = panel.querySelector('.scrub-slider');
    var readout = panel.querySelector('.slider-readout');
    var boxes = panel.querySelectorAll('.scrub-frames');
    if (!slider || !boxes.length) return;
    var template = panel.getAttribute('data-readout') || 'frame {k} of {n}';
    var count = parseInt(boxes[0].getAttribute('data-count'), 10);
    var sets = [];

    for (var b = 0; b < boxes.length; b++) {
      var box = boxes[b];
      sets.push({
        box: box,
        base: box.getAttribute('data-base'),
        ext: box.getAttribute('data-ext') || 'webp',
        pad: parseInt(box.getAttribute('data-pad') || '3', 10),
        frames: []
      });
    }

    function frameUrl(set, i) {
      return set.base + '/' + String(i).padStart(set.pad, '0') + '.' + set.ext;
    }

    var loaded = false;
    function preload() {
      if (loaded) return;
      loaded = true;
      for (var s = 0; s < sets.length; s++) {
        for (var i = 0; i < count; i++) {
          var img = new Image();
          img.decoding = 'async';
          img.src = frameUrl(sets[s], i);
          sets[s].frames[i] = img;
        }
      }
    }

    function label(i) {
      return template
        .replace('{i}', String(i))
        .replace('{k}', String(i + 1))
        .replace('{n}', String(count))
        .replace('{last}', String(count - 1));
    }

    function show(i) {
      for (var s = 0; s < sets.length; s++) {
        var img = sets[s].box.querySelector('img');
        if (img) img.src = frameUrl(sets[s], i);
      }
      var text = label(i);
      if (readout) readout.textContent = text;
      slider.setAttribute('aria-valuetext', text);
    }

    slider.max = String(count - 1);
    slider.addEventListener('input', function () {
      preload();
      show(parseInt(slider.value, 10));
    });
    slider.addEventListener('pointerdown', preload);
    slider.addEventListener('focus', preload);

    if ('IntersectionObserver' in window) {
      var io = new IntersectionObserver(function (entries) {
        for (var e = 0; e < entries.length; e++) {
          if (entries[e].isIntersecting) {
            preload();
            io.disconnect();
          }
        }
      }, { rootMargin: '400px 0px' });
      io.observe(panel);
    } else {
      preload();
    }
    show(parseInt(slider.value, 10) || 0);
  }

  function wireScrub() {
    var panels = document.querySelectorAll('.scrub');
    for (var p = 0; p < panels.length; p++) wireScrubPanel(panels[p]);
  }

  // ---- BibTeX copy -------------------------------------------------------
  function wireCopy() {
    var button = document.getElementById('bibtex-copy');
    var code = document.getElementById('bibtex-code');
    if (!button || !code) return;
    var label = button.querySelector('.copy-label');

    function selectCode() {
      var range = document.createRange();
      range.selectNodeContents(code);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
    }

    button.addEventListener('click', function () {
      var text = code.innerText;
      var status = document.getElementById('bibtex-status');
      var done = function (msg, spoken) {
        if (label) label.textContent = msg;
        if (status) status.textContent = spoken;
        setTimeout(function () {
          if (label) label.textContent = 'Copy';
          if (status) status.textContent = '';
        }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          done('Copied', 'BibTeX copied to the clipboard');
        }, function () {
          selectCode();
          done('Selected', 'BibTeX selected, press Control C to copy');
        });
      } else {
        selectCode();
        done('Selected', 'BibTeX selected, press Control C to copy');
      }
    });
  }

  function start() {
    wireRepoLinks();
    wireBurger();
    calmVideos();
    wireCarousel();
    playWhenVisible(document.querySelectorAll('#results-carousel video, video.stacked-video, video.race-video, video.explainer-video'));
    wireScrub();
    wireCopy();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
