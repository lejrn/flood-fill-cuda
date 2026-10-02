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
  function wireBurger() {
    var burger = document.querySelector('.navbar-burger');
    var menu = document.querySelector('.navbar-menu');
    if (!burger || !menu) return;
    burger.addEventListener('click', function () {
      var open = burger.classList.toggle('is-active');
      menu.classList.toggle('is-active', open);
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
    });
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
    // the carousel clones items for the infinite loop; keep clones calm too
    calmVideos();
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
      var done = function (msg) {
        if (label) label.textContent = msg;
        setTimeout(function () { if (label) label.textContent = 'Copy'; }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(function () {
          done('Copied');
        }, function () {
          selectCode();
          done('Selected');
        });
      } else {
        selectCode();
        done('Selected');
      }
    });
  }

  function start() {
    wireRepoLinks();
    wireBurger();
    calmVideos();
    wireCarousel();
    wireScrub();
    wireCopy();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
