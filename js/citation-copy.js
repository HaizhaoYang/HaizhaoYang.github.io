// Adds a "Copy BibTeX" button to every .citation-block so visitors can grab the
// citation in one click. Self-contained; no dependencies.
(function () {
  var blocks = document.querySelectorAll('.citation-block');

  Array.prototype.forEach.call(blocks, function (block) {
    // Capture the BibTeX text before injecting the button so the button's own
    // label can never end up in the copied output.
    var bibtex = block.textContent.replace(/\s+$/, '');

    // Wrap the block so the button can sit at the top-right without being
    // clipped by the block's own horizontal scroll.
    var wrap = document.createElement('div');
    wrap.className = 'citation-wrap';
    block.parentNode.insertBefore(wrap, block);
    wrap.appendChild(block);

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'citation-copy-btn';
    btn.setAttribute('aria-label', 'Copy BibTeX citation');
    btn.textContent = 'Copy BibTeX';
    wrap.appendChild(btn);

    var resetTimer;
    btn.addEventListener('click', function () {
      copyText(bibtex).then(function () {
        flash('Copied', true);
      }).catch(function () {
        flash('Copy failed', false);
      });
    });

    function flash(label, copied) {
      btn.textContent = label;
      btn.classList.toggle('is-copied', copied);
      clearTimeout(resetTimer);
      resetTimer = setTimeout(function () {
        btn.textContent = 'Copy BibTeX';
        btn.classList.remove('is-copied');
      }, 1800);
    }
  });

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    // Fallback for non-secure contexts (e.g. file://) and older browsers.
    return new Promise(function (resolve, reject) {
      try {
        var ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.left = '-9999px';
        document.body.appendChild(ta);
        ta.select();
        var ok = document.execCommand('copy');
        document.body.removeChild(ta);
        ok ? resolve() : reject();
      } catch (e) {
        reject(e);
      }
    });
  }
})();
