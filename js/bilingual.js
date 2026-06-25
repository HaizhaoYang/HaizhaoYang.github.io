/* Site preferences: bilingual (EN / 中) + theme (dark / light).
 *
 * The initial theme + language are resolved and written onto <html> by a tiny
 * inline script in each page's <head>, so the first paint is already correct
 * (no flash). This deferred script reads that state, swaps the page text to
 * match, and wires up the floating toggle controls. Choices persist in
 * localStorage and apply across pages without a reload.
 *
 * Kept in plain ES5 to match the rest of the site's scripts. */
(function () {
  var root = document.documentElement;
  var THEME_KEY = 'site-theme';
  var LANG_KEY = 'site-lang';

  function store(key, value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode, etc. */ }
  }

  /* ---------------- Theme ---------------- */

  function currentTheme() {
    return root.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function setTheme(theme) {
    root.setAttribute('data-theme', theme);
    store(THEME_KEY, theme);
    syncControls();
  }

  /* ---------------- Language ---------------- */

  /* Snapshot the English markup once so the toggle can restore it. The English
   * copy carries <strong> emphasis (innerHTML); the Chinese copy in data-zh is
   * plain text, so it goes in via textContent. */
  var i18nNodes = document.querySelectorAll('[data-zh]');
  var enCache = [];
  for (var n = 0; n < i18nNodes.length; n += 1) {
    enCache[n] = i18nNodes[n].innerHTML;
  }

  function currentLang() {
    return root.getAttribute('data-lang') === 'zh' ? 'zh' : 'en';
  }

  function renderLang(lang) {
    for (var i = 0; i < i18nNodes.length; i += 1) {
      if (lang === 'zh') {
        i18nNodes[i].textContent = i18nNodes[i].getAttribute('data-zh');
      } else {
        i18nNodes[i].innerHTML = enCache[i];
      }
    }
    root.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
    root.setAttribute('data-lang', lang);
  }

  function setLang(lang) {
    renderLang(lang);
    store(LANG_KEY, lang);
    syncControls();
  }

  /* ---------------- Controls ---------------- */

  var SUN_SVG = '<svg class="sun" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"></circle><path d="M12 2.6v2.1M12 19.3v2.1M4.7 4.7l1.5 1.5M17.8 17.8l1.5 1.5M2.6 12h2.1M19.3 12h2.1M4.7 19.3l1.5-1.5M17.8 6.2l1.5-1.5"></path></svg>';
  var MOON_SVG = '<svg class="moon" viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.5 14.6A8 8 0 0 1 9.4 3.5 7 7 0 1 0 20.5 14.6z"></path></svg>';

  var controlsEl = null;

  function syncControls() {
    if (!controlsEl) { return; }
    var lang = currentLang();
    var langBtns = controlsEl.querySelectorAll('.lang-btn');
    for (var i = 0; i < langBtns.length; i += 1) {
      var on = langBtns[i].getAttribute('data-lang-set') === lang;
      langBtns[i].setAttribute('aria-pressed', on ? 'true' : 'false');
    }
    var themeBtn = controlsEl.querySelector('.theme-toggle');
    if (themeBtn) {
      var goingLight = currentTheme() !== 'light';
      themeBtn.setAttribute('aria-label',
        goingLight
          ? (lang === 'zh' ? '切换到浅色主题' : 'Switch to light theme')
          : (lang === 'zh' ? '切换到深色主题' : 'Switch to dark theme'));
    }
  }

  function makeLangButton(code, label) {
    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'lang-btn';
    btn.setAttribute('data-lang-set', code);
    btn.textContent = label;
    btn.addEventListener('click', function () { setLang(code); });
    return btn;
  }

  function buildControls() {
    var wrap = document.createElement('div');
    wrap.className = 'page-controls';
    wrap.setAttribute('role', 'group');
    wrap.setAttribute('aria-label', 'Language and theme');

    var langGroup = document.createElement('div');
    langGroup.className = 'lang-toggle';
    langGroup.appendChild(makeLangButton('en', 'EN'));
    langGroup.appendChild(makeLangButton('zh', '中'));

    var themeBtn = document.createElement('button');
    themeBtn.type = 'button';
    themeBtn.className = 'theme-toggle';
    themeBtn.innerHTML = SUN_SVG + MOON_SVG;
    themeBtn.addEventListener('click', function () {
      setTheme(currentTheme() === 'light' ? 'dark' : 'light');
    });

    wrap.appendChild(langGroup);
    wrap.appendChild(themeBtn);
    return wrap;
  }

  function init() {
    /* Bring the DOM text in line with the language the head script chose. */
    renderLang(currentLang());
    controlsEl = buildControls();
    document.body.appendChild(controlsEl);
    syncControls();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
