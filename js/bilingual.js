(function () {
  var firstLanguage = '';

  if (navigator.languages && navigator.languages.length) {
    firstLanguage = navigator.languages[0];
  } else if (navigator.language) {
    firstLanguage = navigator.language;
  }

  if (!/^zh(?:-|$)/i.test(String(firstLanguage || '').trim())) {
    document.documentElement.setAttribute('lang', 'en');
    return;
  }

  document.documentElement.setAttribute('lang', 'zh-CN');

  var style = document.createElement('style');
  style.textContent = 'html[lang="zh-CN"] body{font-family:Inter,"PingFang SC","Microsoft YaHei","Noto Sans SC","Noto Sans CJK SC",Arial,sans-serif;}html[lang="zh-CN"] p,html[lang="zh-CN"] li{line-height:1.8;}';
  document.head.appendChild(style);

  function applyChinese() {
    var nodes = document.querySelectorAll('[data-zh]');
    for (var i = 0; i < nodes.length; i += 1) {
      nodes[i].textContent = nodes[i].getAttribute('data-zh');
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyChinese);
  } else {
    applyChinese();
  }
}());
