/* Docs shell: builds nav, TOC, search, theme toggle and copy buttons. No dependencies; works from file://. */
(function () {
  "use strict";

  var REPO = "https://github.com/f4w4z/vaani-technical-assignment";
  var NAV = [
    { group: "Start", pages: [
      ["index", "Overview"], ["quickstart", "Quickstart"], ["requirements", "Requirements"] ] },
    { group: "Part I: Analysis", pages: [
      ["task1", "Architecture analysis"], ["proposal", "Yielding-controller proposal"] ] },
    { group: "Part II: Pipeline", pages: [
      ["pipeline", "Pipeline and config"], ["dataset", "Dataset"], ["metrics", "Metrics and evaluation"],
      ["results", "Results"], ["extension", "Denoising extension"] ] },
    { group: "Reference", pages: [
      ["modules", "Modules and CLI"], ["tests", "Tests"], ["decisions", "Decision log"],
      ["limitations", "Limitations"], ["repro", "Reproducibility"], ["files", "Repository map"],
      ["glossary", "Glossary and FAQ"] ] }
  ];
  var FLAT = [];
  NAV.forEach(function (g) { g.pages.forEach(function (p) { FLAT.push({ slug: p[0], title: p[1] }); }); });

  var slug = document.body.getAttribute("data-page") || "index";
  var idx = FLAT.findIndex(function (p) { return p.slug === slug; });

  function el(tag, attrs, html) {
    var n = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { n.setAttribute(k, attrs[k]); });
    if (html != null) n.innerHTML = html;
    return n;
  }
  function store(key, val) { try { if (val === undefined) return localStorage.getItem(key); localStorage.setItem(key, val); } catch (e) { return null; } }

  /* ---------- theme ---------- */
  var root = document.documentElement;
  var saved = store("docs-theme");
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);
  function isDark() {
    var t = root.getAttribute("data-theme");
    return t ? t === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  /* ---------- shell ---------- */
  var main = document.getElementById("content");
  main.classList.add("content");

  function svg(paths, extra) {
    return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"' + (extra || "") + ">" + paths + "</svg>";
  }
  var ICON = {
    search: svg('<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>'),
    menu: svg('<line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/><line x1="4" x2="20" y1="18" y2="18"/>'),
    sun: svg('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>'),
    moon: svg('<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>'),
    github: svg('<path d="M15 22v-4a4.8 4.8 0 0 0-1-3.5c3 0 6-2 6-5.5.08-1.25-.27-2.48-1-3.5.28-1.15.28-2.35 0-3.5 0 0-1 0-3 1.5-2.64-.5-5.36-.5-8 0C6 2 5 2 5 2c-.3 1.15-.3 2.35 0 3.5A5.4 5.4 0 0 0 4 9c0 3.5 3 5.5 6 5.5-.39.49-.68 1.05-.85 1.65-.17.6-.22 1.23-.15 1.85v4"/><path d="M9 18c-4.51 2-5-2-7-2"/>')
  };

  var top = el("header", { "class": "topbar" });
  top.innerHTML =
    '<button class="btn icon menu-btn" aria-label="Toggle navigation">' + ICON.menu + "</button>" +
    '<a class="brand" href="index.html">Report</a>' +
    '<span class="spacer"></span>' +
    '<button class="btn search-btn" aria-label="Search documentation">' + ICON.search + "<span>Search documentation...</span><kbd>Ctrl K</kbd></button>" +
    '<a class="btn icon top-link" href="' + REPO + '" target="_blank" rel="noopener" aria-label="GitHub repository">' + ICON.github + "</a>" +
    '<button class="btn icon theme-btn" aria-label="Toggle theme"></button>';

  var side = el("nav", { "class": "sidebar", "aria-label": "Documentation" });
  NAV.forEach(function (g) {
    side.appendChild(el("h6", null, g.group));
    g.pages.forEach(function (p) {
      var a = el("a", { href: p[0] + ".html" }, p[1]);
      if (p[0] === slug) a.className = "active";
      side.appendChild(a);
    });
  });

  var toc = el("aside", { "class": "toc", "aria-label": "On this page" });
  var col = el("div", { "class": "main-col" });
  var layout = el("div", { "class": "layout" });

  main.parentNode.insertBefore(top, main);
  main.parentNode.insertBefore(layout, main);
  col.appendChild(main);
  col.appendChild(toc);
  layout.appendChild(side);
  layout.appendChild(col);

  var themeBtn = top.querySelector(".theme-btn");
  function paintTheme() { themeBtn.innerHTML = isDark() ? ICON.sun : ICON.moon; }
  paintTheme();
  themeBtn.addEventListener("click", function () {
    var next = isDark() ? "light" : "dark";
    root.setAttribute("data-theme", next); store("docs-theme", next); paintTheme();
  });
  top.querySelector(".menu-btn").addEventListener("click", function () { document.body.classList.toggle("nav-open"); });
  side.addEventListener("click", function (e) { if (e.target.tagName === "A") document.body.classList.remove("nav-open"); });

  /* ---------- headings, anchors, TOC ---------- */
  var heads = Array.prototype.slice.call(main.querySelectorAll("h2[id], h3[id]"));
  heads.forEach(function (h) {
    var a = el("a", { "class": "anchor", href: "#" + h.id, "aria-label": "Link to this section" }, "#");
    h.appendChild(a);
  });
  if (heads.length > 2) {
    toc.appendChild(el("h6", null, "On this page"));
    heads.forEach(function (h) {
      var label = h.cloneNode(true); var anc = label.querySelector(".anchor"); if (anc) anc.remove();
      toc.appendChild(el("a", { href: "#" + h.id, "class": h.tagName.toLowerCase(), "data-id": h.id }, label.textContent));
    });
    var links = {};
    Array.prototype.forEach.call(toc.querySelectorAll("a"), function (a) { links[a.getAttribute("data-id")] = a; });
    if ("IntersectionObserver" in window) {
      var current = null;
      var io = new IntersectionObserver(function (entries) {
        entries.forEach(function (en) {
          if (en.isIntersecting) {
            if (current) current.classList.remove("active");
            current = links[en.target.id]; if (current) current.classList.add("active");
          }
        });
      }, { rootMargin: "-70px 0px -70% 0px" });
      heads.forEach(function (h) { io.observe(h); });
    }
  } else { toc.style.display = "none"; }

  /* ---------- tables and code ---------- */
  Array.prototype.forEach.call(main.querySelectorAll("table"), function (t) {
    var w = el("div", { "class": "table-wrap" });
    t.parentNode.insertBefore(w, t); w.appendChild(t);
  });
  Array.prototype.forEach.call(main.querySelectorAll("pre"), function (pre) {
    var b = el("button", { "class": "copy-btn", type: "button" }, "Copy");
    b.addEventListener("click", function () {
      var text = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
      function done() { b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1400); }
      if (navigator.clipboard && window.isSecureContext) { navigator.clipboard.writeText(text).then(done, fallback); } else { fallback(); }
      function fallback() {
        var ta = el("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
        document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); done(); } catch (e) { b.textContent = "Press Ctrl+C"; }
        ta.remove();
      }
    });
    pre.appendChild(b);
  });

  /* ---------- pager ---------- */
  var pager = el("div", { "class": "pager" });
  if (idx > 0) pager.appendChild(el("a", { href: FLAT[idx - 1].slug + ".html", "class": "prev" }, "<small>Previous</small>" + FLAT[idx - 1].title));
  else pager.appendChild(el("span", { "class": "gap" }));
  if (idx >= 0 && idx < FLAT.length - 1) pager.appendChild(el("a", { href: FLAT[idx + 1].slug + ".html", "class": "next" }, "<small>Next</small>" + FLAT[idx + 1].title));
  else pager.appendChild(el("span", { "class": "gap" }));
  main.appendChild(pager);
  main.appendChild(el("p", { "class": "footer" },
    'Source code and data: <a href="' + REPO + '" target="_blank" rel="noopener">GitHub repository</a>. ' +
    "Numbers on these pages come from the saved evaluation reports; nothing here is projected."));

  /* ---------- search ---------- */
  var INDEX = window.SEARCH_INDEX || [];
  var overlay = el("div", { "class": "search-overlay", role: "dialog", "aria-label": "Search" });
  overlay.innerHTML =
    '<div class="search-box"><div class="search-head">' + ICON.search +
    '<input type="text" placeholder="Search documentation..." aria-label="Search query" autocomplete="off"></div>' +
    '<div class="search-results"></div><div class="search-foot"><kbd>Up</kbd> <kbd>Down</kbd> move, <kbd>Enter</kbd> open, <kbd>Esc</kbd> close</div></div>';
  document.body.appendChild(overlay);
  var input = overlay.querySelector("input"), results = overlay.querySelector(".search-results");
  var shown = [], sel = 0;

  function esc(s) { return s.replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
  function mark(text, terms) {
    var out = esc(text);
    terms.forEach(function (t) {
      if (!t) return;
      out = out.replace(new RegExp("(" + t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig"), "<mark>$1</mark>");
    });
    return out;
  }
  function snippet(text, terms) {
    var low = text.toLowerCase(), at = -1;
    for (var i = 0; i < terms.length && at < 0; i++) at = low.indexOf(terms[i]);
    var start = Math.max(0, at - 50);
    return (start > 0 ? "..." : "") + text.slice(start, start + 150) + (start + 150 < text.length ? "..." : "");
  }
  function search(q) {
    var terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) return [];
    var scored = [];
    INDEX.forEach(function (e) {
      var h = (e.h || "").toLowerCase(), t = (e.t || "").toLowerCase(), x = (e.x || "").toLowerCase(), s = 0, ok = true;
      terms.forEach(function (term) {
        var m = 0;
        if (h.indexOf(term) >= 0) m += 6;
        if (t.indexOf(term) >= 0) m += 2;
        if (x.indexOf(term) >= 0) m += 1 + Math.min(3, (x.split(term).length - 1) * 0.3);
        if (!m) ok = false; s += m;
      });
      if (ok) scored.push({ e: e, s: s });
    });
    scored.sort(function (a, b) { return b.s - a.s; });
    return scored.slice(0, 12).map(function (r) { return r.e; });
  }
  function render() {
    var q = input.value.trim(), terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    shown = search(q); sel = 0;
    if (!q) { results.innerHTML = '<div class="search-empty">Type to search every page.</div>'; return; }
    if (!shown.length) { results.innerHTML = '<div class="search-empty">No results for "' + esc(q) + '".</div>'; return; }
    results.innerHTML = shown.map(function (e, i) {
      return '<a href="' + e.p + ".html" + (e.id ? "#" + e.id : "") + '" class="' + (i === 0 ? "sel" : "") + '">' +
        '<div class="rt">' + mark(e.h || e.t, terms) + '<span class="rp">' + esc(e.t) + "</span></div>" +
        '<div class="rs">' + mark(snippet(e.x || "", terms), terms) + "</div></a>";
    }).join("");
  }
  function open() { overlay.classList.add("open"); input.value = ""; render(); setTimeout(function () { input.focus(); }, 0); }
  function close() { overlay.classList.remove("open"); }
  function move(d) {
    var items = results.querySelectorAll("a"); if (!items.length) return;
    items[sel].classList.remove("sel"); sel = (sel + d + items.length) % items.length;
    items[sel].classList.add("sel"); items[sel].scrollIntoView({ block: "nearest" });
  }
  top.querySelector(".search-btn").addEventListener("click", open);
  overlay.addEventListener("mousedown", function (e) { if (e.target === overlay) close(); });
  results.addEventListener("click", function () { close(); });
  input.addEventListener("input", render);
  input.addEventListener("keydown", function (e) {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") { var a = results.querySelectorAll("a")[sel]; if (a) { window.location.href = a.getAttribute("href"); close(); } }
  });
  document.addEventListener("keydown", function (e) {
    var typing = /INPUT|TEXTAREA|SELECT/.test((document.activeElement || {}).tagName || "");
    if ((e.key === "/" && !typing) || ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k")) { e.preventDefault(); open(); }
    else if (e.key === "Escape") close();
  });
})();
