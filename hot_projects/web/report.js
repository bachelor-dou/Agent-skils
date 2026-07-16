/* 报告页交互：主从面板切换 / 侧栏搜索 / 移动端抽屉 / 复制与收藏按钮 */

(function setupMasterDetail() {
  const panels = Array.prototype.slice.call(document.querySelectorAll(".repo-detail"));
  const navItems = Array.prototype.slice.call(document.querySelectorAll(".repo-nav__item"));
  const pager = document.getElementById("pager");
  const pagerPrev = document.getElementById("pager-prev");
  const pagerNext = document.getElementById("pager-next");
  const pagerPos = document.getElementById("pager-pos");
  const sidebarScroll = document.getElementById("sidebar-scroll");

  // 结构化面板不存在（旧版 Markdown 报告）→ 保持平铺 + 锚点跳转
  if (!panels.length || !navItems.length) {
    return;
  }

  document.body.classList.add("js-master");
  let activeIndex = -1;

  function indexFromHash() {
    const hash = decodeURIComponent(window.location.hash || "").replace(/^#/, "");
    if (!hash) {
      return 0;
    }
    const idx = panels.findIndex(function (panel) {
      return panel.id === hash || hash.indexOf(panel.id + "-") === 0;
    });
    return idx >= 0 ? idx : 0;
  }

  function updatePager() {
    if (!pager) {
      return;
    }
    pager.hidden = false;
    if (pagerPos) {
      pagerPos.textContent = (activeIndex + 1) + " / " + panels.length;
    }
    if (pagerPrev) {
      pagerPrev.disabled = activeIndex <= 0;
    }
    if (pagerNext) {
      pagerNext.disabled = activeIndex >= panels.length - 1;
    }
  }

  function activate(index, options) {
    const opts = options || {};
    if (index < 0 || index >= panels.length || index === activeIndex) {
      if (index === activeIndex && opts.scrollTop) {
        window.scrollTo({ top: 0, behavior: "auto" });
      }
      return;
    }
    activeIndex = index;
    panels.forEach(function (panel, i) {
      panel.classList.toggle("is-active", i === index);
    });
    navItems.forEach(function (item, i) {
      item.classList.toggle("is-active", i === index);
    });

    if (opts.updateHash !== false) {
      history.replaceState(null, "", "#" + panels[index].id);
    }
    if (opts.scrollTop !== false) {
      window.scrollTo({ top: 0, behavior: "auto" });
    }
    // 保证侧栏当前项可见
    const activeItem = navItems[index];
    if (sidebarScroll && activeItem && typeof activeItem.scrollIntoView === "function") {
      activeItem.scrollIntoView({ block: "nearest" });
    }
    updatePager();
  }

  navItems.forEach(function (item, index) {
    item.addEventListener("click", function (event) {
      event.preventDefault();
      activate(index);
      document.body.classList.remove("sidebar-open");
    });
  });

  if (pagerPrev) {
    pagerPrev.addEventListener("click", function () {
      activate(activeIndex - 1);
    });
  }
  if (pagerNext) {
    pagerNext.addEventListener("click", function () {
      activate(activeIndex + 1);
    });
  }

  document.addEventListener("keydown", function (event) {
    if (event.target && /^(INPUT|TEXTAREA|SELECT)$/.test(event.target.tagName)) {
      return;
    }
    if (event.key === "ArrowLeft") {
      activate(activeIndex - 1);
    } else if (event.key === "ArrowRight") {
      activate(activeIndex + 1);
    }
  });

  window.addEventListener("hashchange", function () {
    activate(indexFromHash(), { updateHash: false });
  });

  activate(indexFromHash(), { updateHash: false, scrollTop: false });
})();

(function setupSidebarFilters() {
  const input = document.getElementById("repo-search");
  const countLabel = document.getElementById("repo-count");
  const navItems = Array.prototype.slice.call(document.querySelectorAll(".repo-nav__item"));
  const favoritesApi = window.GitHubHotFavorites || null;
  if (!input || !navItems.length) {
    if (input) {
      input.parentElement.style.display = "none";
    }
    return;
  }

  // mode: "all" | "fresh" | "fav"
  const state = { query: "", mode: "all" };

  function isFav(item) {
    const repo = item.getAttribute("data-repo") || "";
    return favoritesApi && repo && favoritesApi.isFavorite(repo);
  }

  function apply() {
    let visible = 0;
    navItems.forEach(function (item) {
      const blob = item.getAttribute("data-search") || "";
      let hit = !state.query || blob.indexOf(state.query) >= 0;
      if (hit && state.mode === "fresh") {
        hit = item.hasAttribute("data-fresh");
      } else if (hit && state.mode === "fav") {
        hit = isFav(item);
      }
      item.classList.toggle("is-hidden", !hit);
      if (hit) {
        visible += 1;
      }
    });
    if (countLabel) {
      countLabel.textContent = visible + "/" + navItems.length;
    }
  }

  input.addEventListener("input", function () {
    state.query = input.value.trim().toLowerCase();
    apply();
  });

  // 过滤 chip：全部 /（有对比时）仅上新 / 仅收藏
  const freshCount = navItems.filter(function (item) {
    return item.hasAttribute("data-fresh");
  }).length;

  const bar = document.createElement("div");
  bar.className = "sidebar__filters";
  const chips = {};

  function makeChip(mode, label) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "filter-chip" + (mode === "all" ? " is-active" : "");
    chip.textContent = label;
    chip.addEventListener("click", function () {
      state.mode = mode;
      Object.keys(chips).forEach(function (m) {
        chips[m].classList.toggle("is-active", m === mode);
      });
      apply();
    });
    chips[mode] = chip;
    bar.appendChild(chip);
    return chip;
  }

  makeChip("all", "全部 " + navItems.length);
  if (freshCount > 0) {
    makeChip("fresh", "仅上新 " + freshCount);
  }
  const favChip = makeChip("fav", "仅收藏");

  const searchBox = input.parentElement;
  searchBox.parentElement.insertBefore(bar, searchBox.nextSibling);

  // 收藏数随服务端数据/用户操作更新；仅收藏视图下需实时重筛
  if (favoritesApi && typeof favoritesApi.subscribe === "function") {
    favoritesApi.subscribe(function () {
      const n = navItems.filter(isFav).length;
      favChip.textContent = "仅收藏 " + n;
      if (state.mode === "fav") {
        apply();
      }
    });
  }

  apply();
})();

(function setupSidebarResizer() {
  const resizer = document.getElementById("sidebar-resizer");
  const app = document.querySelector(".app");
  if (!resizer || !app) {
    return;
  }

  const MIN = 240;
  const MAX = 620;
  const STORE_KEY = "hot-report-sidebar-w";

  function clamp(px) {
    return Math.max(MIN, Math.min(MAX, px));
  }

  function applyWidth(px) {
    app.style.setProperty("--sidebar-w", clamp(px) + "px");
  }

  // 拖拽仅在桌面双栏布局下生效（移动端为抽屉，忽略）
  function isDesktop() {
    return window.matchMedia("(min-width: 961px)").matches;
  }

  const saved = parseInt(window.localStorage.getItem(STORE_KEY) || "", 10);
  if (!isNaN(saved) && isDesktop()) {
    applyWidth(saved);
  }

  let dragging = false;

  function onMove(clientX) {
    const left = app.getBoundingClientRect().left;
    applyWidth(clientX - left);
  }

  function stop() {
    if (!dragging) {
      return;
    }
    dragging = false;
    document.body.classList.remove("is-resizing");
    const current = getComputedStyle(app).getPropertyValue("--sidebar-w").trim();
    const px = parseInt(current, 10);
    if (!isNaN(px)) {
      window.localStorage.setItem(STORE_KEY, String(px));
    }
  }

  resizer.addEventListener("pointerdown", function (event) {
    if (!isDesktop()) {
      return;
    }
    dragging = true;
    document.body.classList.add("is-resizing");
    event.preventDefault();
  });

  window.addEventListener("pointermove", function (event) {
    if (dragging) {
      onMove(event.clientX);
    }
  });

  window.addEventListener("pointerup", stop);
  window.addEventListener("pointercancel", stop);

  // 双击手柄恢复默认宽度
  resizer.addEventListener("dblclick", function () {
    app.style.removeProperty("--sidebar-w");
    window.localStorage.removeItem(STORE_KEY);
  });
})();

(function setupSidebarDrawer() {
  const toggle = document.getElementById("sidebar-toggle");
  const backdrop = document.getElementById("sidebar-backdrop");
  if (!toggle) {
    return;
  }

  toggle.addEventListener("click", function () {
    document.body.classList.toggle("sidebar-open");
  });

  if (backdrop) {
    backdrop.addEventListener("click", function () {
      document.body.classList.remove("sidebar-open");
    });
  }

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      document.body.classList.remove("sidebar-open");
    }
  });
})();

(function setupRepoActionButtons() {
  const container = document.querySelector(".content");
  const favoritesApi = window.GitHubHotFavorites || null;
  if (!container) {
    return;
  }

  const setButtonState = window.HotCommon.applyRepoCopyState;
  const copyText = window.HotCommon.copyText;

  function setFavoriteButtonState(button, repo) {
    const favorited = favoritesApi && favoritesApi.isFavorite(repo);
    const idleMessage = favorited ? "取消收藏 " + repo : "收藏 " + repo;
    button.classList.toggle("is-favorited", !!favorited);
    button.setAttribute("title", idleMessage);
    button.setAttribute("aria-label", idleMessage);
    button.textContent = favorited ? "★" : "☆";
  }

  container.querySelectorAll(".repo-detail").forEach(function (panel) {
    const repo = panel.getAttribute("data-repo") || "";
    const heading = panel.querySelector("h2");
    if (!repo || !heading) {
      return;
    }

    if (!heading.querySelector(".repo-copy-btn")) {
      const copyBtn = document.createElement("button");
      copyBtn.type = "button";
      copyBtn.className = "repo-copy-btn";
      copyBtn.setAttribute("data-repo", repo);
      setButtonState(copyBtn, "idle", repo);
      heading.appendChild(document.createTextNode(" "));
      heading.appendChild(copyBtn);
    }

    if (!heading.querySelector(".repo-favorite-btn")) {
      const favBtn = document.createElement("button");
      favBtn.type = "button";
      favBtn.className = "repo-favorite-btn";
      favBtn.setAttribute("data-repo", repo);
      setFavoriteButtonState(favBtn, repo);
      heading.appendChild(document.createTextNode(" "));
      heading.appendChild(favBtn);
    }
  });

  // 侧栏项目行的收藏 ★：挂到 meta 行行首（上新/NEW 徽章之后）
  document.querySelectorAll(".repo-nav__item").forEach(function (item) {
    const repo = item.getAttribute("data-repo") || "";
    const meta = item.querySelector(".repo-nav__meta");
    if (!repo || !meta || meta.querySelector(".repo-nav__fav")) {
      return;
    }
    const star = document.createElement("button");
    star.type = "button";
    star.className = "repo-nav__fav";
    star.setAttribute("data-repo", repo);
    setFavoriteButtonState(star, repo);
    // 收藏点击不触发面板切换/跳转
    star.addEventListener("click", function (event) {
      event.preventDefault();
      event.stopPropagation();
      if (favoritesApi) {
        favoritesApi.toggle(repo);
      }
    });
    // 放在增长数与排名变化(↑/↓)之间：紧跟增长数之后
    const growthEl = meta.querySelector(".repo-nav__growth");
    if (growthEl) {
      meta.insertBefore(star, growthEl.nextSibling);
    } else {
      meta.appendChild(star);
    }
  });

  function syncAllFavButtons() {
    document.querySelectorAll(".repo-favorite-btn, .repo-nav__fav").forEach(function (button) {
      const repo = button.getAttribute("data-repo") || "";
      if (repo) {
        setFavoriteButtonState(button, repo);
      }
      const item = button.closest(".repo-nav__item");
      if (item) {
        item.classList.toggle("is-fav", favoritesApi && favoritesApi.isFavorite(repo));
      }
    });
  }

  if (favoritesApi && typeof favoritesApi.subscribe === "function") {
    favoritesApi.subscribe(syncAllFavButtons);
  }
  // 预载服务端收藏数据（内部完成后会经 subscribe 回调刷新所有 ★ 状态）
  if (favoritesApi && typeof favoritesApi.ready === "function") {
    favoritesApi.ready();
  }

  container.querySelectorAll(".repo-copy-btn").forEach(function (button) {
    button.addEventListener("click", async function () {
      const repo = button.getAttribute("data-repo") || "";
      if (!repo) {
        return;
      }
      try {
        await copyText(repo);
        setButtonState(button, "copied", repo);
      } catch (_error) {
        setButtonState(button, "failed", repo);
      }
      window.setTimeout(function () {
        setButtonState(button, "idle", repo);
      }, 1400);
    });
  });

  container.querySelectorAll(".repo-favorite-btn").forEach(function (button) {
    button.addEventListener("click", function () {
      const repo = button.getAttribute("data-repo") || "";
      if (repo && favoritesApi) {
        favoritesApi.toggle(repo);
      }
    });
  });
})();

(function setupStarTrend() {
  const buttons = Array.prototype.slice.call(document.querySelectorAll(".repo-trend-btn"));
  if (!buttons.length) {
    return;
  }

  const escapeHtml = (window.HotCommon && window.HotCommon.escapeHtml) || function (s) { return s; };

  // 共享背景遮罩：走势弹层以居中浮层覆盖显示，不撑长页面
  const backdrop = document.createElement("div");
  backdrop.className = "repo-trend-backdrop";
  backdrop.hidden = true;
  document.body.appendChild(backdrop);
  let openPanel = null;

  function closeTrend() {
    if (openPanel) {
      openPanel.hidden = true;
      openPanel.classList.remove("is-modal");
      openPanel = null;
    }
    backdrop.hidden = true;
  }

  backdrop.addEventListener("click", closeTrend);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") closeTrend();
  });
  document.addEventListener("click", function (e) {
    if (e.target.closest && e.target.closest(".repo-trend__close")) closeTrend();
  });

  function trendHeader(repo) {
    return '<div class="repo-trend__head">'
      + '<span class="repo-trend__title">' + escapeHtml(repo) + " · star 走势</span>"
      + '<button type="button" class="repo-trend__close" aria-label="关闭">×</button>'
      + "</div>";
  }

  function fmt(n) {
    return typeof n === "number" ? n.toLocaleString() : "-";
  }

  function kfmt(n) {          // Y 轴刻度紧凑格式：69503→"69.5k"、1234→"1.2k"、850→"850"
    if (typeof n !== "number") return "";
    if (n >= 1000) return (n / 1000).toFixed(n >= 100000 ? 0 : 1) + "k";
    return "" + Math.round(n);
  }

  function renderChart(series) {
    const pts = series.filter(function (p) { return typeof p.star === "number"; });
    if (pts.length < 2) {
      const only = pts[0];
      return '<p class="repo-trend__empty">只有 ' + series.length + ' 周数据，暂无法画走势'
        + (only ? '（' + only.date + '：' + fmt(only.star) + '★）' : '') + '</p>';
    }
    const W = 560, H = 160, padL = 52, padR = 16, padT = 16, padB = 30;
    const stars = pts.map(function (p) { return p.star; });
    const min = Math.min.apply(null, stars);
    const max = Math.max.apply(null, stars);
    const span = (max - min) || 1;
    const n = pts.length;
    const x = function (i) { return padL + (W - padL - padR) * (n === 1 ? 0 : i / (n - 1)); };
    const y = function (v) { return padT + (H - padT - padB) * (1 - (v - min) / span); };

    // Y 轴：min / 中值 / max 三条网格线 + 刻度值
    const yTicks = [min, (min + max) / 2, max];
    const grid = yTicks.map(function (v) {
      const yy = y(v).toFixed(1);
      return '<line class="repo-trend__grid" x1="' + padL + '" y1="' + yy + '" x2="' + (W - padR) + '" y2="' + yy + '"></line>'
        + '<text class="repo-trend__axis" x="' + (padL - 6) + '" y="' + (parseFloat(yy) + 3).toFixed(1) + '" text-anchor="end">' + kfmt(v) + "</text>";
    }).join("");

    // X 轴：每个点的日期（月-日）；点多时隔一个显示
    const stepEvery = n > 8 ? 2 : 1;
    const xlabels = pts.map(function (p, i) {
      if (i % stepEvery !== 0 && i !== n - 1) return "";
      return '<text class="repo-trend__axis" x="' + x(i).toFixed(1) + '" y="' + (H - 10)
        + '" text-anchor="middle">' + (p.date || "").slice(5) + "</text>";
    }).join("");

    const poly = pts.map(function (p, i) { return x(i).toFixed(1) + "," + y(p.star).toFixed(1); }).join(" ");
    // 折线下方的面积填充（更显眼）：折线 + 右下角 + 左下角
    const baseY = (H - padB).toFixed(1);
    const area = poly + " " + x(n - 1).toFixed(1) + "," + baseY + " " + x(0).toFixed(1) + "," + baseY;
    const dots = pts.map(function (p, i) {
      return '<circle cx="' + x(i).toFixed(1) + '" cy="' + y(p.star).toFixed(1) + '" r="4"><title>'
        + p.date + "：" + fmt(p.star) + "★（#" + p.rank + "）</title></circle>";
    }).join("");

    const first = pts[0], last = pts[n - 1];
    const change = last.star - first.star;
    const sign = change >= 0 ? "+" : "";
    return '<div class="repo-trend__meta">' + first.date + " → " + last.date
      + " · 共 " + n + " 周 · 总 star " + sign + fmt(change) + "</div>"
      + '<svg class="repo-trend__svg" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="star 走势">'
      + grid
      + '<polygon class="repo-trend__area" points="' + area + '"></polygon>'
      + '<polyline class="repo-trend__line" points="' + poly + '"></polyline>'
      + dots
      + xlabels
      + "</svg>";
  }

  buttons.forEach(function (btn) {
    btn.addEventListener("click", async function () {
      const repo = btn.getAttribute("data-repo") || "";
      const actions = btn.closest(".repo-detail__actions");
      const panel = actions && actions.nextElementSibling;
      if (!panel || !panel.classList.contains("repo-trend")) {
        return;
      }
      if (openPanel === panel) {   // 再次点同一个 → 收回
        closeTrend();
        return;
      }
      closeTrend();                // 先关掉别的
      openPanel = panel;
      panel.classList.add("is-modal");
      backdrop.hidden = false;
      panel.hidden = false;
      if (panel.dataset.loaded === "1") {  // 已加载过 → 直接显示缓存
        return;
      }
      panel.innerHTML = trendHeader(repo) + '<p class="repo-trend__empty">加载中…</p>';
      try {
        const resp = await fetch("/api/star-trend?repo=" + encodeURIComponent(repo));
        const data = await resp.json();
        const body = (data.series && data.series.length)
          ? renderChart(data.series)
          : '<p class="repo-trend__empty">' + (data.message || "暂无历史数据") + "</p>";
        panel.innerHTML = trendHeader(repo) + body;
        panel.dataset.loaded = "1";
      } catch (_e) {
        panel.innerHTML = trendHeader(repo) + '<p class="repo-trend__empty">加载失败，请重试</p>';
      }
    });
  });
})();
