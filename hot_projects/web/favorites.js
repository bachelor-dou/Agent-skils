/* 收藏：服务端持久化（按 user_id 全局），前端内存缓存 + 乐观更新。
   对外接口 isFavorite / toggle / subscribe / getAll 保持同步语义，
   report.js 等调用方无需改动；数据在页面加载时经 ready() 预载。 */
(function (global) {
  "use strict";

  const OLD_KEY = "gh-hot-favorites-v1";
  const PING_KEY = "gh-hot-favorites-ping";   // localStorage 跨标签页同步的回退通道
  const CHANNEL = "gh-hot-favorites";
  const listeners = new Set();
  const MAX_TAG_LEN = 20;

  let items = new Map();          // repo -> {repo, short_desc, category, ...}（内存缓存，保序）
  let readyPromise = null;
  let defaultTags = [];           // 服务端预置标签（/api/favorite-tags）
  let activePicker = null;        // 当前打开的标签选择浮层（同一时刻只允许一个）

  // 跨标签页同步：某页改动收藏后广播，其它已打开的页（报告页/聊天页）自动重新拉取。
  // BroadcastChannel 不会回发给发送方本身；无该 API 时退回 localStorage storage 事件。
  let channel = null;
  try {
    channel = "BroadcastChannel" in global ? new global.BroadcastChannel(CHANNEL) : null;
  } catch (_e) {
    channel = null;
  }

  function userId() {
    return global.HotUser ? global.HotUser.getId() : "";
  }

  function broadcastChange() {
    try {
      if (channel) {
        channel.postMessage({ type: "changed", uid: userId() });
      } else {
        global.localStorage.setItem(PING_KEY, userId() + "@" + Date.now());
      }
    } catch (_e) {}
  }

  function onExternalChange(uid) {
    // 只有当广播来自同一用户（或未标注用户）时才刷新，避免多身份互相干扰
    if (!uid || uid === userId()) {
      refresh();
    }
  }

  if (channel) {
    channel.onmessage = function (e) {
      if (e && e.data && e.data.type === "changed") {
        onExternalChange(e.data.uid);
      }
    };
  }
  global.addEventListener("storage", function (e) {
    if (e.key === PING_KEY && e.newValue) {
      onExternalChange(String(e.newValue).split("@")[0]);
    }
  });

  function escHtml(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  function cleanTag(tag) {
    return String(tag || "").replace(/\s+/g, " ").trim().slice(0, MAX_TAG_LEN);
  }

  function currentReport() {
    // 报告页 URL 形如 /api/reports/2026-07-01.md/html
    const m = (global.location.pathname || "").match(/\/api\/reports\/([^/]+)\/html/);
    return m ? decodeURIComponent(m[1]) : "";
  }

  function notify() {
    const list = getAll();
    listeners.forEach(function (fn) {
      try {
        fn(list);
      } catch (_e) {}
    });
  }

  async function apiGet(uid) {
    const resp = await fetch("/api/favorites?user_id=" + encodeURIComponent(uid));
    if (!resp.ok) {
      throw new Error("load favorites failed");
    }
    const data = await resp.json();
    return (data.favorites || []).filter(function (x) {
      return x && x.repo;
    });
  }

  function loadItems(list) {
    items = new Map();
    (list || []).forEach(function (x) {
      items.set(x.repo, x);
    });
  }

  async function apiSet(uid, repo, action, category, shortDesc) {
    const body = {
      user_id: uid,
      repo: repo,
      action: action,
      source_report: currentReport(),
    };
    if (category != null) {
      body.category = category;  // "" 表示未分类；不传则服务端保留原分类
    }
    if (shortDesc !== undefined) {
      body.short_desc = shortDesc;  // 用户手动编辑概要；不传则服务端按需自动生成
    }
    const resp = await fetch("/api/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      throw new Error("update favorite failed");
    }
  }

  async function loadTags() {
    try {
      const resp = await fetch("/api/favorite-tags");
      if (resp.ok) {
        const data = await resp.json();
        defaultTags = (data.tags || []).map(cleanTag).filter(Boolean);
      }
    } catch (_e) {
      defaultTags = [];
    }
  }

  // 一次性迁移旧版 localStorage 收藏到服务端，成功后清除旧 key
  async function migrateLegacy(uid) {
    let legacy = [];
    try {
      const raw = global.localStorage.getItem(OLD_KEY);
      if (!raw) {
        return;
      }
      const parsed = JSON.parse(raw);
      legacy = (Array.isArray(parsed) ? parsed : [])
        .map(function (x) {
          return typeof x === "string" ? x : (x && x.repo) || "";
        })
        .filter(Boolean);
    } catch (_e) {
      return;
    }
    for (const repo of legacy) {
      try {
        await apiSet(uid, repo, "add");
      } catch (_e) {}
    }
    try {
      global.localStorage.removeItem(OLD_KEY);
    } catch (_e) {}
  }

  function ready() {
    if (!readyPromise) {
      readyPromise = (async function () {
        const uid = userId();
        await Promise.all([loadTags(), migrateLegacy(uid)]);
        try {
          loadItems(await apiGet(uid));
        } catch (_e) {
          items = new Map();
        }
        notify();
      })();
    }
    return readyPromise;
  }

  // 从服务端重新拉取（agent 在对话中新增收藏后调用，使收藏栏同步）
  async function refresh() {
    try {
      loadItems(await apiGet(userId()));
      notify();
    } catch (_e) {}
    return getAll();
  }

  function isFavorite(repo) {
    return items.has(String(repo || "").trim());
  }

  function getAll() {
    return Array.from(items.values());
  }

  function getCategory(repo) {
    const it = items.get(String(repo || "").trim());
    return (it && it.category) || "";
  }

  // 乐观新增置顶（带分类），失败回滚
  async function addFav(name, category) {
    const next = new Map([[name, { repo: name, short_desc: "", category: category }]]);
    items.forEach(function (v, k) {
      next.set(k, v);
    });
    items = next;
    notify();
    try {
      await apiSet(userId(), name, "add", category);
      broadcastChange();
      await refresh();  // 概要与上榜次数由服务端补全，乐观插入的占位项里没有
    } catch (_e) {
      items.delete(name);
      notify();
    }
    return items.has(name);
  }

  async function removeFav(name) {
    const prev = items.get(name);
    items.delete(name);
    notify();
    try {
      await apiSet(userId(), name, "remove");
      broadcastChange();
    } catch (_e) {
      const restore = new Map([[name, prev]]);
      items.forEach(function (v, k) {
        restore.set(k, v);
      });
      items = restore;
      notify();
    }
    return items.has(name);
  }

  // 已收藏→直接取消；未收藏→弹标签选择框，选定后带分类收藏（取消则不收藏）
  async function toggle(repo, anchorEl) {
    const name = String(repo || "").trim();
    if (!name) {
      return false;
    }
    if (items.has(name)) {
      return removeFav(name);
    }
    const tag = await chooseTag(anchorEl, "");
    if (tag === null) {
      return false;  // 用户取消，不收藏
    }
    return addFav(name, tag);
  }

  // 给已收藏项目改分类（收藏栏里点分类标签时用）
  async function retag(repo, anchorEl) {
    const name = String(repo || "").trim();
    const it = items.get(name);
    if (!it) {
      return "";
    }
    const tag = await chooseTag(anchorEl, it.category || "");
    if (tag === null) {
      return it.category || "";
    }
    const prev = it.category || "";
    it.category = tag;
    notify();
    try {
      await apiSet(userId(), name, "add", tag);
      broadcastChange();
    } catch (_e) {
      it.category = prev;
      notify();
    }
    return it.category || "";
  }

  // 手动编辑收藏概要（short_desc 收藏后不会自动更新，供用户自行修改）
  async function setDesc(repo, text) {
    const name = String(repo || "").trim();
    const it = items.get(name);
    if (!it) {
      return "";
    }
    const prev = it.short_desc || "";
    const next = String(text == null ? "" : text).trim().slice(0, 60);
    if (next === prev) {
      return prev;
    }
    it.short_desc = next;
    notify();
    try {
      await apiSet(userId(), name, "add", null, next);
      broadcastChange();
    } catch (_e) {
      it.short_desc = prev;
      notify();
    }
    return it.short_desc;
  }

  // ── 标签选择浮层（report / chat 两页共用；同一时刻只开一个）──
  function closePicker() {
    if (activePicker) {
      activePicker.cleanup();
      activePicker = null;
    }
  }

  function tagOptions(current) {
    const out = [];
    const seen = new Set();
    function add(t) {
      const v = cleanTag(t);
      if (v && !seen.has(v)) {
        seen.add(v);
        out.push(v);
      }
    }
    defaultTags.forEach(add);
    items.forEach(function (x) {
      add(x.category);
    });
    add(current);
    return out;
  }

  function positionPicker(pop, anchorEl) {
    const r = anchorEl && anchorEl.getBoundingClientRect
      ? anchorEl.getBoundingClientRect() : null;
    const w = pop.offsetWidth || 220;
    const h = pop.offsetHeight || 160;
    let left, top;
    if (r) {
      left = Math.min(r.left, global.innerWidth - w - 8);
      top = r.bottom + 6;
      if (top + h > global.innerHeight - 8) {
        top = r.top - h - 6;
      }
    } else {
      left = (global.innerWidth - w) / 2;
      top = (global.innerHeight - h) / 2;
    }
    pop.style.left = Math.max(8, left) + "px";
    pop.style.top = Math.max(8, top) + "px";
  }

  function chooseTag(anchorEl, current) {
    return new Promise(function (resolve) {
      closePicker();
      const cur = cleanTag(current);
      const pop = document.createElement("div");
      pop.className = "tag-picker";
      const chips = tagOptions(cur).map(function (t) {
        const on = t === cur ? " is-active" : "";
        return '<button type="button" class="tag-picker__chip' + on +
          '" data-tag="' + escHtml(t) + '">' + escHtml(t) + "</button>";
      }).join("");
      pop.innerHTML =
        '<div class="tag-picker__title">选择分类</div>' +
        '<div class="tag-picker__chips">' + chips +
          '<button type="button" class="tag-picker__chip tag-picker__chip--none' +
          (cur === "" ? " is-active" : "") + '" data-tag="">未分类</button>' +
        "</div>" +
        '<form class="tag-picker__custom">' +
          '<input type="text" class="tag-picker__input" placeholder="自定义标签…" maxlength="' +
          MAX_TAG_LEN + '" />' +
          '<button type="submit" class="tag-picker__add">添加</button>' +
        "</form>";
      document.body.appendChild(pop);
      positionPicker(pop, anchorEl);
      const input = pop.querySelector(".tag-picker__input");

      let done = false;
      function finish(value) {
        if (done) {
          return;
        }
        done = true;
        cleanup();
        resolve(value);
      }
      function onDocClick(e) {
        if (!pop.contains(e.target) && e.target !== anchorEl) {
          finish(null);
        }
      }
      function onKey(e) {
        if (e.key === "Escape") {
          finish(null);
        }
      }
      function cleanup() {
        document.removeEventListener("mousedown", onDocClick, true);
        document.removeEventListener("keydown", onKey, true);
        if (pop.parentNode) {
          pop.parentNode.removeChild(pop);
        }
      }
      activePicker = { cleanup: cleanup };

      pop.addEventListener("click", function (e) {
        const chip = e.target.closest(".tag-picker__chip");
        if (chip) {
          e.preventDefault();
          finish(cleanTag(chip.getAttribute("data-tag")));  // "未分类" → ""
        }
      });
      pop.querySelector(".tag-picker__custom").addEventListener("submit", function (e) {
        e.preventDefault();
        const v = cleanTag(input.value);
        if (v) {
          finish(v);
        }
      });
      // 下一帧再绑外部点击，避免开启这次点击立即把它关掉
      setTimeout(function () {
        document.addEventListener("mousedown", onDocClick, true);
        document.addEventListener("keydown", onKey, true);
      }, 0);
      if (input) {
        input.focus();
      }
    });
  }

  // 登录时把旧身份收藏合并到新身份（HotUser.login 调用）
  async function migrateTo(oldId, newId) {
    let list = [];
    try {
      list = await apiGet(oldId);
    } catch (_e) {
      return;
    }
    for (const item of list) {
      try {
        await apiSet(newId, item.repo, "add");
      } catch (_e) {}
    }
  }

  function subscribe(listener) {
    if (typeof listener !== "function") {
      return function () {};
    }
    listeners.add(listener);
    listener(getAll());
    return function () {
      listeners.delete(listener);
    };
  }

  global.GitHubHotFavorites = {
    ready: ready,
    refresh: refresh,
    isFavorite: isFavorite,
    getCategory: getCategory,
    getAll: getAll,
    toggle: toggle,
    retag: retag,
    setDesc: setDesc,
    subscribe: subscribe,
    migrateTo: migrateTo,
  };
})(window);
