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
  let activePicker = null;        // 当前打开的标签选择浮层（同一时刻只允许一个）

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
        await migrateLegacy(uid);
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

  async function setCategory(repo, category) {
    const name = String(repo || "").trim();
    const it = items.get(name);
    if (!it) {
      return "";
    }
    const next = cleanTag(category);
    const prev = it.category || "";
    if (next === prev) {
      return prev;
    }
    it.category = next;
    notify();
    try {
      await apiSet(userId(), name, "add", next);
      broadcastChange();
    } catch (_e) {
      it.category = prev;
      notify();
    }
    return it.category || "";
  }

  // 分类改名 = 把该分类下每一项改写成新名字；改成已有名字就是合并
  async function renameCategory(oldName, newName) {
    const from = cleanTag(oldName);
    const to = cleanTag(newName);
    if (!from || !to || from === to) {
      return from;
    }
    const moved = getAll().filter(function (x) {
      return (x.category || "") === from;
    });
    moved.forEach(function (x) {
      x.category = to;
    });
    notify();
    const uid = userId();
    let failed = false;
    for (const it of moved) {
      try {
        await apiSet(uid, it.repo, "add", to);
      } catch (_e) {
        failed = true;
      }
    }
    broadcastChange();
    if (failed) {
      await refresh();  // 有失败的就以服务端为准，别留下半真半假的分组
    }
    return to;
  }

  async function retag(repo, anchorEl) {
    const name = String(repo || "").trim();
    const it = items.get(name);
    if (!it) {
      return "";
    }
    const cur = it.category || "";
    const tag = await chooseTag(anchorEl, cur, cur !== "");
    if (tag === null) {
      return it.category || "";
    }
    return setCategory(name, tag);
  }

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

  // allowNone:是否给"未分类"这颗。新收藏时它是"先收着不分类"的确认键；
  // 改分类时只有本来带分类的才需要它来清空,已经在未分类里的点它等于没点。
  function chooseTag(anchorEl, current, allowNone) {
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
      const noneChip = allowNone === false ? "" :
        '<button type="button" class="tag-picker__chip tag-picker__chip--none' +
        (cur === "" ? " is-active" : "") + '" data-tag="">未分类</button>';
      pop.innerHTML =
        '<div class="tag-picker__title">选择分类</div>' +
        '<div class="tag-picker__chips">' + chips + noneChip +
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
      setTimeout(function () {
        document.addEventListener("mousedown", onDocClick, true);
        document.addEventListener("keydown", onKey, true);
      }, 0);
      if (input) {
        input.focus();
      }
    });
  }

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
    setCategory: setCategory,
    renameCategory: renameCategory,
    setDesc: setDesc,
    subscribe: subscribe,
    migrateTo: migrateTo,
  };
})(window);
