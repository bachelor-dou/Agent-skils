/* 收藏：服务端持久化（按 user_id 全局），前端内存缓存 + 乐观更新。
   对外接口 isFavorite / toggle / subscribe / getAll 保持同步语义，
   report.js 等调用方无需改动；数据在页面加载时经 ready() 预载。 */
(function (global) {
  "use strict";

  const OLD_KEY = "gh-hot-favorites-v1";
  const listeners = new Set();

  let items = new Map();          // repo -> {repo, short_desc, ...}（内存缓存，保序）
  let readyPromise = null;

  function userId() {
    return global.HotUser ? global.HotUser.getId() : "";
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

  async function apiSet(uid, repo, action) {
    const resp = await fetch("/api/favorites", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: uid,
        repo: repo,
        action: action,
        source_report: currentReport(),
      }),
    });
    if (!resp.ok) {
      throw new Error("update favorite failed");
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

  // 乐观更新：先改内存并通知，失败再回滚
  async function toggle(repo) {
    const name = String(repo || "").trim();
    if (!name) {
      return isFavorite(name);
    }
    const wasFav = items.has(name);
    const prev = items.get(name);
    const action = wasFav ? "remove" : "add";
    if (wasFav) {
      items.delete(name);
    } else {
      // 新增置顶，短描述待服务端刷新补齐
      const next = new Map([[name, { repo: name, short_desc: "" }]]);
      items.forEach(function (v, k) {
        next.set(k, v);
      });
      items = next;
    }
    notify();
    try {
      await apiSet(userId(), name, action);
    } catch (_e) {
      if (wasFav) {
        const restore = new Map([[name, prev]]);
        items.forEach(function (v, k) {
          restore.set(k, v);
        });
        items = restore;
      } else {
        items.delete(name);
      }
      notify();
    }
    return items.has(name);
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
    getAll: getAll,
    toggle: toggle,
    subscribe: subscribe,
    migrateTo: migrateTo,
  };
})(window);
