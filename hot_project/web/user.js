/* 轻量用户身份：localStorage 存 user_id，无密码。ID 即钥匙。
   - 同浏览器：自动沿用（自动登录）
   - 换设备：自动生成匿名 anon-xxxx；手动输入同一 ID 即可跨设备同步收藏 */
(function (global) {
  "use strict";

  const KEY = "hot-user-id";
  // 存储层可接受的 ID（含系统匿名 anon- 前缀，带连字符）
  const ID_RE = /^[A-Za-z0-9_-]{3,32}$/;
  // 用户手动登录只允许大小写字母和数字
  const MANUAL_ID_RE = /^[A-Za-z0-9]{3,32}$/;

  function generateAnonId() {
    const rand = Math.random().toString(36).slice(2, 10);
    return "anon-" + rand;
  }

  function getId() {
    let id = "";
    try {
      id = global.localStorage.getItem(KEY) || "";
    } catch (_e) {
      id = "";
    }
    if (!ID_RE.test(id)) {
      id = generateAnonId();
      try {
        global.localStorage.setItem(KEY, id);
      } catch (_e) {}
    }
    return id;
  }

  function isAnonymous(id) {
    return String(id || getId()).indexOf("anon-") === 0;
  }

  function isValid(id) {
    return ID_RE.test(String(id || ""));
  }

  // 登录：迁移当前（匿名）身份下的收藏到新 ID，再切换并刷新
  async function login(newId) {
    const target = String(newId || "").trim();
    if (!MANUAL_ID_RE.test(target)) {
      throw new Error("ID 只允许大小写字母和数字，长度 3-32 位");
    }
    const oldId = getId();
    if (target !== oldId && global.GitHubHotFavorites && isAnonymous(oldId)) {
      try {
        await global.GitHubHotFavorites.migrateTo(oldId, target);
      } catch (_e) {
        // 迁移失败不阻断登录，仅记录
        console.warn("收藏迁移失败", _e);
      }
    }
    try {
      global.localStorage.setItem(KEY, target);
    } catch (_e) {}
    global.location.reload();
  }

  function logout() {
    try {
      global.localStorage.setItem(KEY, generateAnonId());
    } catch (_e) {}
    global.location.reload();
  }

  global.HotUser = {
    getId: getId,
    isAnonymous: isAnonymous,
    isValid: isValid,
    login: login,
    logout: logout,
  };

  // 聊天页状态栏登录标签（其他页无此元素时自动跳过）
  function setupLoginLabel() {
    const button = document.getElementById("user-login-button");
    const popover = document.getElementById("user-popover");
    if (!button || !popover) {
      return;
    }
    const input = document.getElementById("user-id-input");
    const confirm = document.getElementById("user-login-confirm");
    const logoutBtn = document.getElementById("user-logout");

    const id = getId();
    const anon = isAnonymous(id);
    button.textContent = anon ? "登录" : id;
    button.classList.toggle("is-logged-in", !anon);
    if (logoutBtn) {
      logoutBtn.hidden = anon;
    }

    function setOpen(open) {
      popover.hidden = !open;
      button.setAttribute("aria-expanded", open ? "true" : "false");
      if (open && input) {
        input.value = anon ? "" : id;
        input.focus();
      }
    }

    button.addEventListener("click", function (e) {
      e.stopPropagation();
      setOpen(popover.hidden);
    });
    popover.addEventListener("click", function (e) {
      e.stopPropagation();
    });
    document.addEventListener("click", function () {
      setOpen(false);
    });

    async function doLogin() {
      try {
        await login(input ? input.value : "");
      } catch (err) {
        if (input) {
          input.setCustomValidity(err.message || "无效 ID");
          input.reportValidity();
          input.setCustomValidity("");
        }
      }
    }

    if (confirm) {
      confirm.addEventListener("click", doLogin);
    }
    if (input) {
      input.addEventListener("keydown", function (e) {
        if (e.key === "Enter") {
          doLogin();
        }
      });
    }
    if (logoutBtn) {
      logoutBtn.addEventListener("click", logout);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", setupLoginLabel);
  } else {
    setupLoginLabel();
  }
})(window);
