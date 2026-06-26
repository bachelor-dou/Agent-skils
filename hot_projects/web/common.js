/* 前端共享工具：转义 / 复制 / 仓库 URL。chat.html、report.html 共用。 */
(function (global) {
  "use strict";

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  async function copyText(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement("textarea");
    textarea.value = text;
    textarea.setAttribute("readonly", "readonly");
    textarea.style.position = "fixed";
    textarea.style.opacity = "0";
    textarea.style.pointerEvents = "none";
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand("copy");
    document.body.removeChild(textarea);
  }

  function buildRepoUrl(repo) {
    return "https://github.com/" + encodeURIComponent(repo).replace(/%2F/g, "/");
  }

  // 复制按钮统一状态：切换样式类、无障碍文案，并在按钮上浮出可见的「已复制 / 复制失败」气泡。
  // report.html 与 chat.html 共用，避免两处实现各自漂移。
  function applyRepoCopyState(button, state, repo) {
    const messages = {
      idle: "复制 " + repo,
      copied: "已复制 " + repo,
      failed: "复制失败 " + repo,
    };
    const message = messages[state] || messages.idle;
    button.setAttribute("title", message);
    button.setAttribute("aria-label", message);

    button.classList.toggle("copied", state === "copied");
    button.classList.toggle("copy-failed", state === "failed");

    let badge = button.querySelector(".repo-copy-feedback");
    if (state === "copied" || state === "failed") {
      if (!badge) {
        badge = document.createElement("span");
        badge.className = "repo-copy-feedback";
        badge.setAttribute("aria-hidden", "true");
        button.appendChild(badge);
      }
      badge.textContent = state === "copied" ? "已复制" : "复制失败";
    } else if (badge) {
      badge.textContent = "";
    }
  }

  global.HotCommon = {
    escapeHtml: escapeHtml,
    copyText: copyText,
    buildRepoUrl: buildRepoUrl,
    applyRepoCopyState: applyRepoCopyState,
  };
})(window);
