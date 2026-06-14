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

  global.HotCommon = {
    escapeHtml: escapeHtml,
    copyText: copyText,
    buildRepoUrl: buildRepoUrl,
  };
})(window);
