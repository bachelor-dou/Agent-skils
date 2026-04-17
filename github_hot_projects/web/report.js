(function setupTocNavigation() {
  const tocOverlay = document.getElementById("toc-overlay");
  const tocTrigger = document.getElementById("toc-trigger");
  const tocHash = "#toc";

  if (!tocOverlay || !tocTrigger) {
    return;
  }

  function setOverlayVisible(visible) {
    tocOverlay.classList.toggle("active", visible);
    tocOverlay.setAttribute("aria-hidden", visible ? "false" : "true");
  }

  function syncFromHash() {
    setOverlayVisible(window.location.hash === tocHash);
  }

  tocTrigger.addEventListener("click", function (event) {
    event.preventDefault();
    if (window.location.hash !== tocHash) {
      history.pushState({ toc: true }, "", tocHash);
    }
    syncFromHash();
  });

  tocOverlay.addEventListener("click", function (event) {
    if (event.target !== tocOverlay) {
      return;
    }
    if (window.location.hash === tocHash) {
      history.replaceState(null, "", window.location.pathname + window.location.search);
    }
    syncFromHash();
  });

  tocOverlay.querySelectorAll('a[href^="#"]').forEach(function (link) {
    link.addEventListener("click", function () {
      setOverlayVisible(false);
    });
  });

  window.addEventListener("hashchange", syncFromHash);
  syncFromHash();
})();

(function setupRepoActionButtons() {
  const container = document.querySelector(".content");
  const favoritesApi = window.GitHubHotFavorites || null;
  if (!container) {
    return;
  }

  function setButtonMessage(button, message) {
    button.setAttribute("title", message);
    button.setAttribute("aria-label", message);
  }

  function setButtonState(button, state, repo) {
    const idleMessage = "复制 " + repo;
    const copiedMessage = "已复制 " + repo;
    const failedMessage = "复制失败 " + repo;

    button.classList.toggle("copied", state === "copied");
    button.classList.toggle("copy-failed", state === "failed");

    if (state === "copied") {
      setButtonMessage(button, copiedMessage);
      return;
    }
    if (state === "failed") {
      setButtonMessage(button, failedMessage);
      return;
    }
    setButtonMessage(button, idleMessage);
  }

  function setFavoriteButtonState(button, repo) {
    const favorited = favoritesApi && favoritesApi.isFavorite(repo);
    const idleMessage = favorited ? "取消收藏 " + repo : "收藏 " + repo;

    button.classList.toggle("is-favorited", !!favorited);
    button.setAttribute("title", idleMessage);
    button.setAttribute("aria-label", idleMessage);
    button.textContent = favorited ? "★" : "☆";
  }

  function createFavoriteButton(repo) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "repo-favorite-btn";
    button.setAttribute("data-repo", repo);
    setFavoriteButtonState(button, repo);
    return button;
  }

  function ensureFavoriteButtons() {
    const titleSelector = "h2";
    const repoPattern = /([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/;
    container.querySelectorAll(titleSelector).forEach(function (heading) {
      const headingText = (heading.textContent || "").trim();
      const match = headingText.match(repoPattern);
      if (!match || heading.querySelector(".repo-favorite-btn")) {
        return;
      }
      heading.appendChild(document.createTextNode(" "));
      heading.appendChild(createFavoriteButton(match[1]));
    });
  }

  function syncFavoriteButtons() {
    container.querySelectorAll(".repo-favorite-btn").forEach(function (button) {
      const repo = button.getAttribute("data-repo") || "";
      if (!repo) {
        return;
      }
      setFavoriteButtonState(button, repo);
    });
  }

  function attachTitleButtonsForLegacyReports() {
    const titleSelector = "h2";
    const repoPattern = /([A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+)/;
    container.querySelectorAll(titleSelector).forEach(function (heading) {
      const headingText = (heading.textContent || "").trim();
      const match = headingText.match(repoPattern);
      if (!match || heading.querySelector(".repo-copy-btn")) {
        return;
      }

      const button = document.createElement("button");
      button.type = "button";
      button.className = "repo-copy-btn repo-copy-btn--title";
      button.setAttribute("data-repo", match[1]);
      setButtonState(button, "idle", match[1]);
      heading.appendChild(document.createTextNode(" "));
      heading.appendChild(button);
    });
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

  attachTitleButtonsForLegacyReports();
  ensureFavoriteButtons();

  if (favoritesApi && typeof favoritesApi.subscribe === "function") {
    favoritesApi.subscribe(syncFavoriteButtons);
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
      if (!repo || !favoritesApi) {
        return;
      }
      favoritesApi.toggle(repo);
      setFavoriteButtonState(button, repo);
    });
  });
})();