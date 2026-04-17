(function setupChatFavoritesModule(global) {
  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildRepoUrl(repo) {
    return "https://github.com/" + encodeURIComponent(repo).replace(/%2F/g, "/");
  }

  function normalizeFavoriteEntry(item) {
    if (typeof item === "string") {
      const repoFromString = String(item || "").trim();
      return repoFromString ? { repo: repoFromString } : null;
    }

    if (!item || typeof item !== "object") {
      return null;
    }

    const repo = String(item.repo || item.name || "").trim();
    return repo ? { repo: repo } : null;
  }

  function setupChatFavoritesPanel(options) {
    const config = options || {};
    const favoritesApi = global.GitHubHotFavorites || null;
    const favoritesButton = document.getElementById(config.buttonId || "favorites-panel-button");
    const favoritesOverlay = document.getElementById(config.overlayId || "favorites-overlay");
    const favoritesClose = document.getElementById(config.closeButtonId || "favorites-close");
    const favoritesListEl = document.getElementById(config.listId || "favorites-list");

    if (!favoritesButton || !favoritesOverlay || !favoritesClose || !favoritesListEl || !favoritesApi) {
      return null;
    }

    function updateFavoritesButton(repos) {
      const count = Array.isArray(repos) ? repos.length : 0;
      const label = count > 0 ? "查看收藏项目，当前 " + count + " 个" : "查看收藏项目";

      favoritesButton.classList.toggle("is-active", count > 0);
      favoritesButton.textContent = count > 0 ? "★" : "☆";
      favoritesButton.setAttribute("aria-label", label);
      favoritesButton.setAttribute("title", label);
    }

    function renderFavorites(repos) {
      const items = (Array.isArray(repos) ? repos : [])
        .map(normalizeFavoriteEntry)
        .filter(Boolean);
      updateFavoritesButton(items);

      if (!items.length) {
        favoritesListEl.innerHTML = '<div class="empty">你还没有收藏项目。打开任意报告，点仓库标题右侧的星标就会出现在这里。</div>';
        return;
      }

      favoritesListEl.innerHTML = items.map(function (favorite) {
        const repo = favorite.repo;
        const repoLabel = escapeHtml(repo);
        const repoUrl = escapeHtml(buildRepoUrl(repo));
        return [
          '<article class="favorite-item">',
          '<a class="favorite-link" href="' + repoUrl + '" target="_blank" rel="noopener">' + repoLabel + '</a>',
          '<button type="button" class="favorite-remove-button" data-repo="' + repoLabel + '" aria-label="取消收藏 ' + repoLabel + '" title="取消收藏 ' + repoLabel + '">★</button>',
          '</article>',
        ].join("");
      }).join("");
    }

    function openFavorites() {
      if (typeof config.onBeforeOpen === "function") {
        config.onBeforeOpen();
      }
      renderFavorites(favoritesApi.getAll());
      favoritesOverlay.hidden = false;
      favoritesButton.setAttribute("aria-expanded", "true");
    }

    function closeFavorites() {
      favoritesOverlay.hidden = true;
      favoritesButton.setAttribute("aria-expanded", "false");
    }

    function isOpen() {
      return !favoritesOverlay.hidden;
    }

    favoritesApi.subscribe(renderFavorites);

    favoritesButton.addEventListener("click", openFavorites);
    favoritesButton.addEventListener("touchend", function (event) {
      event.preventDefault();
      openFavorites();
    }, { passive: false });

    favoritesClose.addEventListener("click", closeFavorites);

    favoritesOverlay.addEventListener("click", function (event) {
      if (event.target === favoritesOverlay) {
        closeFavorites();
        return;
      }

      const removeButton = event.target.closest(".favorite-remove-button");
      if (!removeButton) {
        return;
      }

      const repo = removeButton.getAttribute("data-repo") || "";
      if (!repo) {
        return;
      }

      favoritesApi.remove(repo);
    });

    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && isOpen()) {
        closeFavorites();
      }
    });

    return {
      close: closeFavorites,
      isOpen: isOpen,
    };
  }

  global.setupChatFavoritesPanel = setupChatFavoritesPanel;
})(window);