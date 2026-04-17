(function setupGitHubHotFavorites(global) {
  const STORAGE_KEY = "gh-hot-favorites-v1";
  const listeners = new Set();

  function normalizeFavorite(item) {
    if (typeof item === "string") {
      const repoFromString = String(item || "").trim();
      if (!repoFromString) {
        return null;
      }
      return {
        repo: repoFromString,
        reportName: "",
        reportUrl: "",
      };
    }

    if (!item || typeof item !== "object") {
      return null;
    }

    const repo = String(item.repo || item.name || "").trim();
    if (!repo) {
      return null;
    }

    return {
      repo: repo,
      reportName: String(item.reportName || "").trim(),
      reportUrl: String(item.reportUrl || "").trim(),
    };
  }

  function normalizeFavorites(items) {
    const unique = [];
    const seen = new Map();

    (Array.isArray(items) ? items : []).forEach(function (item) {
      const favorite = normalizeFavorite(item);
      if (!favorite) {
        return;
      }

      const existingIndex = seen.get(favorite.repo);
      if (existingIndex === undefined) {
        seen.set(favorite.repo, unique.length);
        unique.push(favorite);
        return;
      }

      const existing = unique[existingIndex];
      if (!existing.reportName && favorite.reportName) {
        existing.reportName = favorite.reportName;
      }
      if (!existing.reportUrl && favorite.reportUrl) {
        existing.reportUrl = favorite.reportUrl;
      }
    });

    return unique;
  }

  function readFavorites() {
    try {
      const raw = global.localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }
      return normalizeFavorites(JSON.parse(raw));
    } catch (_error) {
      return [];
    }
  }

  function notify() {
    const repos = readFavorites();
    listeners.forEach(function (listener) {
      try {
        listener(repos);
      } catch (_error) {
      }
    });
  }

  function writeFavorites(items) {
    const favorites = normalizeFavorites(items);
    try {
      global.localStorage.setItem(STORAGE_KEY, JSON.stringify(favorites));
    } catch (_error) {
    }
    notify();
    return favorites;
  }

  function findFavorite(repo) {
    const normalized = String(repo || "").trim();
    if (!normalized) {
      return null;
    }
    return readFavorites().find(function (item) {
      return item.repo === normalized;
    }) || null;
  }

  function add(repoOrFavorite, metadata) {
    const favorite = normalizeFavorite(
      typeof repoOrFavorite === "object" && repoOrFavorite !== null
        ? repoOrFavorite
        : Object.assign({ repo: repoOrFavorite }, metadata || {})
    );

    if (!favorite) {
      return readFavorites();
    }

    const next = readFavorites().filter(function (item) {
      return item.repo !== favorite.repo;
    });
    next.unshift(favorite);
    return writeFavorites(next);
  }

  function remove(repo) {
    const normalized = String(repo || "").trim();
    if (!normalized) {
      return readFavorites();
    }
    return writeFavorites(readFavorites().filter(function (item) {
      return item.repo !== normalized;
    }));
  }

  function isFavorite(repo) {
    return !!findFavorite(repo);
  }

  function enrich(repo, metadata) {
    const normalized = String(repo || "").trim();
    if (!normalized) {
      return readFavorites();
    }

    const current = readFavorites();
    const index = current.findIndex(function (item) {
      return item.repo === normalized;
    });
    if (index < 0) {
      return current;
    }

    const updated = normalizeFavorite(Object.assign({}, current[index], metadata || {}, { repo: normalized }));
    if (!updated) {
      return current;
    }

    if (
      current[index].reportName === updated.reportName &&
      current[index].reportUrl === updated.reportUrl
    ) {
      return current;
    }

    current[index] = updated;
    return writeFavorites(current);
  }

  function toggle(repo, metadata) {
    if (isFavorite(repo)) {
      remove(repo);
      return false;
    }
    add(repo, metadata);
    return true;
  }

  function subscribe(listener) {
    if (typeof listener !== "function") {
      return function noop() {};
    }
    listeners.add(listener);
    listener(readFavorites());
    return function unsubscribe() {
      listeners.delete(listener);
    };
  }

  global.addEventListener("storage", function (event) {
    if (event.key === STORAGE_KEY) {
      notify();
    }
  });

  global.GitHubHotFavorites = {
    add: add,
    enrich: enrich,
    get: findFavorite,
    getAll: readFavorites,
    isFavorite: isFavorite,
    key: STORAGE_KEY,
    remove: remove,
    subscribe: subscribe,
    toggle: toggle,
  };
})(window);