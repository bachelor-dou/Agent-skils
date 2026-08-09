    const SESSION_STORAGE_KEY = "gh-hot-session-id";
    const SESSION_LAST_ACTIVE_KEY = "gh-hot-session-last-active";
    const SESSION_TTL_SECONDS = Number("__SESSION_TTL_SECONDS__") || 3600;
    const SESSION_EXPIRED_TEXT = "上一会话已过期，已自动开启新会话。可以直接开始新的提问。";
    const messagesEl = document.getElementById("messages");
    const inputEl = document.getElementById("composer-input");
    const sendButton = document.getElementById("send-button");

    let programmaticScrollUntil = 0;
    function scrollMessagesToBottom() {
      programmaticScrollUntil = performance.now() + 150;
      const prev = messagesEl.style.scrollBehavior;
      messagesEl.style.scrollBehavior = "auto";
      messagesEl.scrollTop = messagesEl.scrollHeight;
      messagesEl.style.scrollBehavior = prev;
    }

    const FOLLOW_BOTTOM_PX = 64;
    function isNearBottom() {
      return messagesEl.scrollHeight - (messagesEl.scrollTop + messagesEl.clientHeight) <= FOLLOW_BOTTOM_PX;
    }
    const endSessionButton = document.getElementById("end-session-button");
    const reportListEl = document.getElementById("report-list");
    const reportCountEl = document.getElementById("report-count");
    const transportStatusEl = document.getElementById("transport-status");
    const statusDotEl = document.getElementById("status-dot");
    const chatPanelEl = document.querySelector(".chat-panel");
    const heroEl = document.querySelector(".hero");
    const statusStripEl = document.querySelector(".status-strip");
    const composerEl = document.querySelector(".composer");
    const usageHelpButton = document.getElementById("usage-help-button");
    const usageHelpOverlay = document.getElementById("usage-help-overlay");
    const usageHelpClose = document.getElementById("usage-help-close");
    const modelButton = document.getElementById("model-button");
    const modelMenu = document.getElementById("model-menu");
    const modelCurrentEl = document.getElementById("model-current");
    const liteMenu = document.getElementById("lite-menu"); // 二级子模型菜单（无独立按钮，由主行展开）

    const CHAT_HISTORY_KEY = "gh-hot-chat-history";
    const MODEL_KEY = "gh-hot-model-id";
    const LITE_KEY = "gh-hot-lite-id";
    const MAX_CHAT_HISTORY = 100; // 聊天历史最多保留的消息数，防止 localStorage 撑爆

    const escapeHtml = window.HotCommon.escapeHtml;

  let initialSessionExpired = false;
    let sessionId = getOrCreateSessionId();

    let socket = null;
    let socketReady = false;
    let latestReports = [];
    let wsReconnectTimer = null;
    let wsReconnectDelay = 1000; // 初始重连间隔 1s，指数退避最大 30s
    const WS_MAX_RECONNECT_DELAY = 30000;
    let chatHistory = []; // {role, content, isHtml}
    let activeRequest = null;
    let availableModels = []; // [{id, label}]
    let selectedModelId = ""; // 当前选用主模型 id（硬切换）
    let availableLiteModels = []; // 跨平台共享子模型池 [{id:"平台id:模型名", label}]
    let selectedLiteId = ""; // 当前选用子模型 id；空=自动（跟随主模型平台）

    restoreMessages();
    if (initialSessionExpired) {
      renderSessionIntro(false, SESSION_EXPIRED_TEXT);
    }

    setupComposer();
    setupModelSelector();
    setupSessionActions();
    setupUsageHelp();
    setupAutoHideStatusStrip();
    connectWebSocket();
    loadReports();
    setupPanelTabs();
    setupFavorites();

    function createSessionId() {
      return `mobile-${Math.random().toString(36).slice(2, 10)}`;
    }

    function getSessionLastActive(sessionIdValue) {
      try {
        const raw = localStorage.getItem(`${SESSION_LAST_ACTIVE_KEY}-${sessionIdValue}`);
        const value = Number(raw);
        return Number.isFinite(value) ? value : null;
      } catch (error) {
        return null;
      }
    }

    function markSessionActive(sessionIdValue = sessionId) {
      try {
        localStorage.setItem(`${SESSION_LAST_ACTIVE_KEY}-${sessionIdValue}`, String(Date.now()));
      } catch (error) {
      }
    }

    function clearStoredSessionState(sessionIdValue) {
      try {
        localStorage.removeItem(CHAT_HISTORY_KEY + "-" + sessionIdValue);
        localStorage.removeItem(`${SESSION_LAST_ACTIVE_KEY}-${sessionIdValue}`);
      } catch (error) {
      }
    }

    function isSessionExpired(sessionIdValue) {
      if (!SESSION_TTL_SECONDS || SESSION_TTL_SECONDS <= 0) {
        return false;
      }

      const lastActive = getSessionLastActive(sessionIdValue);
      if (!lastActive) {
        return false;
      }

      return Date.now() - lastActive > SESSION_TTL_SECONDS * 1000;
    }

    function updateConnectionStatus(text, state) {
      // state: "connected" | "connecting" | "disconnected"
      transportStatusEl.textContent = text;
      statusDotEl.className = "status-dot" + (state ? " " + state : "");
    }

    function getOrCreateSessionId() {
      const existing = localStorage.getItem(SESSION_STORAGE_KEY);
      if (existing) {
        if (isSessionExpired(existing)) {
          initialSessionExpired = true;
          clearStoredSessionState(existing);
        } else {
          return existing;
        }
      }
      const sid = createSessionId();
      localStorage.setItem(SESSION_STORAGE_KEY, sid);
      return sid;
    }

    function refreshExpiredSession(reasonText) {
      if (!isSessionExpired(sessionId)) {
        return false;
      }

      const previousSessionId = sessionId;
      rejectActiveRequest(new Error("会话已过期"));
      closeSocket(false);
      clearStoredSessionState(previousSessionId);
      const nextSessionId = createSessionId();
      updateSessionId(nextSessionId);
      renderSessionIntro(false, reasonText || SESSION_EXPIRED_TEXT);
      connectWebSocket();
      return true;
    }

    function updateSessionId(nextSessionId) {
      sessionId = nextSessionId;
      localStorage.setItem(SESSION_STORAGE_KEY, nextSessionId);
    }

    function renderSessionIntro(keepHistory, introText) {
      if (!keepHistory) {
        messagesEl.innerHTML = "";
        chatHistory = [];
      }
      addMessage("agent", introText || "欢迎，我可以生成综合榜、新项目榜、查看 Trending，也可以输入框自定义参数（如类别、数量、增长阈值等）。", {
        asHtml: false,
        hideLabel: true,
        variant: "intro",
      });
    }

    function saveChatHistory() {
      if (chatHistory.length > MAX_CHAT_HISTORY) {
        chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
      }
      try {
        localStorage.setItem(CHAT_HISTORY_KEY + "-" + sessionId, JSON.stringify(chatHistory));
      } catch (e) {
        console.warn("localStorage 已满，聊天历史无法保存", e);
      }
    }

    function restoreMessages() {
      try {
        const saved = localStorage.getItem(CHAT_HISTORY_KEY + "-" + sessionId);
        if (saved) {
          const items = JSON.parse(saved);
          if (Array.isArray(items) && items.length > 0) {
            chatHistory = items;
            messagesEl.innerHTML = "";
            items.forEach(entry => {
              const item = document.createElement("article");
              item.className = `message ${entry.role}`;
              if (entry.role !== "system") {
                const label = document.createElement("div");
                label.className = "label";
                label.textContent = entry.role === "user" ? "You" : "Agent";
                item.appendChild(label);
              }
              const body = document.createElement("div");
              if (entry.isHtml) {
                body.className = "md-body";
                body.innerHTML = entry.content;
              } else {
                body.textContent = entry.content;
              }
              item.appendChild(body);
              if (entry.role === "user" && !entry.isHtml) {
                const resend = document.createElement("button");
                resend.type = "button";
                resend.className = "message__resend";
                resend.title = "重新发送";
                resend.setAttribute("aria-label", "重新发送");
                resend.textContent = "↻";
                resend.addEventListener("click", () => submitMessage(entry.content));
                item.appendChild(resend);
              }
              messagesEl.appendChild(item);
            });
            scrollMessagesToBottom();
            return;
          }
        }
      } catch (e) {
      }
      renderSessionIntro(false);
    }

    function clearChatHistory() {
      chatHistory = [];
      clearStoredSessionState(sessionId);
    }

    function getSelectedModel() {
      return selectedModelId || "";
    }

    function getSelectedLite() {
      return selectedLiteId || "";
    }

    let pendingMainId = ""; // 主菜单中当前展开子菜单的主模型 id

    async function setupModelSelector() {
      if (!modelButton || !modelMenu || !modelCurrentEl) {
        return;
      }
      try {
        const resp = await fetch("/api/models");
        const data = await resp.json();
        availableModels = Array.isArray(data.models) ? data.models : [];
        availableLiteModels = Array.isArray(data.lite_models) ? data.lite_models : [];
      } catch (_e) {
        availableModels = [];
        availableLiteModels = [];
      }
      if (!availableModels.length) {
        modelButton.style.display = "none";
        return;
      }

      let stored = "";
      let storedLite = "";
      try {
        stored = localStorage.getItem(MODEL_KEY) || "";
        storedLite = localStorage.getItem(LITE_KEY) || "";
      } catch (_e) {}
      selectedModelId = availableModels.some((m) => m.id === stored) ? stored : availableModels[0].id;
      selectedLiteId = availableLiteModels.some((m) => m.id === storedLite) ? storedLite : "";
      updateButtonLabel();
      closeSocket(false);
      connectWebSocket();

      document.body.appendChild(modelMenu);
      if (liteMenu) document.body.appendChild(liteMenu);

      modelButton.addEventListener("click", (e) => {
        e.stopPropagation();
        const willOpen = modelMenu.hidden;
        closeMenus();
        if (willOpen) openMainMenu();
      });
      modelMenu.addEventListener("click", (e) => e.stopPropagation());
      if (liteMenu) liteMenu.addEventListener("click", (e) => e.stopPropagation());
      document.addEventListener("click", closeMenus);
    }

    function closeMenus() {
      modelMenu.hidden = true;
      if (liteMenu) liteMenu.hidden = true;
      modelButton.setAttribute("aria-expanded", "false");
      pendingMainId = "";
    }

    function updateButtonLabel() {
      const cur = availableModels.find((m) => m.id === selectedModelId);
      const sub = availableLiteModels.find((m) => m.id === selectedLiteId);
      const mainLabel = cur ? cur.label : "模型";
      modelCurrentEl.textContent = availableLiteModels.length
        ? `${mainLabel} · ${sub ? sub.label : "自动"}`
        : mainLabel;
    }

    function openMainMenu() {
      const hasPool = availableLiteModels.length > 0;
      modelMenu.innerHTML = availableModels
        .map((m) => {
          const active = m.id === selectedModelId ? " is-active" : "";
          const tail = hasPool
            ? `<span class="model-option__arrow" aria-hidden="true">›</span>`
            : `<span class="model-option__tick">✓</span>`;
          return `<button type="button" class="model-option${active}" data-model="${escapeHtml(m.id)}">` +
            `<span>${escapeHtml(m.label)}</span>${tail}</button>`;
        })
        .join("");
      modelMenu.querySelectorAll(".model-option").forEach((row) => {
        row.addEventListener("click", (e) => {
          e.stopPropagation();
          const mid = row.getAttribute("data-model");
          if (!hasPool) {
            selectCombo(mid, "");
          } else {
            openSubMenu(mid, row);
          }
        });
      });
      modelMenu.hidden = false;
      modelButton.setAttribute("aria-expanded", "true");
      positionUp(modelMenu, modelButton);
    }

    function openSubMenu(mainId, anchorRow) {
      pendingMainId = mainId;
      const options = [{ id: "", label: "自动" }].concat(availableLiteModels);
      liteMenu.innerHTML = options
        .map((m) => {
          const active = mainId === selectedModelId && m.id === selectedLiteId ? " is-active" : "";
          return `<button type="button" class="model-option${active}" data-model="${escapeHtml(m.id)}">` +
            `<span>${escapeHtml(m.label)}</span>` +
            `<span class="model-option__tick">✓</span></button>`;
        })
        .join("");
      liteMenu.querySelectorAll(".model-option").forEach((btn) => {
        btn.addEventListener("click", (e) => {
          e.stopPropagation();
          selectCombo(pendingMainId, btn.getAttribute("data-model"));
        });
      });
      modelMenu.querySelectorAll(".model-option").forEach((r) =>
        r.classList.toggle("is-open", r.getAttribute("data-model") === mainId)
      );
      liteMenu.hidden = false;
      positionRight(liteMenu, anchorRow);
    }

    function positionUp(menu, anchor) {
      const r = anchor.getBoundingClientRect();
      const mw = menu.offsetWidth;
      menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - mw - 8)) + "px";
      menu.style.top = "auto";
      menu.style.bottom = (window.innerHeight - r.top + 6) + "px";
    }

    function positionRight(menu, anchorRow) {
      const r = anchorRow.getBoundingClientRect();
      const mw = menu.offsetWidth;
      const mh = menu.offsetHeight;
      let left = r.right + 4;
      if (left + mw > window.innerWidth - 8) left = r.left - mw - 4;
      let top = r.top;
      if (top + mh > window.innerHeight - 8) top = window.innerHeight - mh - 8;
      menu.style.left = Math.max(8, left) + "px";
      menu.style.top = Math.max(8, top) + "px";
      menu.style.bottom = "auto";
    }

    async function preflightTest(params) {
      try {
        const resp = await fetch("/api/models/test", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(params),
        });
        const data = await resp.json();
        if (data && data.ok === false) {
          return (data.unavailable || []).join("、") || "所选模型";
        }
        return "";
      } catch (_e) {
        return ""; // 预检接口本身异常不拦截使用，交给真实调用报错
      }
    }

    async function selectCombo(mainId, liteId) {
      liteId = liteId || "";
      if (mainId === selectedModelId && liteId === selectedLiteId) {
        closeMenus();
        return;
      }
      const params = {};
      if (mainId !== selectedModelId) params.model = mainId;
      if (liteId) params.lite = liteId;

      if (Object.keys(params).length) {
        const prevLabel = modelCurrentEl.textContent;
        modelCurrentEl.textContent = "测试中…";
        const wasSending = sendButton.disabled;
        setSending(true);
        let failed;
        try {
          failed = await preflightTest(params);
        } finally {
          setSending(wasSending);
        }
        if (failed) {
          modelCurrentEl.textContent = prevLabel; // 恢复原选择（会话不变，换个可用模型重发即可）
          closeMenus();
          showToast(`模型「${failed}」不可用`);
          return;
        }
      }

      selectedModelId = mainId;
      selectedLiteId = liteId;
      try {
        localStorage.setItem(MODEL_KEY, mainId);
        localStorage.setItem(LITE_KEY, liteId);
      } catch (_e) {}
      updateButtonLabel();
      closeMenus();
      closeSocket(false);
      connectWebSocket();
    }

    let toastTimer = null;
    function showToast(text) {
      let el = document.getElementById("app-toast");
      if (!el) {
        el = document.createElement("div");
        el.id = "app-toast";
        el.className = "app-toast";
        document.body.appendChild(el);
      }
      el.textContent = text;
      el.classList.add("is-visible");
      if (toastTimer) clearTimeout(toastTimer);
      toastTimer = setTimeout(() => el.classList.remove("is-visible"), 3000);
    }

    function setupComposer() {
      inputEl.addEventListener("input", autoResize);
      inputEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          sendMessage();
        }
      });
      sendButton.addEventListener("click", sendMessage);
    }

    function setupSessionActions() {
      endSessionButton.addEventListener("click", endSession);
    }

    function setupUsageHelp() {
      if (!usageHelpButton || !usageHelpOverlay || !usageHelpClose) {
        return;
      }

      function openUsageHelp() {
        usageHelpOverlay.hidden = false;
        usageHelpButton.setAttribute("aria-expanded", "true");
      }

      function closeUsageHelp() {
        usageHelpOverlay.hidden = true;
        usageHelpButton.setAttribute("aria-expanded", "false");
      }

      usageHelpButton.addEventListener("click", openUsageHelp);
      usageHelpButton.addEventListener("touchend", (event) => {
        event.preventDefault();
        openUsageHelp();
      }, { passive: false });

      usageHelpClose.addEventListener("click", closeUsageHelp);
      usageHelpOverlay.addEventListener("click", (event) => {
        if (event.target === usageHelpOverlay) {
          closeUsageHelp();
        }
      });

      document.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && !usageHelpOverlay.hidden) {
          closeUsageHelp();
        }
      });
    }

    function setupAutoHideStatusStrip() {
      if (!chatPanelEl || !heroEl || !statusStripEl || !messagesEl || !composerEl) {
        return;
      }

      const REVEAL_DELAY_MS = 320;
      let revealTimer = null;
      let chromeHidden = false;
      let isTouchHolding = false;
      let lastScrollTop = messagesEl.scrollTop;
      let noHideUntil = 0;

      function syncHeights() {
        heroEl.style.setProperty("--chrome-height", `${heroEl.offsetHeight}px`);
        statusStripEl.style.setProperty("--chrome-height", `${statusStripEl.offsetHeight}px`);
        const composerHeight = `${composerEl.offsetHeight}px`;
        composerEl.style.setProperty("--chrome-height", composerHeight);
        chatPanelEl.style.setProperty("--composer-space", composerHeight);
      }

      function clearRevealTimer() {
        if (revealTimer) {
          window.clearTimeout(revealTimer);
          revealTimer = null;
        }
      }

      function showChatChrome() {
        clearRevealTimer();
        syncHeights();
        if (!chromeHidden) {
          return;
        }
        heroEl.classList.remove("is-reading-hidden");
        statusStripEl.classList.remove("is-reading-hidden");
        composerEl.classList.remove("is-reading-hidden");
        chromeHidden = false;
        noHideUntil = performance.now() + 500;  // 显示后短暂禁收起，避免布局变化引发的横跳
      }

      function hideChatChrome() {
        if (chromeHidden || document.activeElement === inputEl || performance.now() < noHideUntil) {
          return;
        }
        syncHeights();
        heroEl.classList.add("is-reading-hidden");
        statusStripEl.classList.add("is-reading-hidden");
        composerEl.classList.add("is-reading-hidden");
        chromeHidden = true;
      }

      function isMessagesScrollable() {
        return (messagesEl.scrollHeight - messagesEl.clientHeight) > 8;
      }

      function scheduleReveal() {
        clearRevealTimer();
        revealTimer = window.setTimeout(() => {
          showChatChrome();
        }, REVEAL_DELAY_MS);
      }

      function handleReadingScroll() {
        if (performance.now() < programmaticScrollUntil) {
          lastScrollTop = messagesEl.scrollTop;
          return;
        }
        if (document.activeElement === inputEl || !isMessagesScrollable()) {
          showChatChrome();
          return;
        }

        const currentScrollTop = messagesEl.scrollTop;
        const delta = currentScrollTop - lastScrollTop;
        lastScrollTop = currentScrollTop;

        const atTop = currentScrollTop <= 4;
        const atBottom =
          messagesEl.scrollHeight - (currentScrollTop + messagesEl.clientHeight) <= 8;

        if (atTop || atBottom) {
          showChatChrome();
        } else if (Math.abs(delta) > 2) {
          hideChatChrome();
        }
      }

      function handleTouchStart() {
        isTouchHolding = true;
        if (document.activeElement === inputEl) {
          showChatChrome();
          return;
        }
        if (!isMessagesScrollable()) {
          showChatChrome();
          return;
        }
        clearRevealTimer();
        hideChatChrome();
      }

      function handleTouchEnd() {
        isTouchHolding = false;
        if (document.activeElement === inputEl || !isMessagesScrollable()) {
          showChatChrome();
        }
      }

      messagesEl.addEventListener("scroll", handleReadingScroll, { passive: true });
      messagesEl.addEventListener("touchstart", handleTouchStart, { passive: true });
      messagesEl.addEventListener("touchmove", handleReadingScroll, { passive: true });
      messagesEl.addEventListener("touchend", handleTouchEnd, { passive: true });
      messagesEl.addEventListener("touchcancel", handleTouchEnd, { passive: true });
      window.addEventListener("resize", showChatChrome);
      statusStripEl.addEventListener("mouseenter", showChatChrome);
      statusStripEl.addEventListener("focusin", showChatChrome);
      composerEl.addEventListener("mouseenter", showChatChrome);
      composerEl.addEventListener("focusin", showChatChrome);
      inputEl.addEventListener("focus", showChatChrome);
      inputEl.addEventListener("blur", () => {
        if (chromeHidden) {
          scheduleReveal();
        }
      });

      showChatChrome();
    }

    function autoResize() {
      inputEl.style.height = "auto";
      inputEl.style.height = `${Math.min(inputEl.scrollHeight, 150)}px`;
    }

    function setSending(isSending) {
      sendButton.disabled = isSending;
      sendButton.setAttribute("aria-busy", isSending ? "true" : "false");
      sendButton.title = isSending ? "发送中…" : "发送";
    }

    function resolveActiveRequest(reply) {
      if (!activeRequest) {
        return false;
      }

      const currentRequest = activeRequest;
      activeRequest = null;
      if (currentRequest.socket) {
        currentRequest.socket._hasActiveRequest = false;
      }
      if (socket) {
        socket._hasActiveRequest = false;
      }
      currentRequest.resolve(reply);
      return true;
    }

    function rejectActiveRequest(error) {
      if (!activeRequest) {
        return false;
      }

      const currentRequest = activeRequest;
      activeRequest = null;
      if (currentRequest.socket) {
        currentRequest.socket._hasActiveRequest = false;
      }
      if (socket) {
        socket._hasActiveRequest = false;
      }
      currentRequest.reject(error instanceof Error ? error : new Error(String(error)));
      return true;
    }

    function buildIntentSummaryFromPayload(payload) {
      if (!payload || typeof payload !== "object") {
        return "";
      }

      const ambiguousFields = Array.isArray(payload.ambiguous_fields)
        ? payload.ambiguous_fields.filter((item) => typeof item === "string" && item.trim().length > 0)
        : [];
      if (ambiguousFields.length > 0) {
        const clarificationText = typeof payload.clarification_text_zh === "string"
          ? payload.clarification_text_zh.trim()
          : "";
        if (clarificationText) {
          return clarificationText;
        }

        const confirmationText = typeof payload.confirmation_text_zh === "string"
          ? payload.confirmation_text_zh.trim()
          : "";
        if (confirmationText) {
          return confirmationText;
        }

        return `我还需要确认这些点：${ambiguousFields.join("；")}。请直接补充，我会继续执行。`;
      }

      const fragments = [];
      const intentLabel = typeof payload.intent_label_zh === "string" ? payload.intent_label_zh.trim() : "";
      if (intentLabel) {
        fragments.push(intentLabel);
      }

      const params = payload.specified_params && typeof payload.specified_params === "object"
        ? payload.specified_params
        : null;

      if (params) {
        if (Array.isArray(params.categories) && params.categories.length > 0) {
          fragments.push(`关注方向为${params.categories.join("、")}`);
        }
        if (Number.isFinite(params.growth_calc_days)) {
          fragments.push(`统计近${params.growth_calc_days}天增长`);
        }
        if (Number.isFinite(params.days_since_created)) {
          fragments.push(`只看近${params.days_since_created}天内创建的项目`);
        }
        if (Number.isFinite(params.top_n)) {
          fragments.push(`返回前${params.top_n}名`);
        }
      }

      if (payload.report_requested === true) {
        fragments.push("结果完成后生成报告");
      }

      if (!fragments.length) {
        return "";
      }

      return `收到！我理解为：${fragments.join("，")}。如果要调整参数请直接告诉我，我会继续执行。`;
    }

    function normalizeAgentReplyText(rawPayload) {
      const rawText = typeof rawPayload === "string"
        ? rawPayload.trim()
        : String(rawPayload == null ? "" : rawPayload).trim();

      if (!rawText) {
        return "服务端未返回内容";
      }

      let parsed = null;
      try {
        parsed = JSON.parse(rawText);
      } catch (error) {
        return rawText;
      }

      if (!parsed || typeof parsed !== "object") {
        return rawText;
      }

      const nestedReply = typeof parsed.reply === "string" ? parsed.reply.trim() : "";
      if (nestedReply) {
        return normalizeAgentReplyText(nestedReply);
      }

      const clarificationText = typeof parsed.clarification_text_zh === "string"
        ? parsed.clarification_text_zh.trim()
        : "";
      if (clarificationText) {
        return clarificationText;
      }

      const confirmationText = typeof parsed.confirmation_text_zh === "string"
        ? parsed.confirmation_text_zh.trim()
        : "";
      if (confirmationText) {
        return confirmationText;
      }

      const messageText = typeof parsed.message === "string" ? parsed.message.trim() : "";
      if (messageText) {
        return messageText;
      }

      const contentText = typeof parsed.content === "string" ? parsed.content.trim() : "";
      if (contentText) {
        return contentText;
      }

      const errorText = typeof parsed.error === "string" ? parsed.error.trim() : "";
      if (errorText) {
        return `请求失败：${errorText}`;
      }

      const summaryText = buildIntentSummaryFromPayload(parsed);
      if (summaryText) {
        return summaryText;
      }

      return rawText;
    }

    function addMessage(role, text, options = {}) {
      const item = document.createElement("article");
      item.className = `message ${role}`;
      if (options.variant) {
        item.classList.add(`message--${options.variant}`);
      }

      if (role !== "system" && !options.hideLabel) {
        const label = document.createElement("div");
        label.className = "label";
        label.textContent = role === "user" ? "You" : "Agent";
        item.appendChild(label);
      }

      const body = document.createElement("div");
      if (options.asHtml) {
        body.innerHTML = text;
      } else {
        body.textContent = text;
      }
      item.appendChild(body);

      if (role === "user") {
        const resend = document.createElement("button");
        resend.type = "button";
        resend.className = "message__resend";
        resend.title = "重新发送";
        resend.setAttribute("aria-label", "重新发送");
        resend.textContent = "↻";
        resend.addEventListener("click", () => submitMessage(text));
        item.appendChild(resend);
      }

      messagesEl.appendChild(item);
      scrollMessagesToBottom();

      if (!options.typing) {
        chatHistory.push({ role, content: text, isHtml: !!options.asHtml });
        saveChatHistory();
      }

      return item;
    }

    function addTypingIndicator() {
      return addMessage("agent", "正在处理", { asHtml: false, typing: true });
    }

    function addPendingProgress() {
      const item = addMessage("agent", "", { asHtml: true, typing: true });
      const body = item.lastElementChild;
      body.className = "md-body";
      body.innerHTML =
        '<div class="agent-progress" data-state="indeterminate">' +
          '<div class="agent-progress__head">' +
            '<span class="agent-progress__label">思考中…</span>' +
            '<span class="agent-progress__pct"></span>' +
          '</div>' +
          '<div class="agent-progress__track"><div class="agent-progress__fill"></div></div>' +
        '</div>';
      const wrap = body.querySelector(".agent-progress");
      const fill = body.querySelector(".agent-progress__fill");
      const labelEl = body.querySelector(".agent-progress__label");
      const pctEl = body.querySelector(".agent-progress__pct");

      const startedAt = Date.now();
      let elapsedTimer = window.setInterval(function () {
        if (wrap.getAttribute("data-state") === "indeterminate") {
          pctEl.textContent = Math.floor((Date.now() - startedAt) / 1000) + "s";
        }
      }, 1000);
      function stopTimer() {
        if (elapsedTimer !== null) {
          window.clearInterval(elapsedTimer);
          elapsedTimer = null;
        }
      }

      function onProgress(percent, label) {
        if (typeof percent === "number" && isFinite(percent)) {
          const p = Math.max(0, Math.min(100, Math.round(percent)));
          wrap.setAttribute("data-state", "determinate");
          fill.style.width = p + "%";
          pctEl.textContent = p + "%";
        }
        if (label) {
          labelEl.textContent = label;
        }
      }

      let streaming = false;
      let acc = "";
      let renderTimer = null;
      function flush() {
        renderTimer = null;
        const follow = isNearBottom();
        body.innerHTML = enhanceReply(acc);
        if (follow) {
          scrollMessagesToBottom();
        }
      }
      function streamedText() {
        if (!streaming) {
          return null;
        }
        if (renderTimer !== null) {
          window.clearTimeout(renderTimer);
          flush();
        }
        return acc;
      }
      function onDelta(text, reset) {
        stopTimer();  // 正文开始流出：进度条即将被替换，停掉计时
        if (reset) {
          acc = "";
          streaming = true;
          body.innerHTML = "";
        }
        if (!text) {
          return;
        }
        if (!streaming) {
          streaming = true;
          body.innerHTML = "";
        }
        acc += text;
        if (renderTimer === null) {
          renderTimer = window.setTimeout(flush, 40);  // 重渲节流：40ms≈25fps，逐 token 更顺滑
        }
      }

      return { item, onProgress, onDelta, stop: stopTimer, streamedText };
    }

    function renderUnsolicited(displayText) {
      addMessage("agent", "", { asHtml: true, typing: false });
      const msg = messagesEl.lastElementChild;
      const body = msg.lastElementChild;
      body.className = "md-body";
      body.innerHTML = enhanceReply(displayText);
      chatHistory.push({ role: "agent", content: body.innerHTML, isHtml: true });
      saveChatHistory();
    }

    function updateTypingIndicator(el, finalText, streamedText) {
      const body = el.lastElementChild;
      body.className = "md-body";
      let html;
      if (streamedText != null && streamedText === finalText && body.innerHTML) {
        html = body.innerHTML;
      } else {
        html = enhanceReply(finalText);
        body.innerHTML = html;
      }
      body.classList.remove("typing");

      chatHistory.push({ role: "agent", content: html, isHtml: true });
      saveChatHistory();
    }

    async function sendMessage() {
      const message = inputEl.value.trim();
      if (!message) {
        return;
      }
      inputEl.value = "";
      autoResize();
      await submitMessage(message);
    }

    async function submitMessage(message) {
      if (!message) {
        return;
      }

      refreshExpiredSession("上一会话已过期，已自动开启新会话。本次提问将作为新会话开始。");

      addMessage("user", message, { asHtml: false });
      setSending(true);

      const pendingObj = addPendingProgress();
      const pending = pendingObj.item;

      try {
        const reply = socketReady ? await sendViaWebSocket(message, pendingObj.onProgress, pendingObj.onDelta) : await sendViaHttp(message);
        if (typeof pendingObj.stop === "function") {
          pendingObj.stop();  // 兜底：无流式增量的回复也要停掉等待计时
        }
        const streamed = typeof pendingObj.streamedText === "function" ? pendingObj.streamedText() : null;
        updateTypingIndicator(pending, reply, streamed);
        await loadReports();
        if (window.GitHubHotFavorites) {
          window.GitHubHotFavorites.refresh();
        }
      } catch (error) {
        pending.remove();
        addMessage("system", `请求失败：${error.message}`, { asHtml: false });
      } finally {
        if (typeof pendingObj.stop === "function") {
          pendingObj.stop();  // 无论成功/失败/异常，都不留下空转的计时器
        }
        setSending(false);
      }
    }

    async function endSession() {
      const previousSessionId = sessionId;
      endSessionButton.disabled = true;
      rejectActiveRequest(new Error("会话已重置"));
      closeSocket(false);
      updateConnectionStatus("切换中", "connecting");

      try {
        const response = await fetch(`/api/sessions/${encodeURIComponent(previousSessionId)}`, {
          method: "DELETE",
        });
        if (!response.ok && response.status !== 404) {
          throw new Error(`HTTP ${response.status}`);
        }
      } catch (error) {
        addMessage("system", `结束旧会话时出现提示：${error.message}` , { asHtml: false });
      }

      clearStoredSessionState(previousSessionId);
      chatHistory = [];
      const nextSessionId = createSessionId();
      updateSessionId(nextSessionId);
      renderSessionIntro(false, "开始新的提问吧！！");
      connectWebSocket();
      setSending(false);
      endSessionButton.disabled = false;
      inputEl.focus();
    }

    function closeSocket(updateStatus) {
      const currentSocket = socket;
      socket = null;
      socketReady = false;
      if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = null;
      }
      if (updateStatus) {
        updateConnectionStatus("普通模式", "disconnected");
      }
      if (currentSocket && (currentSocket.readyState === WebSocket.OPEN || currentSocket.readyState === WebSocket.CONNECTING)) {
        currentSocket.close(1000, "session-reset");
      }
    }

    function scheduleReconnect() {
      if (wsReconnectTimer) return;
      wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectWebSocket();
      }, wsReconnectDelay);
      wsReconnectDelay = Math.min(wsReconnectDelay * 2, WS_MAX_RECONNECT_DELAY);
    }

    function connectWebSocket() {
      if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
        return;
      }
      socket = null;
      socketReady = false;
      updateConnectionStatus("连接中", "connecting");
      const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const currentSessionId = sessionId;
      const uid = (window.HotUser && window.HotUser.getId()) || "";
      const params = new URLSearchParams();
      if (uid) params.set("user_id", uid);
      if (getSelectedModel()) params.set("model", getSelectedModel());
      if (getSelectedLite()) params.set("lite", getSelectedLite());
      const qs = params.toString();
      const url = `${protocol}//${window.location.host}/ws/chat/${encodeURIComponent(currentSessionId)}`
        + (qs ? `?${qs}` : "");

      try {
        socket = new WebSocket(url);
      } catch (error) {
        socketReady = false;
        updateConnectionStatus("普通模式", "disconnected");
        scheduleReconnect();
        return;
      }

      const connectTimeout = setTimeout(() => {
        if (socket && socket.readyState === WebSocket.CONNECTING) {
          socket.close();
          socketReady = false;
          updateConnectionStatus("普通模式", "disconnected");
          scheduleReconnect();
        }
      }, 30000);

      socket.addEventListener("open", () => {
        clearTimeout(connectTimeout);
        if (currentSessionId !== sessionId) {
          return;
        }
        socketReady = true;
        wsReconnectDelay = 1000; // 连接成功，重置退避
        updateConnectionStatus("已连接", "connected");
      });

      socket.addEventListener("message", (event) => {
        if (currentSessionId !== sessionId) return;
        markSessionActive(currentSessionId);

        let envelope = null;
        try {
          envelope = JSON.parse(event.data);
        } catch (_error) {
          envelope = null;
        }

        if (envelope && typeof envelope === "object" && typeof envelope.type === "string") {
          if (envelope.type === "heartbeat") {
            return; // 保活帧，忽略
          }
          if (envelope.type === "progress") {
            if (activeRequest && typeof activeRequest.onProgress === "function") {
              activeRequest.onProgress(envelope.percent, envelope.label);
            }
            return;
          }
          if (envelope.type === "delta") {
            if (activeRequest && typeof activeRequest.onDelta === "function") {
              activeRequest.onDelta(envelope.text || "", !!envelope.reset);
            }
            return;
          }
          if (envelope.type === "reply") {
            const replyText = normalizeAgentReplyText(envelope.reply || "");
            if (resolveActiveRequest(replyText)) return;
            renderUnsolicited(replyText);
            return;
          }
          if (envelope.type === "error") {
            const errMessage = envelope.error || "未知错误";
            if (rejectActiveRequest(new Error(errMessage))) return;
            renderUnsolicited("请求失败：" + errMessage);
            return;
          }
        }

        const normalizedText = normalizeAgentReplyText(event.data);
        if (resolveActiveRequest(normalizedText)) return;
        renderUnsolicited(normalizedText);
      });

      socket.addEventListener("close", (event) => {
        clearTimeout(connectTimeout);
        if (currentSessionId !== sessionId) {
          return;
        }
        socketReady = false;
        updateConnectionStatus("普通模式", "disconnected");
        if (event.code !== 1000) {
          scheduleReconnect();
        }
      });

      socket.addEventListener("error", () => {
        clearTimeout(connectTimeout);
        if (currentSessionId !== sessionId) {
          return;
        }
        socketReady = false;
        updateConnectionStatus("普通模式", "disconnected");
        scheduleReconnect();
      });
    }

    function sendViaWebSocket(message, onProgress, onDelta) {
      return new Promise((resolve, reject) => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
          sendViaHttp(message).then(resolve).catch(reject);
          return;
        }

        if (activeRequest) {
          reject(new Error("上一条消息仍在处理中"));
          return;
        }

        const requestSocket = socket;
        activeRequest = { resolve, reject, socket: requestSocket, onProgress, onDelta };
        requestSocket._hasActiveRequest = true;

        try {
          requestSocket.send(message);
        } catch (error) {
          rejectActiveRequest(error);
        }
      });
    }

    async function sendViaHttp(message) {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message, user_id: (window.HotUser && window.HotUser.getId()) || "", model: getSelectedModel(), lite: getSelectedLite() }),
      });

      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const errorPayload = await response.json();
          detail = errorPayload.detail || detail;
        } catch (error) {
          detail = `HTTP ${response.status}`;
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      markSessionActive(sessionId);
      return normalizeAgentReplyText(payload.reply || "");
    }

    const reportsRenderedListeners = [];
    function onReportsRendered(callback) {
      if (typeof callback === "function") {
        reportsRenderedListeners.push(callback);
      }
    }

    async function loadReports() {
      const prevCount = latestReports.length;
      reportListEl.innerHTML = '<div class="empty">正在读取报告列表...</div>';

      try {
        const response = await fetch("/api/reports");
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        latestReports = Array.isArray(payload.reports) ? payload.reports : [];
        reportCountEl.textContent = String(latestReports.length);
        renderReports(latestReports);
        reportsRenderedListeners.forEach((cb) => {
          try {
            cb(prevCount, latestReports.length);
          } catch (_error) {
          }
        });
      } catch (error) {
        reportListEl.innerHTML = `<div class="empty">读取报告失败：${escapeHtml(error.message)}</div>`;
      }
    }

    function renderReports(reports) {
      if (!reports.length) {
        reportListEl.innerHTML = '<div class="empty">当前还没有日报。你可以点击"生成综合热榜"看看效果。</div>';
        return;
      }

      reportListEl.innerHTML = reports.map((report) => {
        return `
          <div class="report-card-wrapper">
            <a class="report-card" href="${getReportHtmlUrl(report.name)}" target="_blank" rel="noopener">
              <strong>${escapeHtml(report.name)}</strong>
              <div class="report-meta">
                <span>${formatReportTime(report.modified_at)}</span>
                <span>${formatSize(report.size)}</span>
              </div>
            </a>
            <button type="button" class="report-delete-btn" data-report="${escapeHtml(report.name)}" title="删除报告" aria-label="删除报告">
              <svg viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M6 6H14M6 6V14C6 15.1046 6.89543 16 8 16H12C13.1046 16 14 15.1046 14 14V6M6 6H4M14 6H16M9 9V12M11 9V12" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
              </svg>
            </button>
          </div>
        `;
      }).join("");

      reportListEl.querySelectorAll(".report-delete-btn").forEach(function(btn) {
        btn.addEventListener("click", async function(e) {
          e.stopPropagation();
          e.preventDefault();
          const reportName = btn.getAttribute("data-report");
          if (!reportName) return;

          const confirmed = window.confirm("确定要删除报告 " + reportName + " 吗？\n删除后将无法恢复。");
          if (!confirmed) return;

          btn.disabled = true;
          try {
            const resp = await fetch("/api/reports/" + encodeURIComponent(reportName), { method: "DELETE" });
            if (!resp.ok) {
              const data = await resp.json().catch(() => ({}));
              throw new Error(data.detail || "删除失败");
            }
            loadReports();
          } catch(err) {
            window.alert("删除失败: " + err.message);
            btn.disabled = false;
          }
        });
      });
    }

    function setupPanelTabs() {
      const tabs = Array.prototype.slice.call(document.querySelectorAll(".panel-tab"));
      if (!tabs.length) {
        return;
      }
      const ACTIVE_PANE_KEY = "gh-hot-active-pane";

      function activate(pane, persist) {
        let matched = false;
        tabs.forEach(function (t) {
          const on = t.getAttribute("data-pane") === pane;
          matched = matched || on;
          t.classList.toggle("is-active", on);
          t.setAttribute("aria-selected", String(on));
          const el = document.getElementById(t.getAttribute("data-pane"));
          if (el) {
            el.classList.toggle("is-active", on);
          }
        });
        if (persist && matched) {
          try {
            window.localStorage.setItem(ACTIVE_PANE_KEY, pane);
          } catch (_e) {}
        }
      }

      tabs.forEach(function (tab) {
        tab.addEventListener("click", function () {
          activate(tab.getAttribute("data-pane"), true);
        });
      });

      let saved = "";
      try {
        saved = window.localStorage.getItem(ACTIVE_PANE_KEY) || "";
      } catch (_e) {}
      if (saved && document.getElementById(saved)) {
        activate(saved, false);
      }
    }

    function setupFavorites() {
      const favList = document.getElementById("fav-list");
      const favCount = document.getElementById("fav-count");
      if (!favList || !favCount) {
        return;
      }

      const expanded = new Set();
      favList.addEventListener("toggle", function (e) {
        const key = e.target.getAttribute ? e.target.getAttribute("data-group") : null;
        if (key === null) {
          return;
        }
        if (e.target.open) {
          expanded.add(key);
        } else {
          expanded.delete(key);
        }
      }, true);

      // ponytail: 拖到边缘不自动滚动 —— 分组默认收起，标题行都挤在一屏内够得着；
      // 哪天分类多到一屏放不下，再补边缘自动滚。
      const DRAG_SLOP = 6;        // 位移不到这个数就算点击，交给标签浮层
      let drag = null;
      let dropTarget = null;
      let dragEndedAt = 0;

      function markDropTarget(group) {
        if (dropTarget === group) {
          return;
        }
        if (dropTarget) {
          dropTarget.classList.remove("is-drop-target");
        }
        dropTarget = group;
        if (dropTarget) {
          dropTarget.classList.add("is-drop-target");
        }
      }

      function groupUnder(x, y) {
        const el = document.elementFromPoint(x, y);
        return el && el.closest ? el.closest(".fav-group") : null;
      }

      function moveGhost(ghost, x, y) {
        ghost.style.transform =
          `translate(${x - ghost.offsetWidth / 2}px, ${y - ghost.offsetHeight / 2}px)`;
      }

      function finishDrag(commit) {
        if (!drag) {
          return;
        }
        const d = drag;
        const group = dropTarget;
        drag = null;
        markDropTarget(null);
        if (!d.moved) {
          return;                     // 只是点了一下 → 让 click 去开浮层
        }
        d.ghost.remove();
        d.item.classList.remove("is-dragging");
        dragEndedAt = Date.now();     // 松手后浏览器还会补一个 click，得挡掉
        const key = group ? group.getAttribute("data-group") : null;
        const api = window.GitHubHotFavorites;
        if (commit && api && key !== null) {
          api.setCategory(d.repo, key === UNCATEGORIZED ? "" : key);
        }
      }

      // 整块条目都是拖拽把手；按钮/输入框除外，点击（位移 < SLOP）不受影响
      favList.addEventListener("pointerdown", function (e) {
        const item = e.target.closest(".fav-item");
        if (!item || e.target.closest("button, input") ||
            (e.pointerType === "mouse" && e.button !== 0)) {
          return;
        }
        drag = {
          repo: item.getAttribute("data-repo"), item: item,
          pointerId: e.pointerId, x: e.clientX, y: e.clientY,
          moved: false, ghost: null,
        };
        // 捕获推迟到真的拖起来那一刻：按下就捕获会让 click 派发到 .fav-item 上，
        // 概要那种"点子元素进编辑"的分支就永远匹配不到了。
      });

      favList.addEventListener("dragstart", function (e) {
        e.preventDefault();
      });

      // 建新分类的入口：右键（触屏长按）条目弹标签浮层；已有分类之间只用拖
      favList.addEventListener("contextmenu", function (e) {
        const item = e.target.closest(".fav-item");
        if (!item || e.target.closest("a")) {
          return;  // 链接上保留浏览器原生菜单
        }
        e.preventDefault();
        finishDrag(false);
        const api = window.GitHubHotFavorites;
        if (api) {
          api.retag(item.getAttribute("data-repo"), item);
        }
      });

      favList.addEventListener("pointermove", function (e) {
        if (!drag || e.pointerId !== drag.pointerId) {
          return;
        }
        if (!drag.moved) {
          if (Math.abs(e.clientX - drag.x) + Math.abs(e.clientY - drag.y) < DRAG_SLOP) {
            return;
          }
          drag.moved = true;
          drag.item.setPointerCapture(e.pointerId);  // 拖出列表也能收到后续事件
          drag.ghost = drag.item.cloneNode(true);
          drag.ghost.className = "fav-item fav-ghost";
          drag.ghost.style.width = drag.item.offsetWidth + "px";
          document.body.appendChild(drag.ghost);
          drag.item.classList.add("is-dragging");
        }
        moveGhost(drag.ghost, e.clientX, e.clientY);
        markDropTarget(groupUnder(e.clientX, e.clientY));
      });

      favList.addEventListener("pointerup", function () {
        finishDrag(true);
      });
      favList.addEventListener("pointercancel", function () {
        finishDrag(false);
      });

      favList.addEventListener("click", function (e) {
        const api = window.GitHubHotFavorites;
        if (!api || Date.now() - dragEndedAt < 400) {
          e.preventDefault();  // 拖完松手浏览器会补 click，别让链接跳走
          return;
        }
        const renameBtn = e.target.closest(".fav-group__rename");
        if (renameBtn) {
          e.preventDefault();  // 在 <summary> 里，默认会连带折叠分组
          startRenameGroup(renameBtn);
          return;
        }
        const copyBtn = e.target.closest(".fav-item__copy");
        if (copyBtn) {
          const repo = copyBtn.getAttribute("data-repo");
          if (repo) {
            window.HotCommon.copyText(repo).then(function () {
              copyBtn.classList.add("is-copied");
              copyBtn.textContent = "✓";
              window.setTimeout(function () {
                copyBtn.classList.remove("is-copied");
                copyBtn.textContent = "⧉";
              }, 1200);
            }).catch(function () {});
          }
          return;
        }
        const rmBtn = e.target.closest(".fav-item__remove");
        if (rmBtn) {
          const repo = rmBtn.getAttribute("data-repo");
          if (repo) {
            api.toggle(repo);
          }
          return;
        }
        const descEl = e.target.closest(".fav-item__desc");
        if (descEl) {
          startEditDesc(descEl);
        }
      });

      function startEditDesc(descEl) {
        const repo = descEl.getAttribute("data-repo");
        if (!repo || descEl.dataset.editing === "1") {
          return;
        }
        descEl.dataset.editing = "1";
        const cur = descEl.classList.contains("fav-item__desc--empty")
          ? "" : descEl.textContent.trim();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "fav-item__desc-input";
        input.value = cur;
        input.maxLength = 60;
        input.placeholder = "一句话概要…";
        descEl.replaceWith(input);
        input.focus();
        input.select();
        let done = false;
        function finish(save) {
          if (done) {
            return;
          }
          done = true;
          const api = window.GitHubHotFavorites;
          if (save && api) {
            api.setDesc(repo, input.value);  // 改动则乐观更新
          }
          render(api ? api.getAll() : []);  // 始终重建，替换掉 input（含未改动/取消）
        }
        input.addEventListener("keydown", function (ev) {
          if (ev.key === "Enter") {
            ev.preventDefault();
            finish(true);
          } else if (ev.key === "Escape") {
            ev.preventDefault();
            finish(false);
          }
        });
        input.addEventListener("blur", function () {
          finish(true);
        });
      }

      function startRenameGroup(btn) {
        const nameEl = btn.parentNode.querySelector(".fav-group__name");
        if (!nameEl) {
          return;
        }
        const old = nameEl.textContent.trim();
        const input = document.createElement("input");
        input.type = "text";
        input.className = "fav-group__rename-input";
        input.value = old;
        input.maxLength = 20;
        nameEl.replaceWith(input);
        input.focus();
        input.select();
        let done = false;
        function finish(save) {
          if (done) {
            return;
          }
          done = true;
          const api = window.GitHubHotFavorites;
          const next = input.value.trim();
          if (save && api && next && next !== old) {
            expanded.delete(old);
            expanded.add(next);          // 改完保持展开；重名即合并到那一组
            api.renameCategory(old, next);
          } else {
            render(api ? api.getAll() : []);
          }
        }
        // 输入框长在 <summary> 里，点击和回车都会被 details 抢去折叠
        input.addEventListener("click", function (ev) {
          ev.preventDefault();
        });
        input.addEventListener("keydown", function (ev) {
          ev.stopPropagation();
          if (ev.key === "Enter") {
            ev.preventDefault();
            finish(true);
          } else if (ev.key === "Escape") {
            ev.preventDefault();
            finish(false);
          }
        });
        input.addEventListener("blur", function () {
          finish(true);
        });
      }

      const UNCATEGORIZED = "未分类";

      function renderItem(item) {
        const repo = escapeHtml(item.repo);
        const desc = escapeHtml(item.short_desc || "");
        const descRow = desc
          ? `<span class="fav-item__desc" data-repo="${repo}" title="${desc}&#10;（点击编辑）">${desc}</span>`
          : `<span class="fav-item__desc fav-item__desc--empty" data-repo="${repo}" title="点击添加概要">暂无描述</span>`;
        const seen = Number(item.report_count) || 0;
        const total = Number(item.report_total) || 0;
        const seenBadge = total
          ? `<span class="fav-item__seen" title="共 ${total} 期定时周报，上榜 ${seen} 期">${seen}/${total}</span>`
          : "";
        return `
            <div class="fav-item" data-repo="${repo}" title="按住拖到分组上改分类；右键改分类/细分">
              <div class="fav-item__main">
                <div class="fav-item__name">
                  <a class="fav-item__repo" href="https://github.com/${repo}" target="_blank" rel="noopener" title="${repo}">${repo}</a>
                  <button type="button" class="fav-item__copy" data-repo="${repo}" title="复制名字" aria-label="复制 ${repo}">⧉</button>
                  ${seenBadge}
                </div>
                ${descRow}
              </div>
              <button type="button" class="fav-item__remove" data-repo="${repo}" title="取消收藏" aria-label="取消收藏">✕</button>
            </div>
          `;
      }

      function subHead(name, count, loose) {
        return `<div class="fav-sub${loose ? " fav-sub--loose" : ""}">` +
          `<span class="fav-sub__name">${escapeHtml(name)}</span>` +
          `<span class="fav-sub__count">${count}</span></div>`;
      }

      // 组内二级：细分的排在上面，没细分的坠底 —— 跟一级「未分类」排最后同一个规矩
      function renderGroupItems(groupItems) {
        const subs = new Map();
        const loose = [];
        groupItems.forEach(function (item) {
          const sub = (item.subcategory || "").trim();
          if (!sub) {
            loose.push(item);
            return;
          }
          if (!subs.has(sub)) {
            subs.set(sub, []);
          }
          subs.get(sub).push(item);
        });
        if (!subs.size) {
          return groupItems.map(renderItem).join("");  // 没人分过就还是一张平铺清单
        }
        const parts = [];
        subs.forEach(function (subItems, name) {
          parts.push(subHead(name, subItems.length, false), subItems.map(renderItem).join(""));
        });
        if (loose.length) {
          // 上面已经有小标题了，散项不给个头会看着像挂在最后一个细分底下
          parts.push(subHead("未细分", loose.length, true), loose.map(renderItem).join(""));
        }
        return parts.join("");
      }

      function render(list) {
        favCount.textContent = String(list.length);
        if (!list.length) {
          favList.innerHTML = '<div class="fav-empty">还没有收藏。在对话里让我分析后说“收藏它”，或在报告页点⭐。</div>';
          return;
        }
        const groups = new Map();
        list.forEach(function (item) {
          const key = (item.category || "").trim() || UNCATEGORIZED;
          if (!groups.has(key)) {
            groups.set(key, []);
          }
          groups.get(key).push(item);
        });
        const keys = Array.from(groups.keys()).sort(function (a, b) {
          if (a === UNCATEGORIZED) {
            return 1;
          }
          if (b === UNCATEGORIZED) {
            return -1;
          }
          return 0;  // 其余保持首次出现顺序（稳定排序）
        });
        favList.innerHTML = keys.map(function (key) {
          const groupItems = groups.get(key);
          const renameBtn = key === UNCATEGORIZED ? "" :
            `<button type="button" class="fav-group__rename" title="重命名分类" aria-label="重命名分类 ${escapeHtml(key)}">✎</button>`;
          return `<details class="fav-group" data-group="${escapeHtml(key)}"${expanded.has(key) ? " open" : ""}>
              <summary class="fav-group__head"><span class="fav-group__name">${escapeHtml(key)}</span>${renameBtn}<span class="fav-group__count">${groupItems.length}</span></summary>
              <div class="fav-group__items">${renderGroupItems(groupItems)}</div>
            </details>`;
        }).join("");
      }

      if (window.GitHubHotFavorites) {
        window.GitHubHotFavorites.subscribe(render);
        window.GitHubHotFavorites.ready();
      }
    }

    const REPORT_NAME_RE = /[\w.-]+\.md/g;

    function linkKnownReports(html) {
      const names = new Set((latestReports || []).map((r) => r && r.name).filter(Boolean));
      if (names.size === 0) {
        return html;
      }
      const host = document.createElement("div");
      host.innerHTML = html;

      const walker = document.createTreeWalker(host, NodeFilter.SHOW_TEXT);
      const targets = [];
      while (walker.nextNode()) {
        const node = walker.currentNode;
        const parent = node.parentElement;
        if (parent && parent.closest("a, code, pre")) continue;
        if (node.nodeValue && node.nodeValue.includes(".md")) targets.push(node);
      }

      for (const node of targets) {
        const text = node.nodeValue;
        const frag = document.createDocumentFragment();
        let cursor = 0;
        for (const match of text.matchAll(REPORT_NAME_RE)) {
          if (!names.has(match[0])) continue;
          if (match.index > cursor) {
            frag.appendChild(document.createTextNode(text.slice(cursor, match.index)));
          }
          const link = document.createElement("a");
          link.href = getReportHtmlUrl(match[0]);
          link.target = "_blank";
          link.rel = "noopener";
          link.textContent = match[0];
          frag.appendChild(link);
          cursor = match.index + match[0].length;
        }
        if (cursor === 0) continue;       // 这个节点里没有已知报告名
        if (cursor < text.length) {
          frag.appendChild(document.createTextNode(text.slice(cursor)));
        }
        node.parentNode.replaceChild(frag, node);
      }
      return host.innerHTML;
    }

    function enhanceReply(text) {
      if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
        const rawHtml = marked.parse(text);
        const safeHtml = DOMPurify.sanitize(rawHtml, {
          ADD_ATTR: [
            "target", "class", "type", "data-repo", "title", "aria-label", "aria-hidden",
            "viewBox", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "xmlns",
          ],
          ALLOWED_TAGS: [
            "h1","h2","h3","h4","h5","h6","p","br","hr","ul","ol","li",
            "strong","em","del","code","pre","blockquote","a","img",
            "table","thead","tbody","tfoot","caption","tr","th","td",
            "div","span","sub","sup","b","i","u","s","abbr","input",
            "button","svg","path",
          ],
        });
        return linkKnownReports(safeHtml);
      }
      const escaped = escapeHtml(text);
      return linkKnownReports(escaped).replace(/\n/g, "<br>");
    }

    function getReportHtmlUrl(name) {
      return `/api/reports/${encodeURIComponent(name)}/html`;
    }

    function formatSize(size) {
      if (!Number.isFinite(size)) {
        return "大小未知";
      }
      if (size < 1024) {
        return `${size} B`;
      }
      if (size < 1024 * 1024) {
        return `${(size / 1024).toFixed(1)} KB`;
      }
      return `${(size / (1024 * 1024)).toFixed(1)} MB`;
    }

    function formatReportTime(value) {
      if (!value) {
        return "时间未知";
      }
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) {
        return "时间未知";
      }
      return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")} ${String(date.getHours()).padStart(2, "0")}:${String(date.getMinutes()).padStart(2, "0")}`;
    }

    // ── 双击消息全屏查看 ──
    (function setupFullscreen() {
      const overlay = document.getElementById("fullscreen-overlay");
      const body = document.getElementById("fullscreen-body");
      const closeBtn = document.getElementById("fullscreen-close");

      function openFullscreenFromMessage(msg) {
        const content = msg.querySelector(".md-body") || msg.lastElementChild;
        if (!content) return;

        body.innerHTML = content.innerHTML;
        overlay.hidden = false;
      }

      messagesEl.addEventListener("dblclick", (e) => {
        const msg = e.target.closest(".message.agent");
        if (!msg) return;
        openFullscreenFromMessage(msg);
      });

      function closeFullscreen() {
        overlay.hidden = true;
      }

      closeBtn.addEventListener("click", closeFullscreen);
      overlay.addEventListener("click", (e) => {
        if (e.target === overlay) closeFullscreen();
      });
      document.addEventListener("keydown", (e) => {
        if (e.key === "Escape" && !overlay.hidden) closeFullscreen();
      });
    })();

    // ── 手机端 Tab 切换 ──
    (function setupMobileTabs() {
      const tabsBar = document.getElementById("mobile-tabs");
      const shellEl = document.querySelector(".shell");
      const chatPanel = document.querySelector(".chat-panel");
      const reportPanel = document.querySelector(".report-panel");
      if (!tabsBar || !shellEl || !chatPanel || !reportPanel) return;

      const TAB_KEY = "gh-hot-active-tab";
      const buttons = tabsBar.querySelectorAll("button[data-tab]");
      const mobileQuery = window.matchMedia("(max-width: 640px)");
      const validTabs = new Set(["chat", "report"]);
      let swipeStartX = null;
      let swipeStartY = null;

      function getPreferredTab() {
        const hashTab = window.location.hash.replace(/^#/, "");
        if (validTabs.has(hashTab)) {
          return hashTab;
        }
        try {
          const savedTab = localStorage.getItem(TAB_KEY);
          if (validTabs.has(savedTab)) {
            return savedTab;
          }
        } catch (error) {
        }
        return "chat";
      }

      function persistTab(target) {
        try {
          localStorage.setItem(TAB_KEY, target);
        } catch (error) {
        }
        const nextHash = `#${target}`;
        if (window.location.hash !== nextHash) {
          history.replaceState(null, "", nextHash);
        }
      }

      function switchTab(target, options = {}) {
        const normalizedTarget = validTabs.has(target) ? target : "chat";
        buttons.forEach(btn => btn.classList.toggle("active", btn.dataset.tab === target));
        if (normalizedTarget === "chat") {
          chatPanel.hidden = false;
          reportPanel.hidden = true;
        } else {
          chatPanel.hidden = true;
          reportPanel.hidden = false;
          if (options.refreshReports !== false) {
            loadReports();
          }
        }
        if (options.persist !== false) {
          persistTab(normalizedTarget);
        }
      }

      buttons.forEach(btn => {
        btn.addEventListener("click", () => switchTab(btn.dataset.tab));
      });

      shellEl.addEventListener("touchstart", (event) => {
        if (!mobileQuery.matches || event.touches.length !== 1 || !usageHelpOverlay.hidden) {
          return;
        }
        if (event.target.closest("textarea, button, a, .usage-help-card")) {
          swipeStartX = null;
          swipeStartY = null;
          return;
        }
        const touch = event.touches[0];
        swipeStartX = touch.clientX;
        swipeStartY = touch.clientY;
      }, { passive: true });

      shellEl.addEventListener("touchend", (event) => {
        if (!mobileQuery.matches || swipeStartX === null || swipeStartY === null) {
          return;
        }

        const touch = event.changedTouches[0];
        const deltaX = touch.clientX - swipeStartX;
        const deltaY = touch.clientY - swipeStartY;
        swipeStartX = null;
        swipeStartY = null;

        if (Math.abs(deltaX) < 72 || Math.abs(deltaX) < Math.abs(deltaY) * 1.3) {
          return;
        }

        const activeTab = Array.from(buttons).find(btn => btn.classList.contains("active"))?.dataset.tab || "chat";
        if (deltaX < 0 && activeTab === "chat") {
          switchTab("report");
        } else if (deltaX > 0 && activeTab === "report") {
          switchTab("chat", { refreshReports: false });
        }
      }, { passive: true });

      window.addEventListener("hashchange", () => {
        if (!mobileQuery.matches) {
          return;
        }
        const hashTab = window.location.hash.replace(/^#/, "");
        if (validTabs.has(hashTab)) {
          switchTab(hashTab, { persist: true, refreshReports: hashTab === "report" });
        }
      });

      function syncTabLayout() {
        if (mobileQuery.matches) {
          const target = getPreferredTab();
          switchTab(target);
        } else {
          chatPanel.hidden = false;
          reportPanel.hidden = false;
        }
      }

      syncTabLayout();
      mobileQuery.addEventListener("change", syncTabLayout);

      onReportsRendered((prevCount, count) => {
        if (count > prevCount && reportPanel.hidden) {
          const reportTab = tabsBar.querySelector('[data-tab="report"]');
          if (reportTab && !reportTab.classList.contains("active")) {
            reportTab.style.color = "var(--accent)";
            setTimeout(() => { reportTab.style.color = ""; }, 2000);
          }
        }
      });
    })();

    // ── 桌面端侧栏：折叠/展开 + 拖拽调宽（对齐 VSCode sash 语义）──
    (function setupSidebarToggle() {
      const shellEl = document.querySelector(".shell");
      const handle = document.getElementById("sidebar-toggle");
      if (!shellEl || !handle) return;

      const headerBtn = document.getElementById("sidebar-collapse");
      const sash = document.getElementById("sidebar-sash");
      const badge = document.getElementById("sidebar-fav-badge");
      const KEY = "gh-hot-sidebar-collapsed";
      const WIDTH_KEY = "gh-hot-sidebar-width";
      const DEFAULT_W = 38;  // 与 CSS --sidebar-w 默认值保持一致
      const MIN_W = 24;
      const MAX_W = 60;
      const SNAP_W = 18;     // 拖到此百分比以下松手 → 吸附收起
      const desktopQuery = window.matchMedia("(min-width: 981px)");
      const usageOverlay = document.getElementById("usage-help-overlay");
      const fsOverlay = document.getElementById("fullscreen-overlay");

      function isCollapsed() {
        return shellEl.classList.contains("sidebar-collapsed");
      }
      function apply(collapsed) {
        shellEl.classList.toggle("sidebar-collapsed", collapsed);
        handle.classList.toggle("is-collapsed", collapsed);  // 收起态才显示边缘把手
        handle.setAttribute("aria-expanded", String(!collapsed));
        if (headerBtn) {
          headerBtn.setAttribute("aria-expanded", String(!collapsed));
        }
      }
      function persist(collapsed) {
        try { localStorage.setItem(KEY, collapsed ? "1" : "0"); } catch (error) {}
      }
      function toggle() {
        const next = !isCollapsed();
        apply(next);
        persist(next);
      }
      function currentWidth() {
        const raw = parseFloat(shellEl.style.getPropertyValue("--sidebar-w"));
        return isNaN(raw) ? DEFAULT_W : raw;
      }
      function setWidth(pct, save) {
        const w = Math.min(MAX_W, Math.max(MIN_W, pct));
        shellEl.style.setProperty("--sidebar-w", w + "%");
        if (save) {
          try { localStorage.setItem(WIDTH_KEY, String(w)); } catch (error) {}
        }
      }

      let savedW = NaN;
      try { savedW = parseFloat(localStorage.getItem(WIDTH_KEY)); } catch (error) {}
      if (!isNaN(savedW)) setWidth(savedW, false);

      let initial = false;
      try { initial = localStorage.getItem(KEY) === "1"; } catch (error) {}
      apply(initial);  // 手机端 CSS 未启用折叠规则，此 class 无副作用

      handle.addEventListener("click", toggle);
      if (headerBtn) headerBtn.addEventListener("click", toggle);

      document.addEventListener("keydown", (event) => {
        if ((event.ctrlKey || event.metaKey) && !event.altKey && !event.shiftKey
            && (event.key === "b" || event.key === "B")) {
          if (!desktopQuery.matches) return;
          event.preventDefault();
          toggle();
          return;
        }
        if (event.key !== "Escape" || !desktopQuery.matches || !isCollapsed()) {
          return;
        }
        if ((usageOverlay && !usageOverlay.hidden) || (fsOverlay && !fsOverlay.hidden)) {
          return;  // 让位给弹层自己的 Esc 关闭逻辑
        }
        apply(false);
        persist(false);
      });

      if (sash) {
        let dragging = false;
        let rawPct = DEFAULT_W;  // 未截断的拖拽值，用于判断是否吸附收起

        function endDrag(event) {
          if (!dragging) return;
          dragging = false;
          sash.classList.remove("is-dragging");
          shellEl.classList.remove("is-resizing");
          document.body.style.userSelect = "";
          try { sash.releasePointerCapture(event.pointerId); } catch (error) {}
          if (rawPct < SNAP_W) {
            setWidth(DEFAULT_W, true);  // 吸附收起：宽度复位，留给下次展开
            apply(true);
            persist(true);
            return;
          }
          setWidth(rawPct, true);
        }

        sash.addEventListener("pointerdown", (event) => {
          if (!desktopQuery.matches || isCollapsed()) return;
          dragging = true;
          rawPct = currentWidth();
          try { sash.setPointerCapture(event.pointerId); } catch (error) {}
          sash.classList.add("is-dragging");
          shellEl.classList.add("is-resizing");  // 拖拽期间关掉过渡，保证跟手
          document.body.style.userSelect = "none";
          event.preventDefault();
        });
        sash.addEventListener("pointermove", (event) => {
          if (!dragging) return;
          const rect = shellEl.getBoundingClientRect();
          if (!rect.width) return;
          rawPct = (rect.right - event.clientX) / rect.width * 100;
          setWidth(rawPct, false);
        });
        sash.addEventListener("pointerup", endDrag);
        sash.addEventListener("pointercancel", endDrag);

        sash.addEventListener("dblclick", () => {
          setWidth(DEFAULT_W, true);
          if (isCollapsed()) {
            apply(false);
            persist(false);
          }
        });

        sash.addEventListener("keydown", (event) => {
          const step = event.shiftKey ? 5 : 2;
          if (event.key === "ArrowLeft") {
            setWidth(currentWidth() + step, true);  // 分隔条左移 = 侧栏变宽
          } else if (event.key === "ArrowRight") {
            setWidth(currentWidth() - step, true);
          } else if (event.key === "Home") {
            setWidth(DEFAULT_W, true);
          } else {
            return;
          }
          event.preventDefault();
        });
      }

      if (badge && window.GitHubHotFavorites) {
        window.GitHubHotFavorites.subscribe((list) => {
          const n = list.length;
          badge.textContent = n > 99 ? "99+" : String(n);
          badge.hidden = n === 0;
        });
      }
    })();
