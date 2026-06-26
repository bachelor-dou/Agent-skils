    const SESSION_STORAGE_KEY = "gh-hot-session-id";
    const SESSION_LAST_ACTIVE_KEY = "gh-hot-session-last-active";
    const SESSION_TTL_SECONDS = Number("__SESSION_TTL_SECONDS__") || 3600;
    const SESSION_EXPIRED_TEXT = "上一会话已过期，已自动开启新会话。可以直接开始新的提问。";
    const messagesEl = document.getElementById("messages");
    const inputEl = document.getElementById("composer-input");
    const sendButton = document.getElementById("send-button");
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

    const CHAT_HISTORY_KEY = "gh-hot-chat-history";
    const MAX_CHAT_HISTORY = 100; // 聊天历史最多保留的消息数，防止 localStorage 撑爆

    // 共享工具（来自 common.js）
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
    let favoritesPanel = null;

    restoreMessages();
    if (initialSessionExpired) {
      renderSessionIntro(false, SESSION_EXPIRED_TEXT);
    }

    setupComposer();
    setupQuickActions();
    setupSessionActions();
    setupUsageHelp();
    setupAutoHideStatusStrip();
    favoritesPanel = initFavoritesPanel();
    connectWebSocket();
    loadReports();

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
      addMessage("agent", introText || "欢迎，你可以点击快捷按钮生成综合榜、新项目榜、查看 Trending，也可以输入框自定义参数（如类别、数量、增长阈值等）。", {
        asHtml: false,
        hideLabel: true,
        variant: "intro",
      });
    }

    function saveChatHistory() {
      // 截断历史，仅保留最近 MAX_CHAT_HISTORY 条，避免 localStorage 无限增长
      if (chatHistory.length > MAX_CHAT_HISTORY) {
        chatHistory = chatHistory.slice(-MAX_CHAT_HISTORY);
      }
      try {
        localStorage.setItem(CHAT_HISTORY_KEY + "-" + sessionId, JSON.stringify(chatHistory));
      } catch (e) {
        // localStorage 满时静默失败，不调用 addMessage 避免无限递归
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
              messagesEl.appendChild(item);
            });
            messagesEl.scrollTop = messagesEl.scrollHeight;
            return;
          }
        }
      } catch (e) {
        // 解析失败时忽略
      }
      // 无历史记录，显示欢迎消息
      renderSessionIntro(false);
    }

    function clearChatHistory() {
      chatHistory = [];
      clearStoredSessionState(sessionId);
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

    function setupQuickActions() {
      document.querySelectorAll("[data-prompt]").forEach((button) => {
        button.addEventListener("click", () => {
          inputEl.value = button.dataset.prompt || "";
          autoResize();
          sendMessage();
        });
      });
    }

    function setupSessionActions() {
      endSessionButton.addEventListener("click", endSession);
    }

    function setupUsageHelp() {
      if (!usageHelpButton || !usageHelpOverlay || !usageHelpClose) {
        return;
      }

      function openUsageHelp() {
        if (favoritesPanel && typeof favoritesPanel.close === "function") {
          favoritesPanel.close();
        }
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

    function initFavoritesPanel() {
      if (typeof window.setupChatFavoritesPanel !== "function") {
        return null;
      }

      return window.setupChatFavoritesPanel({
        onBeforeOpen() {
          usageHelpOverlay.hidden = true;
          usageHelpButton.setAttribute("aria-expanded", "false");
        },
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
      }

      function hideChatChrome() {
        if (chromeHidden || document.activeElement === inputEl) {
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
        if (document.activeElement === inputEl) {
          showChatChrome();
          return;
        }

        if (!isMessagesScrollable()) {
          showChatChrome();
          return;
        }

        const currentScrollTop = messagesEl.scrollTop;
        const delta = Math.abs(currentScrollTop - lastScrollTop);
        lastScrollTop = currentScrollTop;
        if (delta < 2) {
          return;
        }

        hideChatChrome();
        if (!isTouchHolding) {
          scheduleReveal();
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
          return;
        }
        scheduleReveal();
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
      sendButton.textContent = isSending ? "发送中" : "发送";
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
      messagesEl.appendChild(item);
      messagesEl.scrollTop = messagesEl.scrollHeight;

      // 持久化（跳过 typing 指示器）
      if (!options.typing) {
        chatHistory.push({ role, content: text, isHtml: !!options.asHtml });
        saveChatHistory();
      }

      return item;
    }

    function addTypingIndicator() {
      return addMessage("agent", "正在处理", { asHtml: false, typing: true });
    }

    // 带进度条的"正在处理"气泡：默认流动动画（indeterminate）；收到真实百分比后切为确定态。
    // 返回 { item, onProgress }
    function addPendingProgress() {
      const item = addMessage("agent", "", { asHtml: true, typing: true });
      const body = item.lastElementChild;
      body.className = "md-body";
      body.innerHTML =
        '<div class="agent-progress" data-state="indeterminate">' +
          '<div class="agent-progress__head">' +
            '<span class="agent-progress__label">正在处理…</span>' +
            '<span class="agent-progress__pct"></span>' +
          '</div>' +
          '<div class="agent-progress__track"><div class="agent-progress__fill"></div></div>' +
        '</div>';
      const wrap = body.querySelector(".agent-progress");
      const fill = body.querySelector(".agent-progress__fill");
      const labelEl = body.querySelector(".agent-progress__label");
      const pctEl = body.querySelector(".agent-progress__pct");

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

      return { item, onProgress };
    }

    // 渲染一条非应答（重连推送 / 服务端主动消息）的 Agent 回复
    function renderUnsolicited(displayText) {
      addMessage("agent", "", { asHtml: true, typing: false });
      const msg = messagesEl.lastElementChild;
      const body = msg.lastElementChild;
      body.className = "md-body";
      body.innerHTML = enhanceReply(displayText);
      chatHistory.push({ role: "agent", content: body.innerHTML, isHtml: true });
      saveChatHistory();
    }

    function updateTypingIndicator(el, finalText) {
      const body = el.lastElementChild;
      body.className = "md-body";
      const html = enhanceReply(finalText);
      body.innerHTML = html;
      body.classList.remove("typing");

      // 持久化最终回复
      chatHistory.push({ role: "agent", content: html, isHtml: true });
      saveChatHistory();
    }

    async function sendMessage() {
      const message = inputEl.value.trim();
      if (!message) {
        return;
      }

      refreshExpiredSession("上一会话已过期，已自动开启新会话。本次提问将作为新会话开始。");

      addMessage("user", message, { asHtml: false });
      inputEl.value = "";
      autoResize();
      setSending(true);

      const pendingObj = addPendingProgress();
      const pending = pendingObj.item;

      try {
        const reply = socketReady ? await sendViaWebSocket(message, pendingObj.onProgress) : await sendViaHttp(message);
        updateTypingIndicator(pending, reply);
        await loadReports();
      } catch (error) {
        pending.remove();
        addMessage("system", `请求失败：${error.message}`, { asHtml: false });
      } finally {
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
      const url = `${protocol}//${window.location.host}/ws/chat/${encodeURIComponent(currentSessionId)}`;

      try {
        socket = new WebSocket(url);
      } catch (error) {
        socketReady = false;
        updateConnectionStatus("普通模式", "disconnected");
        scheduleReconnect();
        return;
      }

      // 连接超时：30 秒内未 open 则放弃
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

      // 消息分发：progress/heartbeat/reply 信封 + 兼容纯文本回复（重连推送）
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

        // 非信封消息（重连后推送的纯文本回复）：当作最终回复
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
        // 非主动关闭时自动重连
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

    function sendViaWebSocket(message, onProgress) {
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
        activeRequest = { resolve, reject, socket: requestSocket, onProgress };
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
        body: JSON.stringify({ session_id: sessionId, message }),
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

    // 报告列表刷新后的回调（取代旧的 loadReports 猴补丁）
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

      // 绑定删除按钮事件
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
            // 刷新报告列表
            loadReports();
          } catch(err) {
            window.alert("删除失败: " + err.message);
            btn.disabled = false;
          }
        });
      });
    }

    // 仅把"真实存在的报告名（latestReports 里有的 *.md）"转成链接，
    // 避免正文里随便出现的 README.md 之类被链成死链。
    function linkKnownReports(html) {
      const names = new Set((latestReports || []).map((r) => r && r.name).filter(Boolean));
      if (names.size === 0) {
        return html;
      }
      return html.replace(/([\w.-]+\.md)/g, (match) =>
        names.has(match)
          ? `<a href="${getReportHtmlUrl(match)}" target="_blank" rel="noopener">${match}</a>`
          : match
      );
    }

    function enhanceReply(text) {
      // 使用 marked 渲染 Markdown，DOMPurify 防 XSS
      if (typeof marked !== "undefined" && typeof DOMPurify !== "undefined") {
        const rawHtml = marked.parse(text);
        const safeHtml = DOMPurify.sanitize(rawHtml, {
          ADD_ATTR: [
            "target", "class", "type", "data-repo", "title", "aria-label", "aria-hidden",
            "viewBox", "fill", "stroke", "stroke-width", "stroke-linecap", "stroke-linejoin", "xmlns",
          ],
          ALLOW_TAGS: [
            "h1","h2","h3","h4","h5","h6","p","br","hr","ul","ol","li",
            "strong","em","del","code","pre","blockquote","a","img",
            "table","thead","tbody","tr","th","td","div","span","sub","sup",
            "button","svg","path",
          ],
        });
        return linkKnownReports(safeHtml);
      }
      // Fallback：无 marked/DOMPurify 时简单渲染
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
        if (!mobileQuery.matches || event.touches.length !== 1 || !usageHelpOverlay.hidden || (favoritesPanel && favoritesPanel.isOpen())) {
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

      // 生成报告后自动闪烁报告 Tab 提示用户（通过回调，而非重写 loadReports）
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
