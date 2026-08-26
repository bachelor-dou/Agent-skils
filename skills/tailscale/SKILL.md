---
name: tailscale
description: tailscale 的安装与用法速查。当用户要安装/使用 tailscale、加 serve 端口映射或问其命令细节时使用（先读 skills/install-kb）。
---
# tailscale 速查

配套：无

## 安装（上次实测命令）

本机（GPU 服务器 gpu24）已装好并登录，未记录安装过程；只记录 serve 用法。

## 用法

```bash
# 查看当前 serve 映射
tailscale serve status

# 新增一条 HTTPS 映射：tailnet 内经 https://<主机名>.tail9c237a.ts.net:9001 访问本机 9001
tailscale serve --bg --https=9001 http://127.0.0.1:9001

# 关闭指定端口的映射（不影响其他端口的映射）
tailscale serve --https=9001 off

# 查看 tailnet 设备列表与本机 IP
tailscale status
```

- 多条 serve 映射按监听端口区分，互不影响、可共存（如 443→8888 与 9001→9001 并存）。
- serve 默认 tailnet only，仅同一 tailnet 内设备可访问。

## 验证

在 tailnet 内**另一台设备**（笔记本/手机）浏览器打开 `https://gpu24.tail9c237a.ts.net:9001/`，能看到服务页面即可。

## 踩坑与解决

- 现象：在本机 curl 自己的 serve 链接（https://gpu24.tail9c237a.ts.net:9001/）报 TLS 错误
  `wrong version number` → 原因：本机发出的流量不经过 tailscaled 的 serve 代理，直接连到了
  监听 0.0.0.0:9001 的明文 HTTP 服务，TLS 握手自然失败 → 解决：serve 链接无法在本机自测，
  必须用 tailnet 内其他设备验证；本机只能 `curl http://127.0.0.1:9001/` 测原始服务。

## 更新记录

2026-08-22 GPU 服务器（gpu24, linux）：为 hot_project(9001) 加 serve 映射实测跑通；确认多条映射可共存；记录"本机无法自测 serve 链接"的坑。
