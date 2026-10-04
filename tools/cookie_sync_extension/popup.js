const serverInput = document.querySelector("#server");
const tokenInput = document.querySelector("#token");
const status = document.querySelector("#status");

function show(message, error = false) {
  status.textContent = message;
  status.style.color = error ? "#b42318" : "#555";
}

async function readCookieHeader() {
  const urls = ["https://h5api.m.goofish.com/", "https://www.goofish.com/"];
  const all = new Map();
  for (const url of urls) {
    const cookies = await chrome.cookies.getAll({url});
    for (const cookie of cookies) all.set(cookie.name, cookie.value);
  }
  return [...all.entries()].map(([name, value]) => `${name}=${value}`).join("; ");
}

async function syncCookie() {
  const server = serverInput.value.trim().replace(/\/$/, "");
  const token = tokenInput.value.trim();
  if (!server || !token) {
    show("请填写服务器地址和同步密钥", true);
    return;
  }

  show("正在读取浏览器 Cookie...");
  try {
    const serverOrigin = `${new URL(server).origin}/*`;
    const granted = await chrome.permissions.request({origins: [serverOrigin]});
    if (!granted) throw new Error("需要允许扩展访问同步服务器");
    const cookie = await readCookieHeader();
    if (!cookie.includes("unb=") || !cookie.includes("_m_h5_tk=")) {
      throw new Error("当前闲鱼登录状态缺少必要 Cookie");
    }
    const response = await fetch(`${server}/api/cookie-sync`, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${token}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({cookie})
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "服务器拒绝了同步请求");
    await chrome.storage.local.set({server, token});
    show("同步成功，客服正在重新连接闲鱼");
  } catch (error) {
    show(error.message || "同步失败", true);
  }
}

document.querySelector("#sync").addEventListener("click", syncCookie);
chrome.storage.local.get(["server", "token"]).then((values) => {
  serverInput.value = values.server || "";
  tokenInput.value = values.token || "";
});
