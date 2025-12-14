// main.js
// 공통 설정 및 index.html 동작 코드

// API 주소를 바꾸고 싶으면 여기만 수정하면 됩니다.
window.APP_API_BASE = "http://localhost:8000";

function qs(sel) { return document.querySelector(sel); }
function qsa(sel) { return Array.from(document.querySelectorAll(sel)); }

function setText(el, text) { if (el) el.textContent = text; }

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const msg = await res.text();
    throw new Error(`HTTP ${res.status}: ${msg}`);
  }
  return await res.json();
}

function renderProducts(items) {
  const grid = qs("#grid");
  if (!grid) return;

  grid.innerHTML = "";

  for (const it of items) {
    const card = document.createElement("div");
    card.className = "product";
    card.addEventListener("click", () => {
      const pid = encodeURIComponent(it.product_id);
      location.href = `./product.html?product_id=${pid}`;
    });

    const img = document.createElement("img");
    img.className = "thumb";
    img.alt = "product image";
    img.src = it.image_url || "";
    img.loading = "lazy";
    img.referrerPolicy = "no-referrer";
    img.onerror = () => { img.alt = "no image"; };

    const box = document.createElement("div");

    const name = document.createElement("p");
    name.className = "pname";
    name.textContent = it.product_name || "";

    const meta = document.createElement("p");
    meta.className = "pmeta";
    meta.textContent = `${it.brand_name || ""} · product_id ${it.product_id || ""}`;

    const score = document.createElement("div");
    score.className = "pscore";
    const s = (typeof it.score === "number") ? it.score.toFixed(4) : String(it.score ?? "");
    score.textContent = `score: ${s}`;

    box.appendChild(name);
    box.appendChild(meta);
    box.appendChild(score);

    card.appendChild(img);
    card.appendChild(box);

    grid.appendChild(card);
  }
}

async function onRecommend() {
  const apiBase = window.APP_API_BASE || "http://localhost:8000";
  setText(qs("#apiBaseLabel"), apiBase);

  const btn = qs("#btnRecommend");
  const status = qs("#status");
  const countLabel = qs("#countLabel");

  const skinType = qs("#skinType")?.value || "dry";
  const category = qs("#category")?.value || "all";
  const topK = Math.max(1, Math.min(50, Number(qs("#topK")?.value || 10)));

  const concerns = qsa('#concerns input[type="checkbox"]:checked').map(el => el.value);

  const payload = {
    skin_type: skinType,
    category: category,
    top_k: topK,
    concerns: concerns,
  };

  if (btn) btn.disabled = true;
  if (status) status.textContent = "추천 불러오는 중...";
  if (countLabel) countLabel.textContent = "";

  try {
    const data = await postJson(`${apiBase}/recommend`, payload);
    const items = data.items || [];
    renderProducts(items);
    if (status) status.textContent = "";
    if (countLabel) countLabel.textContent = `총 ${items.length}개`;
  } catch (e) {
    if (status) status.textContent = "실패: " + (e?.message || String(e));
  } finally {
    if (btn) btn.disabled = false;
  }
}

(function init() {
  const apiBase = window.APP_API_BASE || "http://localhost:8000";
  setText(qs("#apiBaseLabel"), apiBase);

  const btn = qs("#btnRecommend");
  if (btn) btn.addEventListener("click", onRecommend);

  // index.html에서 Enter로 추천
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && qs("#btnRecommend")) {
      onRecommend();
    }
  });
})();
