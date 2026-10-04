// Vercel serverless function: POST /api/chat
// Uses Google's Gemini API (free tier via Google AI Studio).
// The key lives ONLY in the Vercel environment variable GEMINI_API_KEY.
// Never put the key in index.html.

// Primary model; override without code changes by setting GEMINI_MODEL in Vercel.
// If the primary model is not found (404), we fall back to the next one.
const MODELS = [process.env.GEMINI_MODEL || "gemini-2.5-flash-lite", "gemini-2.5-flash"];
const MAX_TURNS = 10;        // how many recent messages we send to the model
const MAX_CHARS = 500;       // max length of each user message
const RATE_LIMIT = 20;       // requests per IP per 10 minutes (best effort)
const WINDOW_MS = 10 * 60 * 1000;

const SYSTEM_PROMPT = `You are the friendly assistant for Allam Chai, a tea and dairy shop in Keesara, Medchal-Malkajgiri District, Telangana 501301. The shop has served fresh brews for 30+ years.

SHOP FACTS
- Open daily 4:00 AM to 9:00 PM.
- Phone / WhatsApp: +91 96762 22557.
- Google Maps: https://maps.google.com/?q=Allam+Chai+Keesara
- Orders: store pickup, or delivery within Keesara. Delivery area, charges and timing are confirmed by the shop on WhatsApp.
- Party and bulk orders (office meetings, family events, flask deliveries): customers fill the "Bulk & Party Catering Inquiry" form on the site or message the shop on WhatsApp.

MENU
Tea: Special Fresh Allam Chai (ginger tea). Ordered by number of people, minimum 2 people.
Cold beverages: Kinley Water 1 L; Bisleri Water 1 L; Kinley Strong Soda 250 ml and 500 ml; Thums Up 250 ml, 1 L, 2 L; Sprite 250 ml, 1 L, 2 L; Pulpy Orange 250 ml and 2 L; Maaza Mango 250 ml, 600 ml, 2 L; Appy Fizz 250 ml; Godrej Jersey Badam Milk 200 ml; Sting 200 ml; Mountain Dew 500 ml.
Dairy (Heritage): Total Curd 1 kg container and 500 g; Premium Salted Butter 500 g; Fresh Paneer 200 g.

HOW TO ORDER ON THE WEBSITE
Use the + and - buttons next to items, tap "Proceed to Order", choose Store Pickup or Express Delivery, enter name and phone (and address for delivery), then tap "Send Order via WhatsApp". The shop confirms availability and the total on WhatsApp.

RULES
- The website shows no prices. Never state or guess prices, discounts, delivery charges, or stock. Say the shop confirms these on WhatsApp or by phone.
- Only use the facts above. If you do not know something, say so and give the phone / WhatsApp number.
- Reply in the language the customer writes in (English, Telugu, Hindi, or a mix). 
- Keep replies short: 1 to 3 sentences, plain text, no markdown, no headings.
- You cannot place orders yourself. Guide people to the site's order buttons or WhatsApp.
- Stay on topic (the shop, its menu, timings, location, orders). Politely decline anything else.`;

// Best-effort in-memory rate limit (resets when the function instance recycles)
const hits = new Map();
function limited(ip) {
  const now = Date.now();
  const list = (hits.get(ip) || []).filter((t) => now - t < WINDOW_MS);
  list.push(now);
  hits.set(ip, list);
  if (hits.size > 5000) hits.clear();
  return list.length > RATE_LIMIT;
}

const API = "https://generativelanguage.googleapis.com/v1beta";

async function callGemini(model, contents) {
  const generationConfig = { maxOutputTokens: 400, temperature: 0.4 };
  // 2.5 Flash models "think" by default, which can use up the token budget; switch it off
  if (model.includes("2.5-flash")) generationConfig.thinkingConfig = { thinkingBudget: 0 };

  const r = await fetch(`${API}/models/${model}:generateContent`, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-goog-api-key": process.env.GEMINI_API_KEY,
    },
    body: JSON.stringify({
      systemInstruction: { parts: [{ text: SYSTEM_PROMPT }] },
      contents,
      generationConfig,
    }),
  });

  if (!r.ok) {
    const text = await r.text();
    console.error(`Gemini error: model=${model} status=${r.status} body=${text.slice(0, 400)}`);
    return { status: r.status, reply: "" };
  }

  const data = await r.json();
  const parts = (data.candidates && data.candidates[0] && data.candidates[0].content && data.candidates[0].content.parts) || [];
  return { status: 200, reply: parts.map((p) => p.text || "").join("").trim() };
}

// Ask Google which models this key can actually use, and pick a Flash model
async function discoverModel() {
  try {
    const r = await fetch(`${API}/models?pageSize=200`, {
      headers: { "x-goog-api-key": process.env.GEMINI_API_KEY },
    });
    if (!r.ok) {
      console.error(`Gemini list models failed: status=${r.status} body=${(await r.text()).slice(0, 300)}`);
      return null;
    }
    const data = await r.json();
    const usable = (data.models || [])
      .filter((m) => (m.supportedGenerationMethods || []).includes("generateContent"))
      .map((m) => m.name.replace("models/", ""))
      .filter((n) => n.includes("flash") && !/image|tts|live|audio|thinking|preview|exp/.test(n));
    console.error("Gemini usable flash models: " + usable.join(", "));
    return usable.find((n) => n.includes("lite")) || usable[0] || null;
  } catch (e) {
    console.error("Gemini discover failed", e);
    return null;
  }
}

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Method not allowed" });
  }

  if (!process.env.GEMINI_API_KEY) {
    return res.status(500).json({ error: "Chat is not configured." });
  }

  const ip = (req.headers["x-forwarded-for"] || "").split(",")[0].trim() || "unknown";
  if (limited(ip)) {
    return res.status(429).json({ error: "Too many messages. Please try again in a few minutes." });
  }

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body || {};
    const incoming = Array.isArray(body.messages) ? body.messages : [];

    // Keep only well-formed user/assistant turns, trimmed and length-limited
    let messages = incoming
      .filter((m) => m && (m.role === "user" || m.role === "assistant") && typeof m.content === "string")
      .map((m) => ({ role: m.role, content: m.content.slice(0, MAX_CHARS) }))
      .filter((m) => m.content.trim().length > 0)
      .slice(-MAX_TURNS);

    // The conversation must start with and end on a user message
    while (messages.length && messages[0].role !== "user") messages.shift();
    if (!messages.length || messages[messages.length - 1].role !== "user") {
      return res.status(400).json({ error: "No message provided." });
    }

    // Gemini uses the role "model" instead of "assistant"
    const contents = messages.map((m) => ({
      role: m.role === "assistant" ? "model" : "user",
      parts: [{ text: m.content }],
    }));

    let reply = "";
    let lastStatus = 0;

    for (const model of MODELS) {
      const out = await callGemini(model, contents);
      lastStatus = out.status;
      reply = out.reply;
      if (out.status !== 404) break; // 404 = model not available to this key: try the next one
    }

    // Every listed model was missing: ask Google which models this key can use
    if (!reply && lastStatus === 404) {
      const found = await discoverModel();
      if (found && !MODELS.includes(found)) {
        const out = await callGemini(found, contents);
        lastStatus = out.status;
        reply = out.reply;
      }
    }

    if (!reply && lastStatus === 429) {
      return res.status(503).json({ error: "The assistant is busy right now. Please try again in a minute." });
    }
    if (!reply && lastStatus && lastStatus !== 200) {
      return res.status(502).json({ error: "The assistant is unavailable right now." });
    }

    return res.status(200).json({
      reply: reply || "Sorry, I couldn't answer that. Please WhatsApp us on +91 96762 22557.",
    });
  } catch (err) {
    console.error(err);
    return res.status(500).json({ error: "Something went wrong." });
  }
};
