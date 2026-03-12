// server-puter.js - OpenAI to Puter API Proxy
// Adapted from the NVIDIA NIM proxy — same interface, different backend.
// Puter exposes a fully OpenAI-compatible endpoint, so the changes are minimal:
//   • NIM_API_BASE  → https://api.puter.com/puterai/openai/v1
//   • NIM_API_KEY   → PUTER_AUTH_TOKEN  (get yours at puter.com/dashboard → Copy)
//   • Model strings → Puter model IDs   (see MODEL_MAPPING below)
//
// Puter does NOT return reasoning_content chunks, so the SHOW_REASONING /
// ENABLE_THINKING_MODE toggles only apply to models that natively emit
// <think> blocks inside the content field (e.g. deepseek-r1 variants).

const express = require('express');
const cors    = require('cors');
const axios   = require('axios');

const app  = express();
const PORT = process.env.PORT || 3000;

// ── Middleware ────────────────────────────────────────────────────────────────
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ limit: '100mb', extended: true }));

// ── Puter configuration ───────────────────────────────────────────────────────
const PUTER_API_BASE  = 'https://api.puter.com/puterai/openai/v1';
const PUTER_AUTH_TOKEN = process.env.PUTER_AUTH_TOKEN; // set in Render env vars

// 🔥 REASONING DISPLAY TOGGLE
// Wraps <think>…</think> blocks that some models emit inside content.
const SHOW_REASONING = true;

// ── Model mapping ─────────────────────────────────────────────────────────────
// Maps OpenAI-style aliases → Puter model IDs.
// Full list: https://docs.puter.com/AI/
const MODEL_MAPPING = {
  // OpenAI
  'gpt-3.5-turbo'   : 'gpt-5-nano',
  'gpt-4'           : 'gpt-4o',
  'gpt-4-turbo'     : 'gpt-4o',
  'gpt-4o'          : 'z-ai/glm-5',          // keep your GLM-5 on gpt-4o alias

  // Anthropic (via Puter)
  'claude-3-opus'   : 'claude-opus-4-5',
  'claude-3-sonnet' : 'claude-sonnet-4-5',
  'claude-3-haiku'  : 'claude-haiku-4-5',

  // Google (via Puter)
  'gemini-pro'      : 'gemini-2.5-flash-lite',
  'gemini-ultra'    : 'gemini-2.5-pro',

  // Misc / reasoning models
  'deepseek'        : 'deepseek-ai/deepseek-r1',
  'grok'            : 'grok-4-1-fast',
  'qwen'            : 'qwen/qwen-2.5-72b-instruct',
  'glm-5'           : 'z-ai/glm-5',
};

// ── Fallback model selection ──────────────────────────────────────────────────
function pickFallbackModel(model) {
  const m = model.toLowerCase();
  if (m.includes('opus') || m.includes('gpt-4') || m.includes('405b'))
    return 'claude-opus-4-5';
  if (m.includes('sonnet') || m.includes('claude') || m.includes('70b'))
    return 'claude-sonnet-4-5';
  if (m.includes('glm') || m.includes('z-ai'))
    return 'z-ai/glm-5';
  return 'gpt-5-nano';
}

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({
    status           : 'ok',
    service          : 'OpenAI → Puter Proxy',
    reasoning_display: SHOW_REASONING,
  });
});

// ── /v1/models ────────────────────────────────────────────────────────────────
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map(id => ({
    id,
    object    : 'model',
    created   : Date.now(),
    owned_by  : 'puter-proxy',
  }));
  res.json({ object: 'list', data: models });
});

// ── /v1/chat/completions ──────────────────────────────────────────────────────
app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream } = req.body;

    // Resolve model ID
    let puterModel = MODEL_MAPPING[model];
    if (!puterModel) {
      // Try the raw model string directly (Puter passes it through)
      puterModel = model || pickFallbackModel(model);
    }

    // Build request
    const puterRequest = {
      model      : puterModel,
      messages,
      temperature: temperature ?? 0.5,
      max_tokens : max_tokens  ?? 32768,
      stream     : stream      || false,
    };

    // Call Puter
    const response = await axios.post(
      `${PUTER_API_BASE}/chat/completions`,
      puterRequest,
      {
        headers     : {
          'Authorization': `Bearer ${PUTER_AUTH_TOKEN}`,
          'Content-Type' : 'application/json',
        },
        responseType: stream ? 'stream' : 'json',
      }
    );

    // ── Streaming ─────────────────────────────────────────────────────────────
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer          = '';
      let thinkOpen       = false; // tracks whether we opened a <think> block

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;

          if (line.includes('[DONE]')) {
            // Close any open <think> block before finishing
            if (SHOW_REASONING && thinkOpen) {
              // emit a closing chunk
              const closing = buildDeltaChunk(model, '</think>\n\n');
              res.write(`data: ${JSON.stringify(closing)}\n\n`);
              thinkOpen = false;
            }
            res.write(line + '\n');
            continue;
          }

          try {
            const data = JSON.parse(line.slice(6));
            if (data.choices?.[0]?.delta) {
              const delta    = data.choices[0].delta;
              // Puter returns content directly; reasoning_content may still
              // appear on some model wrappers, so we handle both.
              const reasoning = delta.reasoning_content;
              const content   = delta.content;

              if (SHOW_REASONING) {
                let out = '';
                if (reasoning && !thinkOpen) { out = '<think>\n' + reasoning; thinkOpen = true; }
                else if (reasoning)           { out = reasoning; }
                if (content && thinkOpen)     { out += '</think>\n\n' + content; thinkOpen = false; }
                else if (content)             { out += content; }

                if (out) delta.content = out;
                else     delta.content = delta.content ?? '';
              } else {
                delta.content = content ?? '';
              }
              delete delta.reasoning_content;
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch {
            res.write(line + '\n');
          }
        }
      });

      response.data.on('end',   ()    => res.end());
      response.data.on('error', (err) => { console.error('Stream error:', err); res.end(); });

    // ── Non-streaming ─────────────────────────────────────────────────────────
    } else {
      const openaiResponse = {
        id     : `chatcmpl-${Date.now()}`,
        object : 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model,
        choices: response.data.choices.map(choice => {
          let fullContent = choice.message?.content || '';

          if (SHOW_REASONING && choice.message?.reasoning_content) {
            fullContent =
              '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' +
              fullContent;
          }

          return {
            index         : choice.index,
            message       : { role: choice.message.role, content: fullContent },
            finish_reason : choice.finish_reason,
          };
        }),
        usage: response.data.usage || {
          prompt_tokens    : 0,
          completion_tokens: 0,
          total_tokens     : 0,
        },
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy error:', error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.message || 'Internal server error',
        type   : 'invalid_request_error',
        code   : error.response?.status || 500,
      },
    });
  }
});

// ── 404 catch-all ─────────────────────────────────────────────────────────────
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type   : 'invalid_request_error',
      code   : 404,
    },
  });
});

// ── Helpers ───────────────────────────────────────────────────────────────────
function buildDeltaChunk(model, text) {
  return {
    id     : `chatcmpl-${Date.now()}`,
    object : 'chat.completion.chunk',
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [{ index: 0, delta: { content: text }, finish_reason: null }],
  };
}

// ── Start ─────────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
  console.log(`OpenAI → Puter Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
});
