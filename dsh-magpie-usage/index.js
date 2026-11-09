/**
 * @local/magpie-usage — host half.
 *
 * Magpie runs an OpenAI-compatible gateway on this machine and already keeps the
 * ledger every agent using it writes into. Three of its read-only endpoints are
 * the whole data source:
 *
 *   GET /v1/magpie/usage?period=1d|7d|30d   totals + one bucket per day
 *   GET /v1/magpie/quotas                   per-provider balance and plan windows
 *   GET /v1/magpie/statusline               the one-line quota summary
 *   GET /v1/magpie/usage/requests?limit=N   the newest routed requests
 *
 * The browser page cannot read them directly: the gateway answers on another
 * origin and sends no CORS headers. So this half reads the gateway on the Host
 * and republishes a trimmed snapshot under the Harness origin:
 *
 *   GET /api/magpie-usage/snapshot?period=1d&recent=12&fresh=1
 *   GET /api/magpie-usage/health
 *
 * The same snapshot backs the agent tool `magpie_usage`, so the panel and the
 * model read one implementation instead of two.
 *
 * @module @local/magpie-usage
 */

const ROUTE_BASE = '/api/magpie-usage'
const PERIODS = new Set(['today', '7d', '30d'])
const TIMEOUT_MS = 8000
/**
 * Quota cards are the slow read: Magpie asks every provider, and one unreachable
 * host (a remote magpie on a LAN that is down) holds the whole call until its own
 * deadline. Give that one a longer budget than the ledger read.
 */
const QUOTA_TIMEOUT_MS = 16000
/** Two reads of the gateway inside this window share one result (panel + tool + reload). */
const CACHE_MS = 1500

const DEFAULTS = Object.freeze({
  gateway: 'http://127.0.0.1:3425',
  apiKey: '',
  period: 'today',
  refreshSeconds: 10,
  tool: true,
  /**
   * The ledger prices in USD. `cnyRate` converts every cost for display; `0`
   * leaves the USD numbers alone. Quota balances are not touched — those arrive
   * as strings in each provider's own currency.
   */
  cnyRate: 6.8,
})

/** Trim to a finite number, or 0. Gateway counters are sometimes absent on old rows. */
function num(value) {
  const n = typeof value === 'number' ? value : Number(value)
  return Number.isFinite(n) ? n : 0
}

/** Keep only strings that carry something; everything else becomes undefined. */
function str(value) {
  return typeof value === 'string' && value !== '' ? value : undefined
}

function sendJson(response, status, body) {
  const payload = JSON.stringify(body)
  response.statusCode = status
  response.setHeader('content-type', 'application/json; charset=utf-8')
  response.setHeader('cache-control', 'no-store')
  response.end(payload)
}

function resolveConfig(config) {
  const raw = config !== null && typeof config === 'object' ? config : {}
  const gateway = str(raw.gateway) ?? DEFAULTS.gateway
  const period = PERIODS.has(raw.period) ? raw.period : DEFAULTS.period
  const seconds = Number(raw.refreshSeconds)
  const cny = Number(raw.cnyRate)
  const rate = Number.isFinite(cny) && cny >= 0 ? cny : DEFAULTS.cnyRate
  return {
    gateway: gateway.replace(/\/+$/, ''),
    apiKey: str(raw.apiKey) ?? '',
    period,
    refreshSeconds: Number.isFinite(seconds) && seconds >= 2 ? Math.min(seconds, 300) : DEFAULTS.refreshSeconds,
    tool: raw.tool === false ? false : true,
    currency: rate > 0
      ? { code: 'CNY', symbol: '¥', rate, base: 'USD' }
      : { code: 'USD', symbol: '$', rate: 1, base: 'USD' },
  }
}

/** One conversion point: every cost in the snapshot passes through here. */
function toDisplay(rate, value) {
  return num(value) * rate
}

/** Aggregate fields that are cheap to read and mean the same thing in every period. */
function trimUsage(usage, rate) {
  const timed = num(usage.timed)
  const decodeMs = num(usage.decode_ms)
  const decodeOut = num(usage.decode_out)
  return {
    period: str(usage.period) ?? '',
    since: str(usage.since) ?? '',
    calls: num(usage.calls),
    errors: num(usage.errors),
    retried: num(usage.retried),
    input: num(usage.input),
    output: num(usage.output),
    cacheRead: num(usage.cache_read),
    cacheWrite: num(usage.cache_write),
    reasoning: num(usage.reasoning),
    cost: toDisplay(rate, usage.cost),
    unpriced: num(usage.unpriced),
    timed,
    avgTtftMs: timed > 0 ? num(usage.ttft_ms) / timed : null,
    avgTokensPerSecond: decodeMs > 0 ? decodeOut / (decodeMs / 1000) : null,
  }
}

function trimSeries(series, rate) {
  if (!Array.isArray(series)) return []
  return series.map((row) => ({
    label: str(row.label) ?? '',
    time: str(row.time) ?? '',
    calls: num(row.calls),
    errors: num(row.errors),
    cost: toDisplay(rate, row.cost),
    input: num(row.input),
    output: num(row.output),
    cacheRead: num(row.cache_read),
  }))
}

function trimQuotas(quotas) {
  const rows = Array.isArray(quotas?.data) ? quotas.data : []
  return rows.map((row) => ({
    provider: str(row.provider) ?? '',
    name: str(row.name) ?? str(row.provider) ?? '',
    kind: str(row.kind) ?? '',
    plan: str(row.plan),
    user: str(row.user),
    balance: str(row.balance),
    readAt: str(row.readAt),
    lastServedAt: str(row.lastServedAt),
    error: str(row.error),
    windows: Array.isArray(row.windows)
      ? row.windows.map((window) => ({
          name: str(window.name) ?? '',
          // Provider windows arrive as raw floats (35.629999991999995). One decimal is
          // the most any of them means.
          used: Math.round(num(window.used) * 10) / 10,
          remaining: Math.round(num(window.remaining) * 10) / 10,
          resetsAt: str(window.resetsAt),
        }))
      : [],
  }))
}

function trimRequests(payload, limit, rate) {
  const rows = Array.isArray(payload?.rows) ? payload.rows : []
  return rows.slice(0, limit).map((row) => ({
    t: str(row.t) ?? '',
    agent: str(row.agent) ?? '',
    agentName: str(row.agentName) ?? str(row.agent) ?? '',
    provider: str(row.provider) ?? '',
    providerName: str(row.providerName) ?? str(row.provider) ?? '',
    model: str(row.model) ?? '',
    req: str(row.req) ?? '',
    in: num(row.in),
    out: num(row.out),
    cacheRead: num(row.cache_read),
    reasoning: num(row.reasoning),
    ms: num(row.ms),
    ttftMs: num(row.ttft_ms),
    status: num(row.status),
    cost: toDisplay(rate, row.cost),
    priced: row.priced !== false,
    ep: str(row.ep) ?? '',
  }))
}

function trimModels(usage, limit, rate) {
  const rows = Array.isArray(usage.models) ? usage.models.slice() : []
  rows.sort((a, b) => num(b.cost) - num(a.cost) || num(b.calls) - num(a.calls))
  return rows.slice(0, limit).map((row) => ({
    id: str(row.id) ?? '',
    provider: str(row.provider) ?? '',
    model: str(row.model) ?? '',
    name: str(row.name) ?? str(row.model) ?? '',
    sub: str(row.sub) ?? str(row.provider) ?? '',
    calls: num(row.calls),
    errors: num(row.errors),
    input: num(row.input),
    output: num(row.output),
    cacheRead: num(row.cache_read),
    cost: toDisplay(rate, row.cost),
  }))
}


/** 1_234_567 → 1.23M. The ledger prints raw counters; a report does not need every digit. */
function compact(value) {
  const n = num(value)
  const abs = Math.abs(n)
  if (abs >= 1e9) return `${(n / 1e9).toFixed(abs >= 1e10 ? 1 : 2)}B`
  if (abs >= 1e6) return `${(n / 1e6).toFixed(abs >= 1e7 ? 1 : 2)}M`
  if (abs >= 1e3) return `${(n / 1e3).toFixed(abs >= 1e4 ? 0 : 1)}K`
  return String(Math.round(n))
}

function money(value, symbol = '$') {
  const n = num(value)
  if (n >= 1000) return `${symbol}${Math.round(n)}`
  if (n >= 1) return `${symbol}${n.toFixed(2)}`
  return `${symbol}${n.toFixed(4)}`
}

function elapsed(ms) {
  const n = num(ms)
  if (n <= 0) return '—'
  if (n < 1000) return `${Math.round(n)}ms`
  if (n < 60000) return `${(n / 1000).toFixed(1)}s`
  return `${Math.floor(n / 60000)}m${String(Math.round((n % 60000) / 1000)).padStart(2, '0')}s`
}

function at(iso) {
  if (typeof iso !== 'string' || iso === '') return '—'
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return '—'
  const pad = (n) => String(n).padStart(2, '0')
  return `${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
}

/**
 * The model's view of one snapshot: the same numbers the panel draws, as text.
 * Ordered by what the question "how much am I using" needs first — spend, then
 * the windows that will run out, then the models and requests behind it.
 */
function report(value, recent) {
  const usage = value.usage ?? {}
  const currency = value.currency ?? { code: 'USD', symbol: '$', rate: 1, base: 'USD' }
  const cash = (amount) => money(amount, currency.symbol)
  const converted = currency.code === 'CNY' ? `（成本按 1 USD = ${currency.symbol}${currency.rate} 折算）` : ''
  const lines = []
  lines.push(`Magpie 用量 · ${value.period}${usage.since === '' ? '' : `（自 ${usage.since}）`} · 读取于 ${at(value.fetchedAt)}${converted}`)
  lines.push(`调用 ${compact(usage.calls)}（错误 ${compact(usage.errors)}${num(usage.retried) > 0 ? `，重试 ${compact(usage.retried)}` : ''}）· 成本 ${cash(usage.cost)}${num(usage.unpriced) > 0 ? ` · ${compact(usage.unpriced)} 条未计价` : ''}`)
  lines.push(`输入 ${compact(usage.input)} · 输出 ${compact(usage.output)} · 缓存读取 ${compact(usage.cacheRead)} · 缓存写入 ${compact(usage.cacheWrite)} · 推理 ${compact(usage.reasoning)}`)
  if (usage.avgTtftMs !== null && usage.avgTtftMs !== undefined) {
    const speed = usage.avgTokensPerSecond === null || usage.avgTokensPerSecond === undefined
      ? ''
      : ` · 解码 ${Number(usage.avgTokensPerSecond).toFixed(1)} tok/s`
    lines.push(`平均首字 ${elapsed(usage.avgTtftMs)}${speed}`)
  }
  if (value.statusline !== '') lines.push(`状态行：${value.statusline}`)

  const quotas = (value.quotas ?? []).filter((quota) => quota.windows.length > 0 || quota.balance !== undefined || quota.error !== undefined)
  if (quotas.length > 0) {
    lines.push('额度：')
    for (const quota of quotas.slice(0, 10)) {
      const windows = quota.windows.map((window) => `${window.name} 已用 ${num(window.used)}%`).join(' / ')
      const parts = [windows, quota.balance === undefined ? '' : `余额 ${quota.balance}`, quota.error === undefined ? '' : `读数失败：${quota.error}`]
      lines.push(`  - ${quota.name || quota.provider}：${parts.filter(Boolean).join(' · ')}`)
    }
  }

  if ((value.models ?? []).length > 0) {
    lines.push('模型（按成本）：')
    for (const model of value.models.slice(0, 5)) {
      lines.push(`  - ${model.model || model.name}（${model.sub || model.provider}）：${compact(model.calls)} 次 · ${cash(model.cost)} · 入 ${compact(model.input)} / 出 ${compact(model.output)}`)
    }
  }

  const rows = (value.requests ?? []).slice(0, recent)
  if (rows.length > 0) {
    lines.push(`最近 ${rows.length} 条请求：`)
    for (const row of rows) {
      lines.push(`  - ${at(row.t)} ${row.agentName || row.agent} · ${row.model} · ${row.status} · ${elapsed(row.ms)} · ${row.priced === false ? '未计价' : cash(row.cost)}`)
    }
  }

  if ((value.problems ?? []).length > 0) lines.push(`读数提示：${value.problems.join('；')}`)
  return lines.join('\n')
}

export function apply(ctx, config) {
  const resolved = resolveConfig(config)
  /** @type {Map<string, { at: number, value: unknown }>} */
  const cache = new Map()

  /** `/v1/magpie/statusline` answers text/plain, everything else JSON. */
  async function readText(path, signal, budget = TIMEOUT_MS) {
    const url = new URL(path, `${resolved.gateway}/`)
    const headers = { accept: 'text/plain' }
    if (resolved.apiKey !== '') headers.authorization = `Bearer ${resolved.apiKey}`
    const response = await fetch(url, {
      headers,
      signal: AbortSignal.any([signal, AbortSignal.timeout(budget)]),
    })
    if (!response.ok) throw new Error(`${path} answered HTTP ${response.status}`)
    return await response.text()
  }

  async function readJson(path, signal, budget = TIMEOUT_MS) {
    const url = new URL(path, `${resolved.gateway}/`)
    const headers = { accept: 'application/json' }
    if (resolved.apiKey !== '') headers.authorization = `Bearer ${resolved.apiKey}`
    const response = await fetch(url, {
      headers,
      signal: AbortSignal.any([signal, AbortSignal.timeout(budget)]),
    })
    if (!response.ok) throw new Error(`${path} answered HTTP ${response.status}`)
    return await response.json()
  }

  /**
   * Read the gateway and trim it.
   *
   * `/usage` is the only required endpoint: a gateway that answers it and not the
   * others is still worth drawing, so the rest are settled independently and any
   * failure is reported beside the numbers instead of replacing them.
   */
  async function snapshot(period, recent, signal) {
    const usage = await readJson(`/v1/magpie/usage?period=${encodeURIComponent(period)}`, signal)
    const [quotas, statusline, requests] = await Promise.allSettled([
      readJson('/v1/magpie/quotas', signal, QUOTA_TIMEOUT_MS),
      readText('/v1/magpie/statusline', signal),
      readJson(`/v1/magpie/usage/requests?limit=${recent}`, signal, 12000),
    ])
    const problems = []
    if (quotas.status === 'rejected') problems.push(`额度读取失败：${describe(quotas.reason)}`)
    if (statusline.status === 'rejected') problems.push(`状态行读取失败：${describe(statusline.reason)}`)
    if (requests.status === 'rejected') problems.push(`最近请求读取失败：${describe(requests.reason)}`)
    return {
      ok: true,
      gateway: resolved.gateway,
      fetchedAt: new Date().toISOString(),
      period,
      defaults: { period: resolved.period, refreshSeconds: resolved.refreshSeconds },
      currency: resolved.currency,
      usage: trimUsage(usage ?? {}, resolved.currency.rate),
      series: trimSeries(usage?.series, resolved.currency.rate),
      quotas: quotas.status === 'fulfilled' ? trimQuotas(quotas.value) : [],
      statusline: statusline.status === 'fulfilled' ? String(statusline.value ?? '').trim() : '',
      models: trimModels(usage ?? {}, 6, resolved.currency.rate),
      requests: requests.status === 'fulfilled' ? trimRequests(requests.value, recent, resolved.currency.rate) : [],
      problems,
    }
  }

  function cachedSnapshot(period, recent, signal, fresh) {
    const key = `${period}|${recent}`
    if (fresh) cache.delete(key)
    const hit = cache.get(key)
    if (hit !== undefined && Date.now() - hit.at < CACHE_MS) return Promise.resolve(hit.value)
    return snapshot(period, recent, signal).then((value) => {
      cache.set(key, { at: Date.now(), value })
      if (cache.size > 8) {
        const oldest = [...cache.entries()].sort((a, b) => a[1].at - b[1].at)[0]
        if (oldest !== undefined) cache.delete(oldest[0])
      }
      return value
    })
  }

  function describe(error) {
    const message = error instanceof Error ? error.message : String(error)
    return message === '' ? '未知原因' : message
  }

  function parsePeriod(request) {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const asked = url.searchParams.get('period')
      return PERIODS.has(asked) ? asked : resolved.period
    } catch {
      return resolved.period
    }
  }

  function parseRecent(request, fallback) {
    try {
      const url = new URL(request.url ?? '/', 'http://127.0.0.1')
      const raw = Number.parseInt(url.searchParams.get('recent') ?? '', 10)
      return Number.isFinite(raw) ? Math.min(Math.max(raw, 1), 50) : fallback
    } catch {
      return fallback
    }
  }

  ctx.inject(['webServer'], (webCtx) => {
    webCtx.effect(() => {
      const snapshotRoute = {
        kind: 'exact',
        path: `${ROUTE_BASE}/snapshot`,
        handler: async (request, response) => {
          if (request.method !== 'GET') {
            response.setHeader('allow', 'GET')
            sendJson(response, 405, { ok: false, reason: 'method-not-allowed' })
            return
          }
          const controller = new AbortController()
          request.on('close', () => controller.abort())
          let fresh = false
          try {
            fresh = new URL(request.url ?? '/', 'http://127.0.0.1').searchParams.get('fresh') === '1'
          } catch {
            fresh = false
          }
          try {
            const body = await cachedSnapshot(parsePeriod(request), parseRecent(request, 12), controller.signal, fresh)
            sendJson(response, 200, body)
          } catch (error) {
            sendJson(response, 502, {
              ok: false,
              reason: controller.signal.aborted ? 'aborted' : 'gateway-unreachable',
              gateway: resolved.gateway,
              error: describe(error),
              fetchedAt: new Date().toISOString(),
            })
          }
        },
      }

      const healthRoute = {
        kind: 'exact',
        path: `${ROUTE_BASE}/health`,
        handler: async (request, response) => {
          if (request.method !== 'GET') {
            response.setHeader('allow', 'GET')
            sendJson(response, 405, { ok: false, reason: 'method-not-allowed' })
            return
          }
          const controller = new AbortController()
          request.on('close', () => controller.abort())
          try {
            const info = await readJson('/', controller.signal)
            sendJson(response, 200, {
              ok: true,
              gateway: resolved.gateway,
              magpie: str(info?.version) ?? 'unknown',
              pid: num(info?.pid),
              models: num(info?.models),
            })
          } catch (error) {
            sendJson(response, 502, {
              ok: false,
              reason: 'gateway-unreachable',
              gateway: resolved.gateway,
              error: describe(error),
            })
          }
        },
      }

      const disposers = [snapshotRoute, healthRoute].map((route) => {
        try {
          return webCtx.webServer.register(route)
        } catch (error) {
          webCtx.logger?.warn?.(`magpie-usage: 路由 ${route.path} 注册失败：${describe(error)}`)
          return () => {}
        }
      })
      return () => {
        for (const dispose of disposers) {
          try {
            dispose?.()
          } catch {
            /* unloading must not throw */
          }
        }
      }
    }, 'magpie-usage: browser routes')
  })

  if (resolved.tool) {
    ctx.inject(['tools'], (toolCtx) => {
      toolCtx.effect(() => {
        const definition = {
          name: 'magpie_usage',
          description:
            '查本机 Magpie 网关的用量与额度：调用次数、成本（按配置汇率折算为人民币）、输入/输出/缓存 token、各 provider 的余额与 5 小时/7 天/月度窗口、最近请求。'
            + 'period 缺省取面板设置（today，按小时分桶）；recent 是最近请求条数，默认 5。',
          parameters: {
            type: 'object',
            properties: {
              period: { type: 'string', enum: ['today', '7d', '30d'], description: '统计周期，缺省 today' },
              recent: { type: 'integer', description: '最近请求条数，1–50，缺省 5' },
            },
            required: [],
            additionalProperties: false,
          },
          timeoutMs: 22000,
          isConcurrencySafe: () => true,
          // `output` is mandatory for a registered tool: the schema validates the returned
          // value and `render` is what the model reads. One lossless string keeps both
          // honest and costs the turn far fewer tokens than the raw ledger.
          output: {
            schema: { type: 'string' },
            render(_args, value) {
              return [{ type: 'text', text: typeof value === 'string' ? value : JSON.stringify(value) }]
            },
          },
          async execute(args, exec) {
            const asked = typeof args?.period === 'string' && PERIODS.has(args.period) ? args.period : resolved.period
            const recent = Number.isInteger(args?.recent) ? Math.min(Math.max(args.recent, 1), 50) : 5
            const value = await cachedSnapshot(asked, Math.max(recent, 5), exec?.signal ?? new AbortController().signal, false)
            return report(value, recent)
          },
        }
        try {
          return toolCtx.tools.register(definition)
        } catch (error) {
          ctx.logger?.warn?.(`magpie-usage: 工具注册失败：${describe(error)}`)
          return () => {}
        }
      }, 'magpie-usage: agent tool')
    })
  }
}
