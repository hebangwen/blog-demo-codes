/**
 * @local/magpie-usage — client half (browser bundle).
 *
 * One sidebar entry (`magpie-usage`) and the main panel it opens. The panel reads
 * the Host route `api/magpie-usage/snapshot` — a document-relative path, because
 * the GUI is served under `<base href="./">` — and re-reads it on a timer while it
 * is open and the page is visible.
 *
 * Constraints this file obeys on purpose: only `react` from the platform module
 * table, colors from `--dsw-alias-*` theme tokens only, one stylesheet injected by
 * `apply` and removed on unload, and an error boundary around the panel so a bad
 * snapshot shows a message instead of retiring the slot entry.
 *
 * @module @local/magpie-usage/client
 */

window.__ModuleLoader__.load({
  id: '@local/magpie-usage',

  factory(require) {
    const React = require('react')
    const h = React.createElement
    const { useState, useEffect, useRef, useCallback } = React

    /** Sidebar entry id and main panel key must match: that is how the shell links them. */
    const SLOT_ID = 'magpie-usage'
    const LOCALE_NS = 'magpie-usage'
    /** Route prefix, relative to the document base. */
    const API = 'api/magpie-usage'
    const PERIODS = ['today', '7d', '30d']

    // ------------------------------------------------------------------ 文案

    const zh = {
      'entry.label': 'Magpie 用量',
      'panel.title': 'Magpie 用量',
      'panel.subtitle': '本机 Magpie 网关的账本：调用、成本、token、额度窗口与最近请求',
      'action.refresh': '刷新',
      'action.retrying': '读取中…',
      'auto.label': '自动刷新',
      'auto.off': '关闭',
      'auto.seconds': '每 {n} 秒',
      'period.today': '今天',
      'period.7d': '7 天',
      'period.30d': '30 天',
      'stat.calls': '调用',
      'stat.cost': '成本',
      'stat.input': '输入 token',
      'stat.output': '输出 token',
      'stat.cache': '缓存读取',
      'stat.sub.errors': '{n} 次错误',
      'stat.sub.retried': '重试 {n}',
      'stat.sub.cacheHit': '缓存命中 {n}',
      'stat.sub.avgTtft': '首字 {n}',
      'stat.sub.speed': '解码 {n} tok/s',
      'stat.sub.cacheWrite': '写入 {n}',
      'cost.rate': '成本按 1 USD = ¥{rate} 折算',
      'chart.title': '成本分布',
      'chart.title.today': '今天每小时成本',
      'chart.title.day': '每日成本',
      'chart.note': '峰值 {peak} · 合计 {total}',
      'chart.empty': '这个周期还没有产生调用。',
      'quota.title': '额度与余额',
      'quota.note': 'Magpie 自己读到的窗口，只读',
      'quota.empty': '网关没有报出可显示的额度。',
      'quota.balance': '余额',
      'quota.readAt': '读取于 {t}',
      'quota.reset': '{t} 后重置',
      'quota.noReset': '没有重置时间',
      'quota.error': '读数失败',
      'quota.usedLeft': '已用 {used}% · 余 {left}%',
      'models.title': '模型消耗',
      'models.note': '按成本排序，前 6 个',
      'models.empty': '没有按模型的账目。',
      'col.model': '模型',
      'col.provider': 'provider',
      'col.calls': '调用',
      'col.cost': '成本',
      'col.tokens': '输入 / 输出',
      'requests.title': '最近请求',
      'requests.note': '网关最新落账的 12 条',
      'requests.empty': '没有请求记录。',
      'col.time': '时间',
      'col.agent': '来源',
      'col.ms': '耗时',
      'col.status': '状态',
      'time.updated': '更新于 {t}',
      'time.never': '还没读到',
      'time.justNow': '刚刚',
      'banner.stale': '上次读取失败，下面是最近一次成功的数据（{t}）',
      'banner.fail': '读不到 Magpie 网关：{message}',
      'banner.gateway': '地址 {gateway}',
      'loading': '正在读取网关…',
      'boundary.error': '面板渲染出错：{message}',
      'empty.value': '—',
    }

    const en = {
      'entry.label': 'Magpie Usage',
      'panel.title': 'Magpie Usage',
      'panel.subtitle': 'The ledger of the local Magpie gateway: calls, cost, tokens, quota windows and recent requests',
      'action.refresh': 'Refresh',
      'action.retrying': 'Reading…',
      'auto.label': 'Auto',
      'auto.off': 'Off',
      'auto.seconds': 'every {n}s',
      'period.today': 'Today',
      'period.7d': '7 days',
      'period.30d': '30 days',
      'stat.calls': 'Calls',
      'stat.cost': 'Cost',
      'stat.input': 'Input tokens',
      'stat.output': 'Output tokens',
      'stat.cache': 'Cache read',
      'stat.sub.errors': '{n} errors',
      'stat.sub.retried': '{n} retried',
      'stat.sub.cacheHit': '{n} cache hit',
      'stat.sub.avgTtft': '{n} to first token',
      'stat.sub.speed': '{n} tok/s decode',
      'stat.sub.cacheWrite': '{n} written',
      'cost.rate': 'converted at 1 USD = ¥{rate}',
      'chart.title': 'Cost',
      'chart.title.today': 'Cost by hour today',
      'chart.title.day': 'Daily cost',
      'chart.note': 'peak {peak} · total {total}',
      'chart.empty': 'No calls in this period yet.',
      'quota.title': 'Quota and balance',
      'quota.note': "Windows as Magpie read them, read-only",
      'quota.empty': 'The gateway reported no quota to show.',
      'quota.balance': 'Balance',
      'quota.readAt': 'read {t}',
      'quota.reset': 'resets in {t}',
      'quota.noReset': 'no reset time',
      'quota.error': 'read failed',
      'quota.usedLeft': '{used}% used · {left}% left',
      'models.title': 'Spend by model',
      'models.note': 'sorted by cost, top 6',
      'models.empty': 'No per-model ledger.',
      'col.model': 'Model',
      'col.provider': 'provider',
      'col.calls': 'Calls',
      'col.cost': 'Cost',
      'col.tokens': 'In / out',
      'requests.title': 'Recent requests',
      'requests.note': 'the newest 12 rows the gateway wrote',
      'requests.empty': 'No requests recorded.',
      'col.time': 'Time',
      'col.agent': 'Source',
      'col.ms': 'Took',
      'col.status': 'Status',
      'time.updated': 'updated {t}',
      'time.never': 'not read yet',
      'time.justNow': 'just now',
      'banner.stale': 'The last read failed; below is the newest successful snapshot ({t})',
      'banner.fail': 'Cannot reach the Magpie gateway: {message}',
      'banner.gateway': 'address {gateway}',
      'loading': 'Reading the gateway…',
      'boundary.error': 'The panel failed to render: {message}',
      'empty.value': '—',
    }

    const DICTS = { en, zh }

    // ------------------------------------------------------------------ 工具

    function resolveLang(ctx) {
      try {
        const snapshot = ctx.locale?.getLocale?.() ?? ctx.locale?.getSnapshot?.()
        const raw = snapshot?.active ?? snapshot?.locale ?? snapshot?.id ?? snapshot?.language ?? ''
        const text = typeof raw === 'string' ? raw.toLowerCase() : ''
        if (text.startsWith('zh')) return 'zh'
        if (text.startsWith('en')) return 'en'
      } catch {
        /* fall through */
      }
      return 'zh'
    }

    function translate(lang, key, params) {
      const dict = DICTS[lang] ?? zh
      const template = dict[key] ?? zh[key] ?? key
      if (params === undefined) return template
      return template.replace(/\{(\w+)\}/g, (whole, name) => {
        const value = params[name]
        return value === undefined || value === null ? whole : String(value)
      })
    }

    /** 1234567 → 1.23M. Counters here run to billions; the exact digits are noise. */
    function compact(value) {
      const n = Number(value)
      if (!Number.isFinite(n) || n === 0) return '0'
      const abs = Math.abs(n)
      if (abs >= 1e9) return `${(n / 1e9).toFixed(abs >= 1e10 ? 1 : 2)}B`
      if (abs >= 1e6) return `${(n / 1e6).toFixed(abs >= 1e7 ? 1 : 2)}M`
      if (abs >= 1e4) return `${(n / 1e3).toFixed(0)}K`
      if (abs >= 1e3) return `${(n / 1e3).toFixed(1)}K`
      return String(Math.round(n))
    }

    function money(value, symbol = '$') {
      const n = Number(value)
      if (!Number.isFinite(n)) return '—'
      if (n === 0) return `${symbol}0.00`
      if (Math.abs(n) >= 1000) return `${symbol}${Math.round(n).toLocaleString('en-US')}`
      if (Math.abs(n) >= 1) return `${symbol}${n.toFixed(2)}`
      return `${symbol}${n.toFixed(n >= 0.01 ? 3 : 4)}`
    }

    function clock(iso) {
      if (typeof iso !== 'string' || iso === '') return ''
      const date = new Date(iso)
      if (Number.isNaN(date.getTime())) return ''
      const pad = (n) => String(n).padStart(2, '0')
      return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`
    }

    function duration(ms) {
      const n = Number(ms)
      if (!Number.isFinite(n) || n <= 0) return '—'
      if (n < 1000) return `${Math.round(n)}ms`
      if (n < 60000) return `${(n / 1000).toFixed(n < 10000 ? 1 : 0)}s`
      const minutes = Math.floor(n / 60000)
      const seconds = Math.round((n % 60000) / 1000)
      return `${minutes}m${String(seconds).padStart(2, '0')}s`
    }

    function until(iso) {
      if (typeof iso !== 'string' || iso === '') return null
      const date = new Date(iso)
      if (Number.isNaN(date.getTime())) return null
      const delta = date.getTime() - Date.now()
      if (delta <= 0) return '0m'
      const minutes = Math.ceil(delta / 60000)
      if (minutes < 60) return `${minutes}m`
      const hours = Math.floor(minutes / 60)
      if (hours < 24) return `${hours}h${minutes % 60 === 0 ? '' : String(minutes % 60).padStart(2, '0') + 'm'}`
      return `${Math.floor(hours / 24)}d${hours % 24 === 0 ? '' : String(hours % 24) + 'h'}`
    }

    function percentOf(used, remaining) {
      const total = Number(used) + Number(remaining)
      if (!Number.isFinite(total) || total <= 0) return null
      return Math.min(100, Math.max(0, (Number(used) / total) * 100))
    }

    function shortAgent(name) {
      const text = typeof name === 'string' ? name : ''
      return text === '' ? '—' : text
    }

    // ------------------------------------------------------------------ 样式

    const CSS = `
.mu-root {
  /* One knob per dimension of "how much room is there": the panel measures itself and
     switches --mu-* through data-density, so a short window compresses the chart, the
     spacing and the row counts instead of pushing everything below the fold. */
  --mu-pad-y: 26px;
  --mu-pad-x: 28px;
  --mu-chart: 104px;
  --mu-stat: 22px;
  --mu-gap: 10px;
  --mu-section: 24px;
  display: flex;
  flex-direction: column;
  height: 100%;
  min-height: 0;
  min-width: 0;
  overflow: auto;
  background: var(--dsw-alias-bg-base);
  color: var(--dsw-alias-label-primary);
}
.mu-root[data-density="tight"] {
  --mu-pad-y: 18px;
  --mu-chart: 74px;
  --mu-stat: 19px;
  --mu-gap: 8px;
  --mu-section: 16px;
}
.mu-root[data-density="short"] {
  --mu-pad-y: 12px;
  --mu-chart: 50px;
  --mu-stat: 17px;
  --mu-gap: 7px;
  --mu-section: 12px;
}
.mu-inner {
  width: 100%; max-width: 1080px; box-sizing: border-box; min-width: 0; min-height: 0;
  margin: 0 auto; padding: var(--mu-pad-y) var(--mu-pad-x) var(--mu-pad-y);
}
.mu-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; flex-wrap: wrap; }
.mu-title { margin: 0; font-size: 20px; font-weight: 600; letter-spacing: .01em; }
.mu-sub { margin: 6px 0 0; font-size: 12.5px; line-height: 1.55; color: var(--dsw-alias-label-secondary); max-width: 62ch; }
/* A short window keeps the numbers, drops the prose. */
.mu-root[data-density="short"] .mu-sub { display: none; }
.mu-root[data-density="short"] .mu-statusline { margin-top: 4px; font-size: 11px; }
.mu-statusline {
  margin: 8px 0 0; font-size: 12px; color: var(--dsw-alias-label-secondary);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums;
  white-space: pre-wrap; word-break: break-word;
}
.mu-controls { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.mu-seg {
  display: inline-flex; border: 1px solid var(--dsw-alias-border-l1); border-radius: 8px;
  overflow: hidden; background: var(--dsw-alias-bg-layer-1);
}
.mu-seg button {
  appearance: none; border: 0; background: transparent; cursor: pointer;
  padding: 6px 11px; font: inherit; font-size: 12.5px; color: var(--dsw-alias-label-secondary);
}
.mu-seg button[data-on="true"] {
  background: var(--dsw-alias-bg-layer-2); color: var(--dsw-alias-label-primary); font-weight: 600;
}
.mu-btn {
  appearance: none; cursor: pointer; font: inherit; font-size: 12.5px;
  padding: 6px 12px; border-radius: 8px;
  border: 1px solid var(--dsw-alias-border-l1);
  background: var(--dsw-alias-bg-layer-1); color: var(--dsw-alias-label-primary);
}
.mu-select {
  appearance: auto; cursor: pointer; font: inherit; font-size: 12.5px;
  padding: 5px 8px; border-radius: 8px;
  border: 1px solid var(--dsw-alias-border-l1);
  background: var(--dsw-alias-bg-layer-1); color: var(--dsw-alias-label-primary);
}
.mu-btn:hover:not(:disabled) { border-color: var(--dsw-alias-border-l2); }
.mu-btn:disabled { opacity: .55; cursor: default; }
.mu-btn[data-busy="true"] { color: var(--dsw-alias-label-secondary); }
.mu-stamp { font-size: 12px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-banner {
  margin-top: 14px; padding: 10px 13px; border-radius: 9px; font-size: 12.5px; line-height: 1.5;
  border: 1px solid var(--dsw-alias-state-warn-primary);
  color: var(--dsw-alias-label-primary);
  background: color-mix(in oklab, var(--dsw-alias-state-warn-primary) 12%, transparent);
}
.mu-banner[data-kind="error"] {
  border-color: var(--dsw-alias-state-error-primary);
  background: color-mix(in oklab, var(--dsw-alias-state-error-primary) 12%, transparent);
}
.mu-banner code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.mu-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
  gap: var(--mu-gap); margin-top: calc(var(--mu-pad-y) * 0.7);
  /* Content-sized, not row-sized: two cards in one row do not have to end at the same baseline. */
  align-items: start;
}
.mu-card {
  background: var(--dsw-alias-bg-layer-1); border: 1px solid var(--dsw-alias-border-l1);
  border-radius: 10px; padding: 13px 15px; min-width: 0;
}
.mu-root[data-density="tight"] .mu-card { padding: 11px 13px; }
.mu-root[data-density="short"] .mu-card { padding: 9px 11px; }
.mu-stat-label { font-size: 11.5px; letter-spacing: .02em; color: var(--dsw-alias-label-secondary); }
.mu-stat-value { margin-top: 5px; font-size: var(--mu-stat); font-weight: 600; font-variant-numeric: tabular-nums; line-height: 1.15; }
.mu-stat-sub { margin-top: 5px; font-size: 11.5px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-section { margin-top: var(--mu-section); }
.mu-section-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 10px; }
.mu-section-title { font-size: 13.5px; font-weight: 600; letter-spacing: .02em; }
.mu-section-note { font-size: 11.5px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-chart-card {
  background: var(--dsw-alias-bg-layer-1); border: 1px solid var(--dsw-alias-border-l1);
  border-radius: 10px; padding: 14px 16px 10px;
}
.mu-bars { display: flex; align-items: flex-end; gap: 2px; height: var(--mu-chart); border-bottom: 1px solid var(--dsw-alias-border-l1); }
.mu-root[data-density="short"] .mu-chart-card { padding: 9px 11px 7px; }
.mu-bar { flex: 1 1 0; min-width: 0; background: var(--dsw-alias-brand-primary); border-radius: 2px 2px 0 0; opacity: .78; }
.mu-bar[data-zero="true"] { background: var(--dsw-alias-border-l1); opacity: .5; }
.mu-bar:hover { opacity: 1; }
.mu-axis { display: flex; justify-content: space-between; margin-top: 6px; font-size: 11px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-quota-grid {
  display: grid; grid-template-columns: repeat(auto-fit, minmax(252px, 1fr)); gap: 10px;
  /* A provider with three windows is taller than one with none; let it be. */
  align-items: start;
}
.mu-quota-name { font-size: 13px; font-weight: 600; }
.mu-quota-plan { margin-top: 3px; font-size: 11.5px; color: var(--dsw-alias-label-secondary); }
.mu-quota-balance { margin-top: 8px; font-size: 15px; font-weight: 600; font-variant-numeric: tabular-nums; }
.mu-quota-error { margin-top: 8px; font-size: 11.5px; color: var(--dsw-alias-state-warn-primary); line-height: 1.45; word-break: break-word; }
.mu-window { margin-top: 11px; }
.mu-window-head { display: flex; justify-content: space-between; gap: 8px; font-size: 11.5px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-window-head b { color: var(--dsw-alias-label-primary); font-weight: 600; }
.mu-track { height: 6px; margin-top: 5px; border-radius: 3px; background: var(--dsw-alias-bg-layer-2); overflow: hidden; }
.mu-fill { height: 100%; border-radius: 3px; background: var(--dsw-alias-state-success-primary); }
.mu-fill[data-level="warn"] { background: var(--dsw-alias-state-warn-primary); }
.mu-fill[data-level="high"] { background: var(--dsw-alias-state-error-primary); }
.mu-table-card {
  background: var(--dsw-alias-bg-layer-1); border: 1px solid var(--dsw-alias-border-l1);
  border-radius: 10px; overflow: auto;
}
.mu-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.mu-table th {
  text-align: left; font-weight: 500; font-size: 11.5px; color: var(--dsw-alias-label-secondary);
  padding: 9px 12px; border-bottom: 1px solid var(--dsw-alias-border-l1); white-space: nowrap;
}
.mu-table td {
  padding: 8px 12px; border-bottom: 1px solid var(--dsw-alias-border-l1);
  font-variant-numeric: tabular-nums; white-space: nowrap;
}
.mu-table tr:last-child td { border-bottom: 0; }
.mu-table td[data-num="true"], .mu-table th[data-num="true"] { text-align: right; }
.mu-table td[data-mono="true"] { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.mu-table td[data-cut="true"] { max-width: 240px; overflow: hidden; text-overflow: ellipsis; }
/* Wrapping cells make the row as tall as its own content instead of one uniform strip. */
.mu-table td[data-wrap="true"] {
  white-space: normal; word-break: break-word; max-width: 220px; line-height: 1.35;
}
.mu-table tbody tr:hover td { background: var(--dsw-alias-bg-layer-2); }
.mu-status { font-weight: 600; }
.mu-status[data-bad="true"] { color: var(--dsw-alias-state-error-primary); }
.mu-status[data-bad="false"] { color: var(--dsw-alias-state-success-primary); }
.mu-empty {
  padding: 20px; text-align: center; font-size: 12.5px; color: var(--dsw-alias-label-secondary);
  border: 1px dashed var(--dsw-alias-border-l1); border-radius: 10px;
}
.mu-footer { margin-top: var(--mu-section); font-size: 11.5px; color: var(--dsw-alias-label-secondary); font-variant-numeric: tabular-nums; }
.mu-root[data-density="short"] .mu-col-optional { display: none; }
.mu-root[data-density="short"] .mu-table th,
.mu-root[data-density="short"] .mu-table td { padding: 5px 9px; }
.mu-root[data-density="short"] .mu-table { font-size: 12px; }
.mu-root[data-density="tight"] .mu-table th,
.mu-root[data-density="tight"] .mu-table td { padding: 7px 10px; }
@media (max-width: 720px) {
  .mu-inner { padding: 20px 16px 48px; }
  .mu-stat-value { font-size: 19px; }
  .mu-table th, .mu-table td { padding: 7px 9px; }
}
`

    function installStyles() {
      if (typeof document === 'undefined') return () => {}
      const stale = document.querySelectorAll(`style[data-plugin-css="${SLOT_ID}"]`)
      for (const tag of stale) {
        if (tag.parentNode !== null && tag.parentNode !== undefined) tag.parentNode.removeChild(tag)
      }
      const tag = document.createElement('style')
      tag.setAttribute('data-plugin-css', SLOT_ID)
      tag.textContent = CSS
      document.head.appendChild(tag)
      return () => {
        if (tag.parentNode !== null && tag.parentNode !== undefined) tag.parentNode.removeChild(tag)
      }
    }

    // ------------------------------------------------------------------ 图标

    function Glyph(props) {
      const size = props?.size ?? 16
      return h(
        'svg',
        { width: size, height: size, viewBox: '0 0 16 16', fill: 'none', 'aria-hidden': 'true' },
        h('rect', { x: 2.1, y: 9.3, width: 2.6, height: 4.6, rx: 0.9, fill: 'currentColor', opacity: 0.95 }),
        h('rect', { x: 6.1, y: 6.4, width: 2.6, height: 7.5, rx: 0.9, fill: 'currentColor', opacity: 0.8 }),
        h('rect', { x: 10.1, y: 2.4, width: 2.6, height: 11.5, rx: 0.9, fill: 'currentColor', opacity: 0.6 }),
        h('path', {
          d: 'M2.4 7.1 6.4 4.6l4.1-1.1 1.9.9',
          stroke: 'currentColor',
          strokeWidth: 1.1,
          strokeLinecap: 'round',
          strokeLinejoin: 'round',
          opacity: 0.55,
        }),
      )
    }

    // ------------------------------------------------------------------ 组件

    function Stat(props) {
      return h(
        'div',
        { className: 'mu-card' },
        h('div', { className: 'mu-stat-label' }, props.label),
        h('div', { className: 'mu-stat-value' }, props.value),
        props.sub === undefined || props.sub === null || props.sub === ''
          ? null
          : h('div', { className: 'mu-stat-sub' }, props.sub),
      )
    }

    /** `today` buckets are hours ("00".."23"); the rest are day labels. */
    function axisLabel(label, hourly) {
      const text = typeof label === 'string' ? label : ''
      if (!hourly) return text
      return text.length === 2 ? `${text}:00` : text
    }

    function CostChart(props) {
      const { series, t, hourly, symbol } = props
      const rows = Array.isArray(series) ? series : []
      if (rows.length === 0) return h('div', { className: 'mu-empty' }, t('chart.empty'))
      let peak = 0
      let total = 0
      for (const row of rows) {
        const cost = Number(row.cost) || 0
        if (cost > peak) peak = cost
        total += cost
      }
      const scale = peak > 0 ? peak : 1
      return h(
        'div',
        { className: 'mu-chart-card' },
        h(
          'div',
          { className: 'mu-bars' },
          rows.map((row, index) => {
            const cost = Number(row.cost) || 0
            // Percent, not pixels: the bar area's height comes from --mu-chart, and the
            // density switch must not need a matching number here.
            const height = cost <= 0 ? '2px' : `${Math.max(3, (cost / scale) * 100).toFixed(2)}%`
            return h('div', {
              key: `${row.time}-${index}`,
              className: 'mu-bar',
              'data-zero': cost <= 0 ? 'true' : 'false',
              style: { height },
              title: `${axisLabel(row.label, hourly)}: ${money(cost, symbol)} · ${compact(row.calls)} calls`,
            })
          }),
        ),
        h(
          'div',
          { className: 'mu-axis' },
          h('span', null, axisLabel(rows[0]?.label, hourly)),
          h('span', null, money(peak, symbol)),
          h('span', null, axisLabel(rows[rows.length - 1]?.label, hourly)),
        ),
      )
    }

    function QuotaCard(props) {
      const { quota, t } = props
      const windows = Array.isArray(quota.windows) ? quota.windows : []
      const readAt = clock(quota.readAt)
      return h(
        'div',
        { className: 'mu-card' },
        h('div', { className: 'mu-quota-name' }, quota.name || quota.provider || '—'),
        quota.plan === undefined || quota.plan === ''
          ? null
          : h('div', { className: 'mu-quota-plan' }, quota.plan),
        quota.user === undefined || quota.user === ''
          ? null
          : h('div', { className: 'mu-quota-plan' }, quota.user),
        quota.balance === undefined || quota.balance === ''
          ? null
          : h(
              'div',
              { className: 'mu-quota-balance' },
              quota.balance,
              readAt === '' ? null : h('span', { className: 'mu-section-note' }, ` · ${t('quota.readAt', { t: readAt })}`),
            ),
        quota.error === undefined || quota.error === ''
          ? null
          : h('div', { className: 'mu-quota-error' }, `${t('quota.error')}: ${quota.error}`),
        windows.map((window, index) => {
          const used = percentOf(window.used, window.remaining)
          const level = used === null ? 'ok' : used >= 90 ? 'high' : used >= 70 ? 'warn' : 'ok'
          const reset = until(window.resetsAt)
          return h(
            'div',
            { className: 'mu-window', key: `${window.name}-${index}` },
            h(
              'div',
              { className: 'mu-window-head' },
              h('b', null, window.name || '—'),
              h(
                'span',
                null,
                used === null
                  ? `余 ${compact(window.remaining)}`
                  : t('quota.usedLeft', { used: Math.round(used), left: Math.round(100 - used) }),
              ),
            ),
            h(
              'div',
              { className: 'mu-track' },
              h('div', {
                className: 'mu-fill',
                'data-level': level,
                style: { width: `${used === null ? 0 : used}%` },
              }),
            ),
            h(
              'div',
              { className: 'mu-window-head', style: { marginTop: '4px' } },
              h('span', null, reset === null ? t('quota.noReset') : t('quota.reset', { t: reset })),
              h('span', null, window.resetsAt === undefined ? '' : clock(window.resetsAt)),
            ),
          )
        }),
      )
    }

    function ModelsTable(props) {
      const { models, t, symbol } = props
      const rows = Array.isArray(models) ? models : []
      if (rows.length === 0) return h('div', { className: 'mu-empty' }, t('models.empty'))
      return h(
        'div',
        { className: 'mu-table-card' },
        h(
          'table',
          { className: 'mu-table' },
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              h('th', null, t('col.model')),
              h('th', { className: 'mu-col-optional' }, t('col.provider')),
              h('th', { 'data-num': 'true' }, t('col.calls')),
              h('th', { 'data-num': 'true' }, t('col.tokens')),
              h('th', { 'data-num': 'true' }, t('col.cost')),
            ),
          ),
          h(
            'tbody',
            null,
            rows.map((row) => {
              const model = row.name || row.model || row.id || '—'
              const label = row.model !== undefined && row.model !== '' ? row.model : model
              return h(
                'tr',
                { key: row.id || `${row.provider}-${model}` },
                h('td', { 'data-wrap': 'true', title: row.id || model }, label),
                h('td', { className: 'mu-col-optional' }, row.sub || row.provider || '—'),
                h('td', { 'data-num': 'true' }, compact(row.calls)),
                h('td', { 'data-num': 'true' }, `${compact(row.input)} / ${compact(row.output)}`),
                h('td', { 'data-num': 'true' }, money(row.cost, symbol)),
              )
            }),
          ),
        ),
      )
    }

    function RequestsTable(props) {
      const { requests, t, symbol } = props
      const rows = Array.isArray(requests) ? requests : []
      if (rows.length === 0) return h('div', { className: 'mu-empty' }, t('requests.empty'))
      return h(
        'div',
        { className: 'mu-table-card' },
        h(
          'table',
          { className: 'mu-table' },
          h(
            'thead',
            null,
            h(
              'tr',
              null,
              h('th', null, t('col.time')),
              h('th', { className: 'mu-col-optional' }, t('col.agent')),
              h('th', null, t('col.model')),
              h('th', { 'data-num': 'true' }, 'in / out'),
              h('th', { 'data-num': 'true' }, t('col.ms')),
              h('th', { 'data-num': 'true' }, t('col.cost')),
              h('th', { 'data-num': 'true' }, t('col.status')),
            ),
          ),
          h(
            'tbody',
            null,
            rows.map((row, index) => {
              const ok = row.status === 200
              return h(
                'tr',
                { key: `${row.t}-${index}` },
                h('td', { 'data-mono': 'true' }, clock(row.t) || '—'),
                h('td', { className: 'mu-col-optional' }, shortAgent(row.agentName)),
                h('td', { 'data-wrap': 'true', title: row.req || row.model }, row.model || '—'),
                h('td', { 'data-num': 'true' }, `${compact(row.in)} / ${compact(row.out)}`),
                h('td', { 'data-num': 'true' }, duration(row.ms)),
                h('td', { 'data-num': 'true' }, row.priced === false ? '—' : money(row.cost, symbol)),
                h('td', { 'data-num': 'true' }, h('span', { className: 'mu-status', 'data-bad': ok ? 'false' : 'true' }, String(row.status))),
              )
            }),
          ),
        ),
      )
    }

    // ------------------------------------------------------------------ 面板

    /** How many rows each short window can still afford to show. */
    const ROW_LIMITS = {
      normal: { requests: 12, models: 6 },
      tight: { requests: 8, models: 5 },
      short: { requests: 5, models: 4 },
    }

    /**
     * Measure the panel and pick a density tier.
     *
     * The slot's height is the only trustworthy number: the same panel is a full column
     * on a desktop window and a narrower one beside a session, and a `height: 100%` slot
     * that resolves to auto must fall back to the viewport rather than to `0`.
     */
    function useDensity() {
      const ref = useRef(null)
      const [density, setDensity] = useState('normal')
      useEffect(() => {
        const node = ref.current
        if (node === null || node === undefined) return undefined
        const measure = () => {
          const own = node.getBoundingClientRect().height
          const viewport = typeof window === 'undefined' ? 900 : window.innerHeight
          const height = own > 0 ? Math.min(own, viewport) : viewport
          setDensity(height < 520 ? 'short' : height < 780 ? 'tight' : 'normal')
        }
        measure()
        let observer
        if (typeof ResizeObserver !== 'undefined') {
          observer = new ResizeObserver(measure)
          observer.observe(node)
        }
        if (typeof window !== 'undefined') window.addEventListener('resize', measure)
        return () => {
          try {
            observer?.disconnect()
          } catch {
            /* already gone */
          }
          if (typeof window !== 'undefined') window.removeEventListener('resize', measure)
        }
      }, [])
      return [ref, density]
    }

    function Panel(props) {
      const ctx = props.ctx
      const [rootRef, density] = useDensity()
      const limits = ROW_LIMITS[density] ?? ROW_LIMITS.normal
      const lang = useLang(ctx)
      const t = useCallback((key, params) => translate(lang, key, params), [lang])
      const [period, setPeriod] = useState(PERIODS.includes(props.period) ? props.period : 'today')
      const [auto, setAuto] = useState(typeof props.autoSeconds === 'number' ? props.autoSeconds : 10)
      const [data, setData] = useState(null)
      const [failure, setFailure] = useState(null)
      const [busy, setBusy] = useState(true)
      const abortRef = useRef(null)
      const aliveRef = useRef(true)
      /** Once the reader picks a period or an interval, the deployment defaults stop steering. */
      const touchedRef = useRef(props.period !== undefined || props.autoSeconds !== undefined)

      const load = useCallback(
        async (fresh) => {
          if (typeof document !== 'undefined' && document.visibilityState === 'hidden') return
          try {
            abortRef.current?.abort()
          } catch {
            /* replaced below */
          }
          const controller = new AbortController()
          abortRef.current = controller
          // A background poll must not flip the header into "reading…"; only an explicit
          // refresh (or the very first read) owns that state.
          if (fresh) setBusy(true)
          try {
            const query = `${API}/snapshot?period=${encodeURIComponent(period)}&recent=12${fresh ? '&fresh=1' : ''}`
            const response = await fetch(query, { signal: controller.signal, headers: { accept: 'application/json' } })
            const body = await response.json()
            if (!aliveRef.current) return
            if (response.ok && body?.ok === true) {
              setData(body)
              setFailure(null)
              // The panel opens on whatever period the deployment configured; adopt it once,
              // unless the reader already picked one.
              if (!touchedRef.current && body.defaults !== undefined) {
                touchedRef.current = true
                if (PERIODS.includes(body.defaults.period) && body.defaults.period !== period) setPeriod(body.defaults.period)
                if (Number.isFinite(body.defaults.refreshSeconds)) {
                  const seconds = Number(body.defaults.refreshSeconds)
                  if ([5, 10, 30, 60].includes(seconds)) setAuto(seconds)
                }
              }
            } else {
              setFailure({
                message: body?.error ?? `HTTP ${response.status}`,
                gateway: body?.gateway,
              })
            }
          } catch (error) {
            if (!aliveRef.current) return
            if (error?.name === 'AbortError') return
            setFailure({ message: error?.message ?? String(error), gateway: undefined })
          } finally {
            if (fresh && aliveRef.current) setBusy(false)
          }
        },
        [period],
      )

      useEffect(() => {
        aliveRef.current = true
        void load(true)
        return () => {
          aliveRef.current = false
          try {
            abortRef.current?.abort()
          } catch {
            /* already gone */
          }
        }
      }, [load])

      useEffect(() => {
        if (!(auto > 0)) return undefined
        const timer = setInterval(() => {
          void load(false)
        }, auto * 1000)
        const onVisible = () => {
          if (typeof document !== 'undefined' && document.visibilityState === 'visible') void load(false)
        }
        if (typeof document !== 'undefined') document.addEventListener('visibilitychange', onVisible)
        return () => {
          clearInterval(timer)
          if (typeof document !== 'undefined') document.removeEventListener('visibilitychange', onVisible)
        }
      }, [auto, load])

      const usage = data?.usage ?? {}
      const series = data?.series ?? []
      const quotas = data?.quotas ?? []
      const models = (data?.models ?? []).slice(0, limits.models)
      const requests = (data?.requests ?? []).slice(0, limits.requests)
      const resultCurrency = data?.currency ?? { code: 'USD', symbol: '$', rate: 1 }
      const symbol = resultCurrency.symbol ?? '$'
      const rateNote = resultCurrency.code === 'CNY' ? t('cost.rate', { rate: resultCurrency.rate }) : ''
      const cacheBase = (usage?.input ?? 0) + (usage?.cacheRead ?? 0)
      const cacheHit = cacheBase > 0 ? ((usage?.cacheRead ?? 0) / cacheBase) * 100 : null
      const updated = data?.fetchedAt === undefined ? '' : clock(data.fetchedAt)

      const periodButtons = PERIODS.map((id) =>
        h(
          'button',
          {
            key: id,
            type: 'button',
            'data-on': id === period ? 'true' : 'false',
            onClick: () => {
              touchedRef.current = true
              setPeriod(id)
            },
          },
          t(`period.${id}`),
        ),
      )

      const autoOptions = [0, 5, 10, 30, 60].map((seconds) =>
        h(
          'option',
          { key: seconds, value: String(seconds) },
          seconds === 0 ? t('auto.off') : t('auto.seconds', { n: seconds }),
        ),
      )

      return h(
        'div',
        { className: 'mu-root', ref: rootRef, 'data-density': density },
        h(
          'div',
          { className: 'mu-inner' },
          h(
            'div',
            { className: 'mu-head' },
            h(
              'div',
              null,
              h('h1', { className: 'mu-title' }, t('panel.title')),
              h('p', { className: 'mu-sub' }, t('panel.subtitle')),
              data?.statusline !== undefined && data.statusline !== ''
                ? h('p', { className: 'mu-statusline' }, data.statusline)
                : null,
            ),
            h(
              'div',
              { className: 'mu-controls' },
              h('div', { className: 'mu-seg', role: 'group', 'aria-label': 'period' }, periodButtons),
              h(
                'select',
                {
                  className: 'mu-select',
                  'aria-label': t('auto.label'),
                  value: String(auto),
                  onChange: (event) => {
                    touchedRef.current = true
                    setAuto(Number(event.target.value))
                  },
                },
                autoOptions,
              ),
              h(
                'button',
                {
                  type: 'button',
                  className: 'mu-btn',
                  'data-busy': busy ? 'true' : 'false',
                  disabled: busy,
                  onClick: () => void load(true),
                },
                busy ? t('action.retrying') : t('action.refresh'),
              ),
              h('span', { className: 'mu-stamp' }, t('time.updated', { t: updated === '' ? t('time.never') : updated })),
            ),
          ),

          failure === null
            ? null
            : h(
                'div',
                { className: 'mu-banner', 'data-kind': data === null ? 'error' : 'warn', role: 'alert' },
                data === null
                  ? t('banner.fail', { message: failure.message })
                  : t('banner.stale', { t: clock(data.fetchedAt) }),
                failure.gateway === undefined || failure.gateway === ''
                  ? null
                  : h('div', null, h('code', null, t('banner.gateway', { gateway: failure.gateway }))),
                h('div', null, h('code', null, failure.message)),
              ),

          data === null
            ? h('div', { className: 'mu-empty', style: { marginTop: '18px' } }, t('loading'))
            : h(
                'div',
                null,
                h(
                  'div',
                  { className: 'mu-grid' },
                  h(Stat, {
                    label: t('stat.calls'),
                    value: compact(usage.calls),
                    sub: [
                      t('stat.sub.errors', { n: compact(usage.errors) }),
                      usage.retried > 0 ? t('stat.sub.retried', { n: compact(usage.retried) }) : '',
                    ]
                      .filter(Boolean)
                      .join(' · '),
                  }),
                  h(Stat, {
                    label: t('stat.cost'),
                    value: money(usage.cost, symbol),
                    sub: [usage.unpriced > 0 ? `${compact(usage.unpriced)} unpriced` : '', rateNote].filter(Boolean).join(' · '),
                  }),
                  h(Stat, {
                    label: t('stat.input'),
                    value: compact(usage.input),
                    sub: cacheHit === null ? '' : t('stat.sub.cacheHit', { n: `${cacheHit.toFixed(1)}%` }),
                  }),
                  h(Stat, {
                    label: t('stat.output'),
                    value: compact(usage.output),
                    sub: [
                      usage.reasoning > 0 ? `reasoning ${compact(usage.reasoning)}` : '',
                      usage.avgTokensPerSecond === null ? '' : t('stat.sub.speed', { n: Number(usage.avgTokensPerSecond).toFixed(1) }),
                    ]
                      .filter(Boolean)
                      .join(' · '),
                  }),
                  h(Stat, {
                    label: t('stat.cache'),
                    value: compact(usage.cacheRead),
                    sub: [
                      usage.cacheWrite > 0 ? t('stat.sub.cacheWrite', { n: compact(usage.cacheWrite) }) : '',
                      usage.avgTtftMs === null ? '' : t('stat.sub.avgTtft', { n: duration(usage.avgTtftMs) }),
                    ]
                      .filter(Boolean)
                      .join(' · '),
                  }),
                ),

                h(
                  'div',
                  { className: 'mu-section' },
                  h(
                    'div',
                    { className: 'mu-section-head' },
                    h(
                      'div',
                      { className: 'mu-section-title' },
                      usage.period === 'today' ? t('chart.title.today') : t('chart.title.day'),
                    ),
                    h(
                      'div',
                      { className: 'mu-section-note' },
                      t('chart.note', {
                        peak: money(Math.max(0, ...series.map((row) => Number(row.cost) || 0)), symbol),
                        total: money(series.reduce((sum, row) => sum + (Number(row.cost) || 0), 0), symbol),
                      }),
                    ),
                  ),
                  h(CostChart, { series, t, hourly: usage.period === 'today', symbol }),
                ),

                h(
                  'div',
                  { className: 'mu-section' },
                  h(
                    'div',
                    { className: 'mu-section-head' },
                    h('div', { className: 'mu-section-title' }, t('quota.title')),
                    h('div', { className: 'mu-section-note' }, t('quota.note')),
                  ),
                  quotas.length === 0
                    ? h('div', { className: 'mu-empty' }, t('quota.empty'))
                    : h('div', { className: 'mu-quota-grid' }, quotas.map((quota, index) => h(QuotaCard, { key: `${quota.provider}-${index}`, quota, t }))),
                ),

                h(
                  'div',
                  { className: 'mu-section' },
                  h(
                    'div',
                    { className: 'mu-section-head' },
                    h('div', { className: 'mu-section-title' }, t('models.title')),
                    h('div', { className: 'mu-section-note' }, t('models.note')),
                  ),
                  h(ModelsTable, { models, t, symbol }),
                ),

                h(
                  'div',
                  { className: 'mu-section' },
                  h(
                    'div',
                    { className: 'mu-section-head' },
                    h('div', { className: 'mu-section-title' }, t('requests.title')),
                    h('div', { className: 'mu-section-note' }, t('requests.note')),
                  ),
                  h(RequestsTable, { requests, t, symbol }),
                ),

                h(
                  'div',
                  { className: 'mu-footer' },
                  [data.gateway ?? '', data.period ?? '', rateNote, t('time.updated', { t: clock(data.fetchedAt) || t('time.never') })]
                    .filter(Boolean)
                    .join(' · '),
                ),
              ),
        ),
      )
    }

    function useLang(ctx) {
      const [lang, setLang] = useState(() => resolveLang(ctx))
      useEffect(() => {
        let dispose
        try {
          dispose = ctx.locale.subscribe(() => setLang(resolveLang(ctx)))
        } catch {
          dispose = undefined
        }
        return () => {
          try {
            dispose?.()
          } catch {
            /* the locale service may be gone at unload */
          }
        }
      }, [ctx])
      return lang
    }

    /** Keep a render crash inside the panel instead of retiring the slot entry. */
    class PanelBoundary extends React.Component {
      constructor(props) {
        super(props)
        this.state = { error: null }
        this.retry = this.retry.bind(this)
      }

      static getDerivedStateFromError(error) {
        return { error }
      }

      componentDidCatch(error) {
        try {
          console.error('[magpie-usage] 面板渲染出错', error)
        } catch {
          /* logging must never throw */
        }
      }

      retry() {
        this.setState({ error: null })
      }

      render() {
        const { error } = this.state
        if (error === null) return h(this.props.panel, this.props.panelProps)
        const lang = resolveLang(this.props.ctx)
        return h(
          'div',
          { className: 'mu-root' },
          h(
            'div',
            { className: 'mu-inner' },
            h(
              'div',
              { className: 'mu-banner', 'data-kind': 'error', role: 'alert' },
              translate(lang, 'boundary.error', { message: error?.message ?? String(error) }),
              h(
                'button',
                { type: 'button', className: 'mu-btn', style: { marginTop: '8px' }, onClick: this.retry },
                translate(lang, 'action.refresh'),
              ),
            ),
          ),
        )
      }
    }

    // ------------------------------------------------------------------ apply

    return {
      inject: ['slots', 'locale'],

      apply(ctx) {
        ctx.effect(() => ctx.locale.register(LOCALE_NS, DICTS), `${LOCALE_NS}: dictionaries`)
        ctx.effect(installStyles, `${LOCALE_NS}: styles`)

        ctx.slots.inject('sidebar.panellist', () =>
          ctx.slots.register(
            {
              name: 'sidebar.panellist',
              id: SLOT_ID,
              order: 88,
              label: () => translate(resolveLang(ctx), 'entry.label'),
              locale: LOCALE_NS,
            },
            Glyph,
          ),
        )
        // The slot hands a component the owner props and hooks, never the plugin context:
        // close over it here so the panel can reach `locale` at all.
        ctx.slots.inject('main', () =>
          ctx.slots.register({ name: 'main', key: SLOT_ID, locale: LOCALE_NS }, (props) =>
            h(PanelBoundary, { ctx, panel: Panel, panelProps: { ...props, ctx } }),
          ),
        )
      },
    }
  },
})
