# dsh-magpie-usage

在 **DeepSeek Harness**（DSH）里实时查看本机 **Magpie** 网关的用量：侧栏多一个柱状图图标，点开就是一张自动刷新的用量面板。

- 五个数字块：调用次数、成本（按汇率折成人民币）、输入 / 输出 token、缓存读取
- 成本柱状图：`今天` 按小时分桶，`7 天` / `30 天` 按天
- 额度卡片：每个 provider 的余额、5 小时 / 7 天 / 月度窗口进度条与重置倒计时
- 模型消耗 Top N、最近请求流水
- 面板会**按可用高度自动收缩**：窗口矮的时候压缩图表高度、行数和留白，而不是把内容推出视口
- 另外给 Agent 一个同名工具 `magpie_usage`，读的是同一份快照

## 为什么要有它

Magpie 的网关已经记了账（`~/.config/magpie/usage.jsonl`），也提供了只读接口，但那些接口在另一个 origin 上、不带 CORS 头，浏览器页面读不到；而「今天花了多少钱、哪个 provider 的 5 小时窗口快满了」是时不时要看一眼的东西，不该每次都去开 Magpie 自己的窗口。

这个插件做的就是：Host 侧去读网关，裁剪成一份小快照，从 Harness 自己的 origin 发出来；客户端画成面板。数据只读，不写 Magpie 的任何文件。

## 安装

插件是一个普通的 DSH bundle（`package.json` 里声明 `dsh.bundle.patch` 与 `dsh.client`），用插件管理器从本地目录装：

```
plugin_manager  action=install_bundle  target=/absolute/path/to/dsh-magpie-usage
```

它会把这个目录 link 进当前 profile、登记 bundle，并立刻生效。装完刷新一次页面（客户端 bundle 不像 Host 侧那样热更）。

要求：本机跑着 Magpie（默认网关 `http://127.0.0.1:3425`），DSH 版本支持 `webServer.register` 与 `slots`。

## 数据链路

Magpie 的账本接口（只读）：

| 端点 | 内容 |
| --- | --- |
| `GET /v1/magpie/usage?period=today\|7d\|30d` | 汇总 + 按小时 / 按天分桶 |
| `GET /v1/magpie/quotas` | 各 provider 余额与 5h / 7d / Month 窗口 |
| `GET /v1/magpie/statusline` | 纯文本的一行额度摘要 |
| `GET /v1/magpie/usage/requests?limit=N` | 最新落账的请求 |

Host 侧读取并裁剪后，从 Harness 同源发出：

```
GET /api/magpie-usage/snapshot?period=today&recent=12&fresh=1
GET /api/magpie-usage/health
```

面板按 `refreshSeconds` 轮询（页面隐藏时停），`fresh=1` 绕开 1.5 秒的请求合并缓存。`/quotas` 是慢读（Magpie 要挨个问 provider，某个 provider 连不上就会等到自己的超时），所以它的超时预算单独放宽，失败也不影响其余数字，只会在快照里附带一条 `problems`。

## 配置

patch 里的行 id 是 `magpie-usage`，在 profile 的 `cordis.patch.yml` 里按 id 覆盖即可（**整块替换**，要保留的键得重写）：

```yaml
- id: magpie-usage
  config:
    gateway: 'http://127.0.0.1:3425'
    apiKey: ''
    period: 'today'      # today | 7d | 30d
    refreshSeconds: 10
    cnyRate: 6.8         # 成本按此汇率折算人民币；0 = 保留 USD
    tool: true           # 是否注册 magpie_usage 工具
```

- 成本：账本以 USD 计价，所有 `cost` 在 Host 侧一次换算（`cnyRate`），面板与工具共用。
- 额度卡里的余额**不换算**：那是各 provider 自报的字符串（`¥20.45` / `$…` / `积分`），换算会算错。
- `apiKey` 留空时请求不带 `Authorization`，本机回环网关就吃这一套。

## Agent 工具

```json
{ "name": "magpie_usage", "parameters": { "period": "today|7d|30d", "recent": 5 } }
```

返回文字版：总量、平均首字与解码速度、各 provider 窗口、按成本排序的模型、最近 N 条请求。和面板读同一个 `snapshot()`，不存在两套口径。

## 高度自适应是怎么做的

面板在根节点上挂一个 `ResizeObserver`，把自己的高度分成三档（`normal` / `tight` / `short`），通过 `data-density` 驱动一组 CSS 变量（`--mu-chart`、`--mu-pad-y`、`--mu-section`…），同时按档位截断表格行数（12 / 8 / 5 条请求，6 / 5 / 4 个模型）、在 `short` 档隐藏次要列与说明文字。柱状图高度用百分比，只由 `--mu-chart` 决定，不需要第二处常数。

槽位高度若解析成 auto（`height: 100%` 的常见退化），量到的 0 会退回视口高度，所以嵌在窄栏里也不会突然变成「一屏塞不下」。

## 文件

| 文件 | 职责 |
| --- | --- |
| `index.js` | Host 半：读网关、裁剪快照、两条同源路由、`magpie_usage` 工具 |
| `client.js` | 客户端半：侧栏图标 + 主面板（纯 JS，只用 `react` 与 `--dsw-alias-*` token） |
| `cordis.patch.yml` | Loader 行与默认配置 |
| `locale/*.json` | 插件管理器里的标题与描述（中 / 英） |
| `icon.svg` | 插件图标 |

## 注意

- 只读：不写 Magpie 的任何文件，不改网关状态；路由只接受 `GET`。
- 样式全部走 `--dsw-alias-*` 主题 token，浅色 / 深色跟随宿主。
- 客户端改完刷新页面；Host 侧改完重载插件行（`plugin_manager` 关 → 开）。

## License

MIT
