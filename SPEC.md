# Evolutionary Ensemble Trading Bot — Техническая Спецификация

> Этот документ является главным контекстом для Claude Code.
> При любых изменениях системы — обновляй этот файл.

---

## 1. Суть системы

**Evolutionary Ensemble Trading System** — скальпер на паре BTC/USDT,
состоящий из двух независимых слоёв:

- **Слой 1 (Research)** — популяция из N ботов с разными параметрами,
  которые торгуют на бумаге, соревнуются между собой и эволюционируют.
  Работает всегда, никогда не касается реальных денег.

- **Слой 2 (Execution)** — голосование популяции определяет итоговое
  решение. Сначала тоже бумага. Переход на реальную биржу — только
  после достижения критериев зрелости.

---

## 2. Архитектура

**Stateless bot + PostgreSQL (TimescaleDB).**

Бот полностью stateless — всё состояние хранится в PostgreSQL.
При рестарте бот загружает ботов, позиции и балансы из DB и продолжает работу.
DB — единственный source of truth и полный audit trail.

| Компонент         | Решение                          |
|-------------------|----------------------------------|
| Язык              | Python 3.12+                     |
| Типизация         | mypy strict mode                 |
| Линтер            | ruff                             |
| Биржа             | Bybit                            |
| Биржевая либа     | ccxt (async WebSocket)           |
| Асинхронность     | asyncio                          |
| База данных       | PostgreSQL 16 + TimescaleDB      |
| DB драйвер        | asyncpg (async)                  |
| Контейнер         | Docker + docker-compose          |
| Логи              | stderr (stdlib logging)          |
| Audit trail       | PostgreSQL (trades, evolutions)  |

---

## 3. Структура проекта

```
trading-bot/
│
├── config/
│   ├── params.yaml          # все параметры системы
│   └── exchange.yaml        # настройки Bybit (API ключи)
│
├── core/
│   ├── market_data.py       # WebSocket: order book, trades, поводыри
│   ├── signals.py           # вычисление сигналов (imbalance, flow, leaders)
│   └── decision.py          # логика входа/выхода для одного бота
│
├── evolution/
│   ├── population.py        # управление популяцией ботов (stateless)
│   ├── fitness.py           # оценка каждого бота (multi-objective)
│   └── genetics.py          # селекция, скрещивание, мутация
│
├── paper/
│   └── simulator.py         # PaperTradingConfig (fees, slippage, balance)
│
├── ensemble/
│   └── voting.py            # голосование популяции → итоговый сигнал
│
├── storage/
│   └── database.py          # StateDB (PostgreSQL), StateDBProtocol, DTOs
│
├── tests/
│   ├── mock_db.py           # MockStateDB — in-memory для тестов без PG
│   └── test_*.py            # тесты каждого модуля (72 теста)
│
├── main.py                  # оркестратор — связывает все модули
├── Dockerfile
├── docker-compose.yml       # TimescaleDB + bot
└── SPEC.md                  # этот файл
```

> **Правило:** каждый файл отвечает за одну вещь.
> Изменение одного модуля не должно ломать остальные.

---

## 4. Торгуемая пара и биржа

- **Пара:** BTC/USDT (спот)
- **Биржа:** Bybit
- **Режим разработки:** Bybit (paper trading — виртуальные балансы)
- **Режим продакшн:** Bybit реальный счёт (только после graduation)

---

## 5. Источники данных (WebSocket)

| Стрим              | Что даёт                          | Частота     |
|--------------------|-----------------------------------|-------------|
| orderbook.50       | Стакан 50 уровней BTC/USDT        | 100ms       |
| publicTrade        | Поток реальных сделок BTC/USDT    | реальное    |
| orderbook.50 ETH   | Стакан ETH/USDT (поводырь)        | 100ms       |
| tickers BTCPERP    | BTC перпетуал фьюч (поводырь)     | реальное    |

**Все боты слушают один общий поток данных** — не N отдельных соединений.

---

## 6. Сигналы входа

### 6.1 Order Book Imbalance
```
bid_volume = сумма объёмов топ-10 уровней bid
ask_volume = сумма объёмов топ-10 уровней ask
imbalance  = bid_volume / (bid_volume + ask_volume)

imbalance > threshold → давление покупателей → сигнал LONG
imbalance < (1 - threshold) → давление продавцов → сигнал SHORT
```

### 6.2 Trade Flow
```
За последние N секунд (flow_window_seconds из config):
buy_flow  = объём агрессивных покупок (taker side = buy)
sell_flow = объём агрессивных продаж (taker side = sell)
flow_ratio = buy_flow / sell_flow  (capped [0.01, 100])

flow_ratio > flow_threshold → подтверждение LONG
flow_ratio < (1 / flow_threshold) → подтверждение SHORT

Вес flow сигнала регулируется flow_weight (config).
```

### 6.3 Поводыри (Lead-Lag)
```
ETH/USDT:
  Если ETH вырос на eth_move_threshold % за последние eth_window сек
  и BTC ещё не отреагировал → усиление сигнала LONG (и наоборот)
  Вес в итоговом score: leader_weight (параметр бота)

BTC Perpetual Funding Rate:
  Считывается из ticker stream, доступен в SignalValues.
  Пока не используется как фильтр — готов для будущего использования.
```

### 6.4 Фильтры (не торговать когда)
- Спред > max_spread_usd
- Волатильность за последние 60 сек < min_volatility (боковик)
- Волатильность > max_volatility (хаос)

---

## 7. Параметры одного бота

Каждый бот — набор из 7 эволюционируемых параметров:

| Параметр               | Диапазон         | Описание                        |
|------------------------|------------------|---------------------------------|
| imbalance_threshold    | 0.55 – 0.85      | Порог дисбаланса стакана        |
| flow_threshold         | 1.2 – 3.0        | Порог давления потока сделок    |
| take_profit_usd        | $8 – $40         | Тейк-профит в долларах          |
| stop_loss_usd          | $5 – $25         | Стоп-лосс в долларах            |
| max_hold_seconds       | 10 – 120         | Макс. время удержания позиции   |
| eth_move_threshold     | 0.01% – 0.05%   | Порог движения ETH              |
| leader_weight          | 0.0 – 1.0        | Вес поводырей в итоговом сигнале|

Окна `flow_window_seconds` и `eth_window_seconds` — глобальные параметры
в `config/params.yaml` (сигналы вычисляются один раз для всех ботов).

---

## 8. Популяция и эволюция

### 8.1 Популяция
- **Размер:** 20 ботов (configurable в params.yaml)
- **Виртуальный баланс каждого бота:** $10,000 (сбрасывается при эволюции)
- **Размер позиции:** $1,000
- **Каждый бот независим** — своя позиция, свой баланс, свои параметры
- **Проверка баланса:** бот не открывает позицию если баланс < position_size_usd

### 8.2 Fitness Score (оценка бота)
```python
fitness = (
    winrate          * 0.30 +
    profit_factor    * 0.30 +
    sharpe_ratio     * 0.20 -
    max_drawdown_pct * 0.20
)
```

| Метрика        | Что означает                              |
|----------------|-------------------------------------------|
| winrate        | % прибыльных сделок                       |
| profit_factor  | сумма профитов / сумма убытков            |
| sharpe_ratio   | прибыль / стандартное отклонение прибыли  |
| max_drawdown   | максимальная просадка от пика             |

### 8.3 Цикл эволюции (каждые 100 сделок популяции)

```
1. Рассчитать fitness каждого бота (по trades текущего поколения из DB)
2. Отсортировать по fitness
3. Селекция:
   - Топ 30%  → выживают без изменений (элита)
   - Средние 40% → скрещивание элиты + лёгкая мутация
   - Худшие 30% → случайные новые параметры
4. Атомарная запись в DB: новые боты + evolution stats + сброс позиций
5. Сброс балансов к initial_balance_usd
```

### 8.4 Скрещивание
```
parent_a = лучший бот
parent_b = второй бот

child.param = mean(parent_a.param, parent_b.param) для каждого параметра
+ лёгкая мутация потомка
```

### 8.5 Мутация
```
Каждый параметр с вероятностью mutation_rate (default: 0.2):
  delta = random(-1, 1) * mutation_strength * (max - min)
  новое_значение = clamp(текущее + delta, min, max)
```

---

## 9. Голосование (Ensemble Voting)

```
На каждый тик данных:

signals = [бот.compute_signal() for бот in популяция]
# Возможные значения: LONG, SHORT, HOLD

long_ratio  = count(LONG) / total
short_ratio = count(SHORT) / total

if long_ratio  >= threshold_long  → итог LONG,  confidence = long_ratio
if short_ratio >= threshold_short → итог SHORT, confidence = short_ratio
else → HOLD
```

---

## 10. Paper Trading

Виртуальная торговля — без реальных ордеров:

- Каждый бот имеет виртуальный баланс (начальный $10,000)
- Позиция открывается/закрывается по текущей рыночной цене
- **Комиссия:** taker fee (0.06%) + slippage (0.5 * spread / price)
- **PnL:** net of fees (fees вычитаются из PnL при закрытии)
- **Audit trail:** каждая сделка в DB с entry/exit signals

---

## 11. Критерии зрелости (graduation to live)

Переход на реальный счёт только при **одновременном** выполнении всех:

| Критерий                      | Порог              |
|-------------------------------|--------------------|
| Минимум сделок (Execution)    | ≥ 500              |
| Winrate                       | ≥ 55% стабильно    |
| Profit Factor                 | ≥ 1.5              |
| Максимальная просадка         | ≤ 10%              |
| Стабильность                  | 3 поколения подряд в плюсе |

Graduation **никогда не происходит автоматически** — только с подтверждения пользователя.

---

## 12. База данных (PostgreSQL + TimescaleDB)

### Схема

- **population** — одна строка: generation, total_trades (CHECK id=1)
- **bots** — текущее поколение ботов (bot_id, generation, params JSONB)
- **positions** — открытые позиции (bot_id PK, side, entry_price, entry_signals)
- **trades** — закрытые сделки (hypertable по created_at). Полный audit: entry/exit signals
- **evolutions** — история эволюций (generation UNIQUE, best/avg fitness, best_params)

### Атомарные операции

- **close_trade** — одна транзакция: DELETE position + INSERT trade + INCREMENT total_trades
- **run_evolution_tx** — одна транзакция: INSERT evolution + DELETE/INSERT bots + DELETE positions + UPDATE population

### Восстановление при рестарте

- Загрузка ботов из DB (`load_bots`)
- Восстановление балансов: `initial_balance + SUM(pnl) - SUM(fees)` за текущее поколение
- Восстановление открытых позиций с entry_signals
- Подключение с retry (10 попыток, backoff 2s)

### Протокол

`StateDBProtocol` — Protocol для тестов. `MockStateDB` — in-memory реализация.

---

## 13. Принципы кода для Claude Code

1. **Один файл — одна ответственность.** Не мешать логику разных слоёв.
2. **Комментарии объясняют "зачем", не "что".** Код говорит сам за себя.
3. **Все параметры в `config/params.yaml`.** Не хардкодить числа в коде.
4. **Каждое решение логируется с контекстом.** Должно быть понятно почему бот сделал что сделал.
5. **mypy strict + ruff** — никаких исключений.
6. **Тест на каждый модуль.** Перед изменением — запустить тесты.
7. **Эволюция прозрачна.** DB хранит историю каждого поколения.
8. **Stateless.** Бот не хранит состояние — всё в PostgreSQL.

---

## 14. Известные ограничения

- **max_hold_seconds** проверяется на каждом тике (не по таймеру). Разрешение зависит от частоты WebSocket (~100ms). Приемлемо для скальпинга.
- **In-memory / DB divergence** — если DB write падает после изменения in-memory state, состояние расходится до рестарта. При рестарте DB побеждает. Приемлемо для paper trading.

---

## 15. Референсы

- John Holland — "Adaptation in Natural and Artificial Systems" (1975)
- Earnest Chan — "Algorithmic Trading" (практика)
- Marcos Lopez de Prado — "Advances in Financial Machine Learning"
- Freqtrade — open source референс архитектуры
- DEAP — Python библиотека генетических алгоритмов
- ccxt — Python библиотека для бирж

---

*Версия документа: 2.0 | Дата: март 2026*
