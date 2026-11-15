# MLOF v2.0 — Offline structural research foundation

This small Python project handles **local candle data only**. It validates CSV files, timestamps, prices, ordering, duplicate bars, gaps, and calendar coverage; then it can write a normalized CSV and a deterministic audit report.

The owner-accepted baseline is M4.5 under governing v0.3.6. It includes sealed structural replay, initial and ordinary controller formation, and offline SCALP setup evaluation. SCALP uses 4H primary context and 15M confirmation for 1M, 3M, and 5M execution streams. Daily observations are nonbinding. SWING is disabled. All results are non-tradable.

See `AGENTS.md` for governing artifact hashes and change-control requirements. Obtain governing documents separately from the owner.

The included CSV is synthetic demonstration data. It is not market data and cannot support trading or profitability conclusions.

## Requirements

- Windows 10/11
- Python 3.11 or newer; this project was prepared for Python 3.15.2
- No third-party Python packages are required. It uses only Python's standard library.
- No network connection is used.

## Set up the project on Windows

Open PowerShell at the repository root.

Create a private virtual environment for this project. This does not install packages:

```powershell
python -m venv .venv
```

Check Python inside it:

```powershell
.\.venv\Scripts\python.exe --version
```

## Run the automated tests

The tests use Python's built-in `unittest`, so there is nothing to install:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -B -m unittest tests.test_scalp -q
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -q
```

The accepted baseline is 7 focused SCALP tests and 147 full regression tests.

## Validate the included synthetic CSV

This command reads the sample, prints a JSON audit, and writes normalized output beneath `data\processed`. It does not modify the source CSV:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m mlof_v2 validate-csv `
  --csv data\sample\synthetic_ohlc.csv `
  --config config\instruments.example.json `
  --calendar config\calendar.synthetic.example.json `
  --as-of-utc 2024-01-01T00:06:00Z `
  --normalized-out data\processed\synthetic_normalized.csv `
  --audit-out reports\synthetic_audit.json
```

The sample is shorter than the example 50-bar data-readiness minimum, so a clean result is expected to be `WARMING`. `NOT_READY` validation returns exit code 2; `WARMING` and `READY` return 0. The audit always states the readiness status and issue list.

## Load a local CSV

The required source columns are:

`instrument,timeframe,interval_start_utc,interval_end_utc,source_available_at_utc,open,high,low,close`

Timestamp values must be ISO-8601 and include an explicit timezone offset. They are converted to UTC. Naive timestamps and the unknown-offset marker `-00:00` are rejected. `--as-of-utc` is mandatory so a final partial candle cannot be guessed from the computer clock.

Before loading a real instrument, create a separate configuration with a **verified tick size** and an explicit session calendar covering the data range. The example `TESTUSD` tick size and calendar are synthetic only. Do not use them for a real instrument. If calendar coverage is unknown, missing intervals cannot be classified and the result is `NOT_READY`.

Example command after you have supplied your own verified local configuration and calendar files:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m mlof_v2 validate-csv `
  --csv "C:\path\to\your\local_data.csv" `
  --config config\your_verified_instruments.json `
  --calendar config\your_verified_calendar.json `
  --as-of-utc 2025-02-01T00:00:00Z `
  --normalized-out data\processed\normalized.csv `
  --audit-out reports\data_quality.json
```

Use `--expected-start-utc` and `--expected-end-utc` together when the expected coverage extends beyond the first or last source row. The end value identifies the final expected candle's **start**. No missing candle is generated.

## Readiness states

- `NOT_READY`: malformed or invalid rows, duplicate timestamps, out-of-order records, overlaps, missing expected bars, unknown calendar coverage, incomplete candles, unknown tick size, or unrepresentable prices prevent safe use.
- `WARMING`: the data is valid and calendar-covered, but a stream has fewer than the configured number of completed candles (50 by default). This is only a data-readiness label; it does not mean strategy structure is initialized.
- `READY`: data checks pass and each stream meets the configured minimum. It is not approval to trade and does not imply that v2 strategy rules or broker data are ready.

Malformed records are listed with deterministic rejection codes. Duplicate timestamp groups exclude every conflicting row. Normalized records retain the input-file hash, source path, source line, source-row hash, raw source fields, and a stable record ID. Output is sorted by instrument, timeframe, interval start, then record ID. The CSV source is read only.

## Calendar behavior

The session-calendar interface returns expected UTC candle starts for an explicitly covered range. `ExplicitSessionCalendar` can represent known open intervals and closures. It is deliberately finite; it does not invent broker hours, holidays, or DST rules. A request outside its coverage returns unknown, which makes the affected stream `NOT_READY`. The included calendar is for the synthetic sample only.

For more than one instrument, supply a `CalendarRegistry` that maps each instrument to its own verified calendar. An unmapped instrument returns unknown and is `NOT_READY`; one instrument's calendar is never silently reused for another.

Incremental ingestion buffers source rows and produces its final result only when finalized. Batch and buffered ingestion use the same validation and canonical ordering. This is an offline data contract, not a live feed.

## Milestone 2 structural observations

`config\structural_research.example.json` contains versioned pivot-width defaults from the Gate 0 research design: 15-second and 30-second bars use `L=3, R=3`; 1-minute through context timeframes use `L=2, R=2`. Per the Gate 0 comparison formula, the engine compares `2L` completed bars to the left and waits for `2R` completed bars to the right, plus the candidate anchor. Thus `L=2, R=2` means 4 left + 1 anchor + 4 right = 9 bars; `L=3, R=3` means 6 + 1 + 6 = 13 bars. The same file sets the 50-contiguous-bar `WARMING` to `READY` threshold. These are research defaults, not optimized strategy parameters.

The `mlof_v2.structure.StructuralAnalyzer` consumes complete candles in interval order. A high/low candidate starts only when its completed candle is an extreme against all `2L` completed left comparison bars. It becomes a confirmed swing only after `2R` right-side completed bars. A more extreme bar invalidates the candidate and may independently start a new one. An equal-price observation extends the unconfirmed cluster without moving its anchor and restarts the right-side count. Once confirmed, the cluster's ID, anchor, members, tick level, confirmation, and availability are immutable; a later equal-price revisit creates a separate candidate. Confirmed swings preserve each comparison candle's ID, interval, source availability timestamp, and source-row provenance, including the left-side evidence.

Confirmed swings produce provisional structural references and liquidity-pool records. These records have no directional-control status and are not sweep or trade signals. A pool cannot be queried for the candle that confirmed it. Its availability must be no later than a later candle's interval start, and registration must have occurred before that candle was observed. If the calendar detects a missing expected interval, provisional state resets and the stream stays `NOT_READY`; missing bars are never fabricated. Events carry stable IDs, configuration version, reason, source candle IDs, and original row provenance. `structural_events_jsonl` emits deterministic audit text.

Example: inspect confirmed swings from local candles only after the existing CSV loader has validated them. This does not submit orders or connect to any service:

```python
from mlof_v2.structure import StructuralAnalyzer, load_structural_config

config = load_structural_config("config/structural_research.example.json")
analyzer = StructuralAnalyzer(config, verified_instrument_specs, verified_session_calendar)
result = analyzer.process_batch(validated_candles)
for swing in result.swings:
    print(swing.swing_event_id, swing.side.value, swing.level_ticks, swing.available_at)
```

Batch replay sorts by instrument, timeframe, and candle interval start. Incremental input must use that same event-time order. Source availability is checked separately; an availability timestamp that moves backward within a stream makes that stream `NOT_READY`. No event can be used during its own candle: references are eligible only at a later candle's interval start and after prior registration.

## Milestone 3A non-tradable bootstrap

`config\bootstrap_research.example.json` contains explicit versioned defaults from Gate 0 for continuous warm-up, ATR, displacement quality, and FVG size/timing. Its `config_version` must match the structural configuration. The values are research defaults, not optimized parameters.

`mlof_v2.structure.BootstrapAnalyzer` consumes the same completed candles as the structural analyzer. It finds consecutive confirmed high/low clusters, checks that both pools were available and registered before the observation candle began, freezes the highest-priority eligible pair, and observes only the specification-defined sweep/reclaim, displacement, and FVG sequence. It then records the first control and frozen range using the original pivot prices. Missing data, unknown sessions, unavailable references, identity collisions, and OHLC ordering ambiguity fail closed.

Pair and manipulation IDs use the approved canonical JSON and SHA-256 rules. Events retain stable lineage, configuration version, source candles, source availability, and pivot comparison provenance. `bootstrap_replay_jsonl(result)` writes deterministic evidence output. All bootstrap events, controls, and ranges are explicitly marked non-tradable. This milestone does not generate setups or signals.

For audit continuity, a bootstrap parent ID is derived from the first qualifying sweep and uses the selected observation-pair ID as its bootstrap control-ancestor reference. Its first candidate is `:candidate:v1`. Each qualifying strictly deeper sweep after reclaim creates the next candidate version and an immutable reset event linking the parent, canceled version, replacement version, triggering candle, interval, source availability, and configuration. Equal or shallower sweeps do not advance the version. Frozen range IDs are collision-checked before any manipulation, exchange, or control is promoted; a collision leaves the stream without a partial promotion.

Example using already validated local candles:

```python
from mlof_v2.structure import BootstrapAnalyzer, load_bootstrap_rules, load_structural_config

structure_config = load_structural_config("config/structural_research.example.json")
bootstrap_rules = load_bootstrap_rules("config/bootstrap_research.example.json")
engine = BootstrapAnalyzer(structure_config, bootstrap_rules, verified_instrument_specs, verified_calendar)
result = engine.process_batch(validated_candles)
print(len(result.controls))  # zero or one initial control per instrument/timeframe
```

Synthetic examples and tests validate code behavior only. They are not market evidence and cannot support profitability conclusions.

## Offline structural analysis modules

`mlof_v2.structure.engine.StructuralAnalyzer` confirms swings and time-gates liquidity pools. `BootstrapAnalyzer` can establish one immutable first controller. `SubsequentExchangeAnalyzer` composes those existing components and evaluates ordinary exchanges using the exact linked frozen-range boundary. It checks independent structural promotion and source availability before a sweep, preserves the association through candidate versions, and rejects a deeper sweep that crosses the boundary. It records manipulations, exchanges, candidate resets, controller ancestry/replacement, frozen ranges, readiness, source evidence, and deterministic IDs. Batch replay orders candles deterministically; incremental replay consumes the same completed candles in stream order. Synthetic end-to-end tests exercise bullish and bearish READY successors.

All records remain non-tradable. The formation module provides structural inputs to the offline setup evaluators. The project provides no order, broker, monetary risk, fill, P&L, or backtest capability.


## Offline 4H and Daily source intervals

Use `4h` and `1d` in canonical candle records. CSV ingestion also accepts `daily`.
Provide the exact source candle intervals in the instrument's versioned calendar.
Each `candle_intervals` row has `instrument`, `timeframe`, `start`, and `end`.
Both timestamps must carry explicit offsets and are normalized to UTC. Declare
every expected candle within calendar coverage, including shortened sessions
and offset changes. An omitted session indicates no expected candle, so verify
the schedule against the source before sealing it. Open-session windows alone
do not establish 4H or Daily candle boundaries. The replay does not aggregate
smaller candles or infer an instrument's session convention.
