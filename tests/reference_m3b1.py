"""Test-only v0.3.4 oracle for one frozen parent, through its first replacement."""
from dataclasses import dataclass, field
from datetime import timedelta


@dataclass
class Leg:
    pool: str
    first: int
    extreme: object
    version: int = 1
    reclaim: int | None = None
    broken: int | None = None
    fvg: int | None = None
    touched: set[int] = field(default_factory=set)

    @property
    def key(self):
        return self.pool, self.first, self.version


def interpret(bars, calendar, declarations, pools, *, bullish, boundary, target,
              tick, first, rules, usable=True):
    """Pools are frozen (ID, price) pairs at each interval start; seals are verified inputs."""
    sign = 1 if bullish else -1
    floor, ceiling = sign * boundary, sign * target
    duration = timedelta(minutes=1)
    normalized = [(sign*c.open, max(sign*c.high, sign*c.low),
                   min(sign*c.high, sign*c.low), sign*c.close) for c in bars]
    true_ranges = [None] + [
        max(h-l, abs(h-normalized[i-1][3]), abs(l-normalized[i-1][3]))
        for i, (_, h, l, _) in enumerate(normalized[1:], 1)
    ]

    def atr(i):
        samples = true_ranges[i-rules.atr_period:i]
        return sum(samples) / len(samples) if i > rules.atr_period else None

    def displacement(i):
        o, h, l, c = normalized[i]
        average = atr(i)
        return (average and h > l and c > o and c >= ceiling+tick
                and (c-o)/(h-l) >= rules.body_ratio_min
                and (c-l)/(h-l) >= rules.close_location_min
                and true_ranges[i] >= rules.range_atr_multiple_min*average)

    live, outcomes, conflicts, rejected = {}, {}, [], set()
    steps, sweeps, releases = [], [], []

    def note(leg, state, i):
        outcomes[leg.key] = state
        steps.append((i, leg.key, state))

    def find_gap(leg, i):
        earliest = max(2, leg.reclaim, leg.broken+rules.fvg_earliest_third_offset)
        latest = min(i, leg.reclaim+rules.leg_max_bars, leg.broken+rules.fvg_latest_third_offset)
        for j in range(earliest, latest+1):
            average = atr(j)
            if (average and not rejected.intersection(range(j-2,j+1))
                    and normalized[j][2]-normalized[j-2][1] >=
                    max(tick*rules.fvg_min_ticks, rules.fvg_atr_ratio_min*average)):
                return j
        return None

    def resolve(at):
        if not usable:
            return None
        prefix = [c for c in bars if c.interval_end <= at]
        for seal in declarations:
            if not seal.seal_available_at <= at <= seal.sealed_end:
                continue
            expected = calendar.expected_starts(seal.instrument, seal.timeframe,
                                                seal.declared_start, at-duration)
            if (expected is not None and tuple(c.interval_start for c in prefix) == expected
                    and all(c.source_available_at <= at for c in prefix)):
                break
        else:
            return None
        groups = {}
        for leg in live.values():
            if leg.fvg is not None:
                groups.setdefault(max(leg.fvg, leg.broken), []).append(leg)
        for completed, cohort in sorted(groups.items()):
            if len(cohort) > 1:
                for leg in cohort:
                    rejected.update(range(leg.fvg-2,leg.fvg+1))
                    rejected.add(leg.broken)
                    note(leg, "UNRESOLVED", completed)
                    del live[leg.pool]
                releases.append((at, completed, len(cohort)))
                continue
            leg = cohort[0]
            evidence = set(range(leg.fvg-2,leg.fvg+1)) | {leg.broken}
            if evidence & rejected:
                note(leg, "UNRESOLVED", completed)
                del live[leg.pool]
                continue
            if any(conflict >= completed or conflict in evidence | leg.touched | {leg.reclaim}
                   for conflict in conflicts):
                return None
            note(leg, "CONFIRMED", completed)
            for other in live.values():
                if other is not leg:
                    note(other, "STALE_PARENT", completed)
            return at, leg.key, tuple(bars[j].provenance.record_id for j in range(leg.fvg-2,leg.fvg+1))
        return None

    winner = None
    for i in range(first, len(bars)):
        o, _, edge, close = normalized[i]
        for name, leg in tuple(live.items()):
            if i > (leg.first+5 if leg.reclaim is None else leg.reclaim+rules.leg_max_bars):
                note(leg, "EXPIRED", i)
                del live[name]
        crossed = {name for name, level in pools[i].items() if edge <= sign*level-tick}
        if len(crossed)>1:
            conflicts.append(i)
            for name in sorted(crossed):
                leg = live.pop(name, Leg(name, i, edge))
                note(leg, "UNRESOLVED", i)
                sweeps.append((i, name, leg.first, leg.version, "CONFLICT"))
            crossed.clear()
        for name in sorted(set(live) | crossed):
            leg = live.get(name)
            reset = False
            terminated = leg is not None and leg.reclaim is not None and close <= floor-tick
            if name in crossed:
                if leg is None:
                    if not (floor < edge <= ceiling and floor <= o <= ceiling):
                        sweeps.append((i, name, None, None, "REJECT"))
                        continue
                    leg = live[name] = Leg(name, i, edge)
                    note(leg, "PENDING", i)
                    sweeps.append((i, name, i, 1, "START"))
                else:
                    sweeps.append((i, name, leg.first, leg.version, "REVISIT"))
                    if edge < leg.extreme:
                        if leg.reclaim is not None:
                            note(leg, "INVALIDATED", i)
                            leg.version += 1
                            leg.reclaim = leg.broken = leg.fvg = None
                            reset = True
                            note(leg, "PENDING", i)
                            sweeps.append((i, name, leg.first, leg.version, "RESET"))
                        leg.extreme = edge
                        if not (floor < edge <= ceiling and floor <= o <= ceiling):
                            note(leg, "INVALIDATED", i)
                            del live[name]
                            continue
                leg.touched.add(i)
            if terminated and not reset:
                note(leg, "INVALIDATED", i)
                del live[name]
            elif leg.reclaim is None:
                if i > leg.first+5:
                    note(leg, "EXPIRED", i)
                    del live[name]
                elif close > sign*pools[i][name]:
                    leg.reclaim = i
                    steps.append((i, leg.key, "RECLAIM"))
                    if close >= ceiling+tick:
                        note(leg, "INVALIDATED", i)
                        del live[name]
            else:
                if leg.broken is None and displacement(i):
                    leg.broken = i
                    steps.append((i, leg.key, "BREAK"))
                if leg.broken is not None:
                    gap = find_gap(leg, i)
                    if gap is not None and leg.fvg is None:
                        leg.fvg = gap
                        note(leg, "EVIDENCE_READY", i)
                    elif gap is None and i > leg.broken+rules.fvg_latest_third_offset:
                        note(leg, "INVALIDATED", i)
                        del live[name]
        arrived = bars[i].source_available_at
        following = bars[i+1].source_available_at if i+1<len(bars) else None
        if following != arrived:
            cuts = {arrived} | {s.seal_available_at for s in declarations
                              if s.seal_available_at > arrived and
                              (following is None or s.seal_available_at < following)}
            for cut in sorted(cuts):
                winner = resolve(cut)
                if winner:
                    break
        if winner:
            break
    return winner, outcomes, sweeps, steps, releases
