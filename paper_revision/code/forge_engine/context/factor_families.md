# Mechanism-family library for factor proposals (V19-K)

This file is METHOD knowledge, not dataset knowledge: it ships with the engine
and is identical for every dataset.  It contains no row labels, no outer-test
statistics, and no dataset-specific values.  It names the mechanism families
that repeatedly carry out-of-period signal in event-stream risk data, and the
families that repeatedly fail, so proposals can start from mechanisms instead
of algebra.

Hard constraints on every proposal (violations are rejected by the engine):
- STRICTLY PAST-ONLY: any aggregate over an entity at time t may use only rows
  of that entity with time < t.  No same-row or future information.
- Never use the label column or any statistic derived from held-out data.
- Must be well-defined for keys never seen before (define an explicit
  cold-start value such as 0, -1 or the global running mean).

## Families that repeatedly carry signal

1. ENTITY ROLLING HISTORY.  Compare the current numeric value against the same
   entity's own trailing history: rolling mean / std / max / count over the
   previous k events (k in 5..50) of the same card, account, device, IP or
   customer.  The strongest forms are normalized: (x - rolling_mean) /
   rolling_std, x / rolling_mean, and the trailing std itself (volatility).
2. RECENCY AND VELOCITY.  Time since the entity's previous event; number of
   events by the entity in a trailing window; ratio of a short-window rate to
   a long-window rate (burst detection).  Sudden acceleration of an entity's
   event rate is a mechanism, not a correlation.
3. PAIR NOVELTY AND PAIR HISTORY.  Whether this (entity_a, entity_b) pairing
   -- card x merchant, account x device, user x IP -- has been seen before;
   how many times; time since the pair's previous event.  First-time pairings
   of two individually active entities are a distinct mechanism from either
   entity's own history.
4. COUNTERPARTY CHURN.  Number of DISTINCT counterparties an entity touched in
   its trailing window (distinct merchants per card, distinct devices per
   account); the rate at which new counterparties appear (new-vs-reuse).
5. PEER-GROUP DISPERSION.  The current value's position within its
   categorical group's PAST distribution: past-only percentile or z-score of
   the amount within the same merchant category / product type / region.
6. ARITHMETIC INTERACTIONS (dense numeric data).  Ratios, differences, and
   products of two existing numeric columns, and log/sqrt rescalings --
   chosen for a stated physical or business reason (a rate = count / duration,
   a density = weight / area), not exhaustively.  On physics-like or
   sensor-like tables this family does most of the work.
7. CALENDAR PERIODICITY (when a time column exists).  Time-of-day and
   day-of-week as sin/cos pairs; deviation of the entity's current hour from
   that entity's own historical hour profile.

## Families that repeatedly fail out-of-period

- TARGET-ENCODING RECOMBINATIONS.  Differences or ratios BETWEEN existing
  target-encoding columns (a "TE context delta") tend to score highly on the
  discovery sample and reverse sign later in time: the encoding shifts with
  the base rate, the recombination inherits that drift twice.  Prefer the raw
  history mechanisms above; if a TE column is used at all, compare it to a
  past-only quantity of the same entity, never to another TE column.
- ALGEBRAIC RESTATEMENTS of factors already on the surface or blacklisted --
  renamed copies, sign flips, monotone rescalings.  These are removed by the
  orthogonalized entry rule at zero gain.
- SAME-ROW LEAKAGE PROXIES: any aggregate that includes the current row, or
  any window keyed on a column that is itself derived from the label.

When evidence in the prompt shows a residual concentrated on a subgroup,
prefer the family that explains WHY that subgroup is mispriced (history,
velocity, novelty, dispersion) over a flag that merely names the subgroup.
