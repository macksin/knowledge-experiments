"""Bitemporal SQL fragments, shared by every store implementation.

Centralizing the point-in-time predicate keeps the SQLite and (future) Postgres
stores in lockstep and gives the one canonical place to reason about correctness.

An edge is *active as of* `:as_of` when:
  * its valid-time interval contains the instant, AND
  * the row has not been superseded in transaction-time.

The canonical "project focus A -> B" case resolves entirely through this clause:
an `as_of` before the cutover returns the A edge; after, the B edge. Both rows
live forever; invalidation only sets `valid_to` / `expired_at`.
"""

from __future__ import annotations

# Active-as-of predicate. Bind :as_of (ISO-8601 string) to use it.
ACTIVE_AS_OF = (
    "valid_from <= :as_of "
    "AND (valid_to IS NULL OR valid_to > :as_of) "
    "AND expired_at IS NULL"
)

# Currently-believed predicate (ignores valid-time; only transaction-time).
CURRENTLY_BELIEVED = "expired_at IS NULL"
