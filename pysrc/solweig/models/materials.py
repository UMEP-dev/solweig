"""Typed views over the materials SimpleNamespace.

The on-disk materials JSON is loaded as a nested :class:`SimpleNamespace` for
flexibility, but the access patterns used during a SOLWEIG calculation are
narrow and well-defined. The helpers here provide typed, null-safe accessors
for those patterns so the orchestrator does not have to chain ``getattr``
calls (which silently swallow typos and bypass static type checking).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace


@dataclass(frozen=True)
class WallMaterialDefaults:
    """Per-property wall overrides extracted from a materials namespace.

    Each field is ``None`` when the underlying materials JSON has no value
    for that property — callers should keep their built-in defaults in that
    case.
    """

    tgk: float | None = None
    tstart: float | None = None
    tmaxlst: float | None = None

    @classmethod
    def from_namespace(cls, materials: SimpleNamespace | None) -> WallMaterialDefaults:
        """Read ``Walls`` overrides from a materials namespace.

        Returns an empty :class:`WallMaterialDefaults` (all fields ``None``)
        when ``materials`` is ``None`` or when an intermediate node is missing.
        """
        if materials is None:
            return cls()

        def _read(top: str) -> float | None:
            return getattr(getattr(getattr(materials, top, None), "Value", None), "Walls", None)

        return cls(
            tgk=_read("Ts_deg"),
            tstart=_read("Tstart"),
            tmaxlst=_read("TmaxLST"),
        )

    def apply(
        self,
        tgk_default: float,
        tstart_default: float,
        tmaxlst_default: float,
    ) -> tuple[float, float, float]:
        """Return wall params with per-field overrides applied to the defaults."""
        return (
            float(self.tgk) if self.tgk is not None else tgk_default,
            float(self.tstart) if self.tstart is not None else tstart_default,
            float(self.tmaxlst) if self.tmaxlst is not None else tmaxlst_default,
        )
