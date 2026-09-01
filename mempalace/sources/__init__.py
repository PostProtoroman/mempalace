"""Source adapter subsystem (RFC 002).

Public surface:

* :class:`BaseSourceAdapter` — per-source read-side contract.
* Typed records: :class:`SourceRef`, :class:`SourceItemMetadata`,
  :class:`DrawerRecord`, :class:`RouteHint`, :class:`SourceSummary`,
  :class:`AdapterSchema`, :class:`FieldSpec`.
* Error classes: :class:`SourceNotFoundError`, :class:`AuthRequiredError`,
  :class:`AdapterClosedError`, :class:`TransformationViolationError`,
  :class:`SchemaConformanceError`.
* Registry: :func:`register`, :func:`get_adapter`, :func:`available_adapters`,
  :func:`resolve_adapter_for_source`.
* :class:`PalaceContext` — facade core passes to adapters during ``ingest``.
* :mod:`transforms` — reference implementations of the reserved §1.4
  transformations + :func:`get_transformation` resolver.
"""

from .base import (
    AdapterClosedError,
    AdapterSchema,
    AuthRequiredError,
    BaseSourceAdapter,
    DrawerRecord,
    FieldSpec,
    IngestMode,
    IngestResult,
    RouteHint,
    SchemaConformanceError,
    SourceAdapterError,
    SourceItemMetadata,
    SourceNotFoundError,
    SourceRef,
    SourceSummary,
    TransformationViolationError,
)
from .codex import CodexSessionAdapter
from .context import PalaceContext, ProgressHook
from .registry import (
    available_adapters,
    get_adapter,
    get_adapter_class,
    register,
    reset_adapters,
    resolve_adapter_for_source,
    unregister,
)

# First-party adapters are registered eagerly: `mempalace codex-capture` and
# `mempalace mine --source codex` must both resolve the name without the user
# installing anything. Explicit registration also wins over entry points (§3.2).
register("codex", CodexSessionAdapter)

__all__ = [
    "AdapterClosedError",
    "CodexSessionAdapter",
    "AdapterSchema",
    "AuthRequiredError",
    "BaseSourceAdapter",
    "DrawerRecord",
    "FieldSpec",
    "IngestMode",
    "IngestResult",
    "PalaceContext",
    "ProgressHook",
    "RouteHint",
    "SchemaConformanceError",
    "SourceAdapterError",
    "SourceItemMetadata",
    "SourceNotFoundError",
    "SourceRef",
    "SourceSummary",
    "TransformationViolationError",
    "available_adapters",
    "get_adapter",
    "get_adapter_class",
    "register",
    "reset_adapters",
    "resolve_adapter_for_source",
    "unregister",
]
