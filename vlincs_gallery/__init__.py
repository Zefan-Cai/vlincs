"""vlincs_gallery - online, revisable, retrieval-based identity assignment for VLINCS TA1.

Per-detection match-or-expand against a live gallery of identities (FAISS hot path +
pgvector system-of-record), replacing the lossy batch funnel. See PROTOCOL.md.
"""

__version__ = "0.0.1"

from .weak_graph import TrackletEvidence, WeakGraphConfig, resolve_weak_graph
from .weak_labels import WeakLabelGenerationConfig, generate_weak_labels

__all__ = [
    "TrackletEvidence",
    "WeakLabelGenerationConfig",
    "WeakGraphConfig",
    "generate_weak_labels",
    "__version__",
    "resolve_weak_graph",
]
