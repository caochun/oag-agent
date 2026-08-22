"""Extension protocol for loading and registering OAG domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .bindings import RuntimeBindings
from .repository import OntologyRepository
from .schema import Ontology
from .source import SourceManager


@dataclass(frozen=True)
class DomainContext:
    """Runtime components passed to a domain after its ontology is loaded."""

    ontology: Ontology
    bindings: RuntimeBindings
    sources: SourceManager
    repository: OntologyRepository


class DomainProvider(Protocol):
    """Provide a domain ontology, then bind its runtime implementations."""

    def load_ontology(self) -> Ontology:
        """Return the final ontology used by every runtime component."""

    def register(self, context: DomainContext) -> None:
        """Bind sources, the Action runtime, and side-effect-free Functions."""
