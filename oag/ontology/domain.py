"""Extension protocol for loading and registering OAG domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .registry import FunctionRegistry
from .repository import OntologyRepository
from .schema import Ontology


@dataclass(frozen=True)
class DomainContext:
    """Runtime components passed to a domain after its ontology is loaded."""

    domain_dir: Path
    ontology: Ontology
    registry: FunctionRegistry
    repository: OntologyRepository


class DomainProvider(Protocol):
    """Provide a domain ontology, then bind its runtime implementations."""

    def load_ontology(self) -> Ontology:
        """Return the final ontology used by every runtime component."""

    def register(self, context: DomainContext) -> None:
        """Register source adapters, runtime services, and functions."""
