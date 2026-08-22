"""Build one OAG domain through the provider lifecycle."""

from __future__ import annotations

from .bindings import RuntimeBindings
from .domain import DomainContext, DomainProvider
from .repository import OntologyRepository
from .schema import Ontology
from .source import SourceManager


def load_domain(
    provider: DomainProvider,
) -> tuple[Ontology, OntologyRepository, RuntimeBindings]:
    ontology = provider.load_ontology()
    if not isinstance(ontology, Ontology):
        raise TypeError("Domain provider load_ontology must return Ontology")

    bindings = RuntimeBindings()
    sources = SourceManager(ontology)
    repository = OntologyRepository(ontology, sources)
    try:
        provider.register(DomainContext(
            ontology=ontology,
            bindings=bindings,
            sources=sources,
            repository=repository,
        ))
        sources.validate_configuration()
        declared_functions = set(ontology.functions)
        registered_functions = {
            name for name, _definition in bindings.list_functions()
        }
        extra_functions = sorted(registered_functions - declared_functions)
        if extra_functions:
            raise ValueError(
                "Domain registered functions absent from the ontology: "
                + ", ".join(extra_functions)
            )
        missing_functions = sorted(
            declared_functions - registered_functions
        )
        if missing_functions:
            raise ValueError(
                "Domain function implementations are missing: "
                + ", ".join(missing_functions)
            )
        mismatched_functions = sorted(
            name
            for name, definition in ontology.functions.items()
            if bindings.get_def(name) != definition
        )
        if mismatched_functions:
            raise ValueError(
                "Domain function definitions do not match the ontology: "
                + ", ".join(mismatched_functions)
            )
        if ontology.actions and bindings.get_action_runtime() is None:
            raise ValueError("Domain actions require an ActionRuntime")
    except Exception:
        repository.close()
        raise

    return ontology, repository, bindings
