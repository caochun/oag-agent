"""Load legacy domains or protocol-based domains."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

from .adapters.json_file import JsonFileAdapter
from .adapters.sqlite_table import SqliteTableAdapter
from .domain import DomainContext
from .registry import FunctionRegistry
from .repository import ObjectRepository
from .schema import Ontology


def load_domain(domain_dir: str | Path) -> tuple[Ontology, ObjectRepository, FunctionRegistry]:
    domain_dir = Path(domain_dir).resolve()
    manifest_path = domain_dir / "domain.yaml"
    manifest = _load_manifest(manifest_path) if manifest_path.is_file() else None
    provider = _load_provider(domain_dir, manifest) if manifest else None
    if provider is not None:
        ontology = provider.load_ontology()
        if not isinstance(ontology, Ontology):
            raise TypeError("Domain provider load_ontology must return Ontology")
    else:
        ontology = Ontology.load(domain_dir / "ontology.yaml")

    registry = FunctionRegistry()
    registry.register_adapter("json_file", JsonFileAdapter.factory(domain_dir))
    registry.register_adapter("sqlite_table", SqliteTableAdapter.factory(domain_dir))

    repository = ObjectRepository(ontology, registry)
    if provider is not None:
        provider.register(DomainContext(
            domain_dir=domain_dir,
            ontology=ontology,
            registry=registry,
            repository=repository,
        ))
    else:
        func_pkg = _import_module(domain_dir, "functions")
        func_pkg.register(registry, repository, ontology)

    return ontology, repository, registry


def _load_manifest(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    if manifest.get("schema") != "oag.domain.v1":
        raise ValueError(f"{path} schema must be oag.domain.v1")
    unknown = set(manifest) - {"schema", "provider"}
    if unknown:
        raise ValueError(
            f"{path} contains unknown fields: {', '.join(sorted(unknown))}"
        )
    provider = manifest.get("provider")
    if not isinstance(provider, str) or ":" not in provider:
        raise ValueError(f"{path} provider must be module:factory")
    return manifest


def _load_provider(domain_dir: Path, manifest: dict):
    module_name, factory_name = manifest["provider"].split(":", 1)
    module = _import_module(domain_dir, module_name)
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ValueError(f"Domain provider factory not found: {manifest['provider']}")
    provider = factory(domain_dir)
    if not callable(getattr(provider, "load_ontology", None)):
        raise TypeError("Domain provider must define load_ontology")
    if not callable(getattr(provider, "register", None)):
        raise TypeError("Domain provider must define register")
    return provider


def _import_module(domain_dir: Path, module_name: str):
    relative = Path(*module_name.split("."))
    module_path = domain_dir / relative
    if module_path.is_dir():
        source_path = module_path / "__init__.py"
        search_locations = [str(module_path)]
    else:
        source_path = module_path.with_suffix(".py")
        search_locations = None
    if not source_path.is_file():
        raise FileNotFoundError(f"Domain module not found: {module_name}")

    pkg_name = f"_domain_{domain_dir.name}_{module_name.replace('.', '_')}"

    spec = importlib.util.spec_from_file_location(
        pkg_name,
        source_path,
        submodule_search_locations=search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import domain module: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = module
    spec.loader.exec_module(module)
    return module
