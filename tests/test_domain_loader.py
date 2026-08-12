from __future__ import annotations

import textwrap
import tempfile
import unittest
from pathlib import Path

from oag.ontology.loader import load_domain


BASE_ONTOLOGY = """
name: Base domain
objects:
  Thing:
    display_name: Thing
    source:
      type: json_file
      id_field: id
      config:
        path: data.json
functions: {}
"""


def write(path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")


class DomainLoaderTest(unittest.TestCase):
    def test_provider_loads_final_ontology_before_repository_creation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root / "data.json", "[]")
            write(root / "domain.yaml", """
                schema: oag.domain.v1
                provider: provider:create_domain
            """)
            write(root / "provider.py", """
                from oag.ontology.schema import Ontology

                class Provider:
                    def load_ontology(self):
                        return Ontology.model_validate({
                            "name": "Provided domain",
                            "objects": {
                                "Thing": {
                                    "display_name": "Thing",
                                    "source": {
                                        "type": "json_file",
                                        "id_field": "id",
                                        "config": {"path": "data.json"},
                                    },
                                },
                            },
                            "functions": {
                                "domain_name": {"summary": "Domain name"},
                            },
                        })

                    def register(self, context):
                        assert context.repository.ontology is context.ontology
                        context.registry.register(
                            "domain_name",
                            lambda: context.repository.ontology.name,
                            context.ontology.functions["domain_name"],
                        )

                def create_domain(domain_dir):
                    return Provider()
            """)

            ontology, repository, registry = load_domain(root)
            try:
                self.assertFalse((root / "ontology.yaml").exists())
                self.assertEqual("Provided domain", ontology.name)
                self.assertIs(ontology, repository.ontology)
                self.assertEqual("Provided domain", registry.call("domain_name"))
            finally:
                repository.close()

    def test_domain_without_manifest_uses_legacy_functions_register(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root / "ontology.yaml", BASE_ONTOLOGY)
            write(root / "data.json", "[]")
            write(root / "functions" / "__init__.py", """
                from oag.ontology.schema import FunctionDef

                def register(registry, repository, ontology):
                    definition = FunctionDef(summary="Legacy")
                    ontology.functions["legacy"] = definition
                    registry.register("legacy", lambda: "legacy", definition)
            """)

            _, repository, registry = load_domain(root)
            try:
                self.assertEqual("legacy", registry.call("legacy"))
            finally:
                repository.close()

    def test_provider_manifest_does_not_describe_ontology_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write(root / "domain.yaml", """
                schema: oag.domain.v1
                provider: provider:create_domain
                ontology: ontology.yaml
            """)

            with self.assertRaisesRegex(ValueError, "unknown fields: ontology"):
                load_domain(root)


if __name__ == "__main__":
    unittest.main()
