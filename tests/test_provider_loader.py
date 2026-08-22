from __future__ import annotations

import pytest

from oag.ontology.loader import load_domain
from oag.ontology.schema import Ontology


class MemorySource:
    def close(self):
        pass


class Provider:
    def __init__(self):
        self.registered_context = None

    def load_ontology(self):
        return Ontology.model_validate({
            "name": "Provided domain",
            "data_sources": {"graph": {"type": "memory"}},
            "objects": {"Thing": {"binding": {"source": "graph"}}},
            "functions": {"domain_name": {"summary": "Domain name"}},
        })

    def register(self, context):
        self.registered_context = context
        context.sources.register("memory", lambda **kwargs: MemorySource())
        context.bindings.register(
            "domain_name",
            lambda: context.repository.ontology.name,
            context.ontology.functions["domain_name"],
        )


def test_provider_loads_final_ontology_before_runtime_registration():
    provider = Provider()

    ontology, repository, bindings = load_domain(provider)
    try:
        assert ontology.name == "Provided domain"
        assert ontology is repository.ontology
        assert provider.registered_context.ontology is ontology
        assert provider.registered_context.repository is repository
        assert bindings.call("domain_name") == "Provided domain"
    finally:
        repository.close()


def test_provider_must_return_an_ontology():
    class InvalidProvider:
        def load_ontology(self):
            return {}

        def register(self, context):
            raise AssertionError("register must not run")

    try:
        load_domain(InvalidProvider())
    except TypeError as exc:
        assert "must return Ontology" in str(exc)
    else:
        raise AssertionError("invalid provider result was accepted")


def test_registration_failure_closes_created_sources():
    source = MemorySource()

    def mark_closed():
        source.closed = True

    source.closed = False
    source.close = mark_closed

    class FailingProvider(Provider):
        def register(self, context):
            context.sources.register("memory", lambda **kwargs: source)
            context.sources.get("graph")
            raise RuntimeError("registration failed")

    try:
        load_domain(FailingProvider())
    except RuntimeError as exc:
        assert str(exc) == "registration failed"
    else:
        raise AssertionError("registration failure was swallowed")
    assert source.closed is True


def test_declared_function_requires_registered_implementation():
    class MissingFunctionProvider(Provider):
        def register(self, context):
            context.sources.register("memory", lambda **kwargs: MemorySource())

    with pytest.raises(ValueError, match="domain_name"):
        load_domain(MissingFunctionProvider())


def test_declared_actions_require_action_runtime():
    class MissingActionRuntimeProvider:
        def load_ontology(self):
            return Ontology.model_validate({
                "name": "Actions",
                "actions": {"approve": {"display_name": "Approve"}},
            })

        def register(self, context):
            pass

    with pytest.raises(ValueError, match="ActionRuntime"):
        load_domain(MissingActionRuntimeProvider())


def test_registered_action_runtime_must_implement_protocol():
    class InvalidActionRuntimeProvider:
        def load_ontology(self):
            return Ontology.model_validate({
                "name": "Actions",
                "actions": {"approve": {"display_name": "Approve"}},
            })

        def register(self, context):
            context.bindings.register_action_runtime(object())

    with pytest.raises(TypeError, match="must implement ActionRuntime"):
        load_domain(InvalidActionRuntimeProvider())


def test_registered_function_definition_must_match_ontology():
    class MismatchedFunctionProvider(Provider):
        def register(self, context):
            context.sources.register("memory", lambda **kwargs: MemorySource())
            context.bindings.register(
                "domain_name",
                lambda: "wrong",
                type(context.ontology.functions["domain_name"])(summary="Different"),
            )

    with pytest.raises(ValueError, match="do not match"):
        load_domain(MismatchedFunctionProvider())


def test_provider_cannot_register_function_absent_from_ontology():
    class ExtraFunctionProvider(Provider):
        def register(self, context):
            super().register(context)
            context.bindings.register(
                "extra",
                lambda: "extra",
                type(context.ontology.functions["domain_name"])(summary="Extra"),
            )

    with pytest.raises(ValueError, match="absent from the ontology: extra"):
        load_domain(ExtraFunctionProvider())


def test_every_declared_source_type_requires_a_registered_factory():
    class MissingSourceFactoryProvider(Provider):
        def register(self, context):
            context.bindings.register(
                "domain_name",
                lambda: "domain",
                context.ontology.functions["domain_name"],
            )

    with pytest.raises(ValueError, match="No source factory registered for: memory"):
        load_domain(MissingSourceFactoryProvider())
