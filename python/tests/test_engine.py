import pytest

from tinyagent.engine import TinyEngine
from tinyengine.runtime import TinyEngineError


class FakeModel:
    def __init__(self, _path, _options):
        self.generate_calls = []
        self.tokenize_calls = 0

    def generate_raw(self, prompt, max_tokens, on_token=None):
        self.generate_calls.append((prompt, max_tokens))
        return "ok"

    def tokenize(self, text, parse_special=True):
        self.tokenize_calls += 1
        return list(range(len(text.split())))


class OverflowOnceModel(FakeModel):
    def generate_raw(self, prompt, max_tokens, on_token=None):
        self.generate_calls.append((prompt, max_tokens))
        if len(self.generate_calls) == 1:
            raise TinyEngineError("unsupported")
        return "capped"


class UnsupportedModel(FakeModel):
    def generate_raw(self, prompt, max_tokens, on_token=None):
        self.generate_calls.append((prompt, max_tokens))
        raise TinyEngineError("unsupported")


def test_tinyengine_generate_avoids_fast_path_pre_count(monkeypatch):
    created = []

    def make_model(path, options):
        model = FakeModel(path, options)
        created.append(model)
        return model

    monkeypatch.setattr("tinyengine.runtime.Model", make_model)
    engine = TinyEngine("model.gguf", context_tokens=16)

    assert engine.generate("one two", 4) == "ok"
    assert created[0].generate_calls == [("one two", 4)]
    assert created[0].tokenize_calls == 0


def test_tinyengine_generate_retries_with_context_cap(monkeypatch):
    created = []

    def make_model(path, options):
        model = OverflowOnceModel(path, options)
        created.append(model)
        return model

    monkeypatch.setattr("tinyengine.runtime.Model", make_model)
    engine = TinyEngine("model.gguf", context_tokens=16)

    assert engine.generate("one two three four five", 8) == "capped"
    assert created[0].generate_calls == [
        ("one two three four five", 8),
        ("one two three four five", 3),
    ]
    assert created[0].tokenize_calls == 1


def test_tinyengine_generate_reraises_non_context_unsupported(monkeypatch):
    created = []

    def make_model(path, options):
        model = UnsupportedModel(path, options)
        created.append(model)
        return model

    monkeypatch.setattr("tinyengine.runtime.Model", make_model)
    engine = TinyEngine("model.gguf", context_tokens=16)

    with pytest.raises(TinyEngineError, match="unsupported"):
        engine.generate("one two", 4)
    assert created[0].generate_calls == [("one two", 4)]
    assert created[0].tokenize_calls == 1


def test_tinyengine_generate_reports_prompt_context_overflow(monkeypatch):
    created = []

    def make_model(path, options):
        model = OverflowOnceModel(path, options)
        created.append(model)
        return model

    monkeypatch.setattr("tinyengine.runtime.Model", make_model)
    engine = TinyEngine("model.gguf", context_tokens=8)

    with pytest.raises(TinyEngineError, match="prompt exceeds context window"):
        engine.generate("one two three four five", 8)
    assert created[0].generate_calls == [("one two three four five", 8)]
    assert created[0].tokenize_calls == 1
