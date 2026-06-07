import threading
import time
from pathlib import Path

from tinyagent.daemon import DaemonEngine, serve_engine


class FakeEngine:
    def __init__(self):
        self.generated = []
        self.counted = []

    def generate(self, prompt, max_tokens, on_token=None):
        self.generated.append((prompt, max_tokens))
        text = "hi"
        if on_token is not None:
            on_token("h", 1)
            on_token("i", 2)
        return text

    def count_tokens(self, text):
        self.counted.append(text)
        return len(text.split())


def _start_fake_daemon(tmp_path, requests=3):
    socket_path = Path("/tmp") / f"tinyagent-test-{time.monotonic_ns()}.sock"
    engine = FakeEngine()
    thread = threading.Thread(target=serve_engine, args=(engine, socket_path, requests), daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and not socket_path.exists():
        time.sleep(0.01)
    assert socket_path.exists()
    return socket_path, engine, thread


def test_daemon_engine_counts_tokens(tmp_path):
    socket_path, engine, thread = _start_fake_daemon(tmp_path, requests=1)
    client = DaemonEngine("fake.gguf", 8, socket_path=socket_path, auto_start=False)

    assert client.count_tokens("one two three") == 3
    thread.join(timeout=2)
    assert engine.counted == ["one two three"]


def test_daemon_engine_streams_generate(tmp_path):
    socket_path, engine, thread = _start_fake_daemon(tmp_path, requests=1)
    client = DaemonEngine("fake.gguf", 8, socket_path=socket_path, auto_start=False)
    chunks = []

    assert client.generate("hello", 4, on_token=lambda chunk, _i: chunks.append(chunk)) == "hi"
    thread.join(timeout=2)
    assert chunks == ["h", "i"]
    assert engine.generated == [("hello", 4)]
