from tinyagent.streaming import StreamFilter


def _run(chunks):
    out = []
    flt = StreamFilter(out.append)
    for c in chunks:
        flt.feed(c)
    flt.close()
    return "".join(out)


def test_plain_text_passes_through():
    assert _run(["Hello ", "world", "!"]) == "Hello world!"


def test_tool_call_span_is_removed():
    chunks = ["Before ", '<tool_call>{"name": "x"}</tool_call>', " after"]
    assert _run(chunks) == "Before  after"


def test_tag_split_across_chunks_is_removed():
    # The opening tag is split across several chunks; nothing of it must leak.
    chunks = ["text <to", "ol_c", "all>", '{"a":1}', "</tool", "_call> done"]
    assert _run(chunks) == "text  done"


def test_partial_open_tag_at_end_is_not_leaked_until_resolved():
    out = []
    flt = StreamFilter(out.append)
    flt.feed("hi <tool")          # could be the start of <tool_call>
    assert "".join(out) == "hi "   # the "<tool" prefix is withheld
    flt.feed("box>")               # turns out NOT to be a tool call
    flt.close()
    assert "".join(out) == "hi <toolbox>"


def test_unterminated_tool_call_is_fully_suppressed():
    chunks = ["keep ", "<tool_call>", '{"name":"x"']  # stream ends mid-call
    assert _run(chunks) == "keep "
