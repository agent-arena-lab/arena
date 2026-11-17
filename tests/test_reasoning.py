from arena.llm import parse_chat_response
from arena.reasoning import extract_tool_calls_from_text, split_reasoning


def test_think_block():
    r = split_reasoning("<think>plan</think>Hello")
    assert r.visible == "Hello" and r.reasoning == "plan"


def test_orphan_close_qwen_template():
    r = split_reasoning("I should share.\n</think>\n\nDone")
    assert r.reasoning == "I should share." and r.visible == "Done"


def test_unterminated_think():
    r = split_reasoning("<think>still thinking")
    assert r.visible == "" and "still thinking" in r.reasoning and "unterminated_think" in r.notes


def test_harmony_channels_and_call():
    txt = ("<|channel|>analysis<|message|>secret plan<|end|><|start|>assistant<|channel|>commentary "
           "to=functions.send_message <|constrain|>json<|message|>{\"text\": \"hi\"}<|call|>")
    r = split_reasoning(txt)
    assert r.reasoning == "secret plan"
    assert r.tool_calls[0].name == "send_message" and r.tool_calls[0].arguments == {"text": "hi"}


def test_qwen_tool_call_tag_fallback():
    calls = extract_tool_calls_from_text('<tool_call>\n{"name": "end_turn", "arguments": {}}\n</tool_call>')
    assert calls[0].name == "end_turn"


def test_parse_response_native_and_reasoning_content():
    data = {"choices": [{"finish_reason": "tool_calls", "message": {
        "content": "", "reasoning_content": "think",
        "tool_calls": [{"id": "x", "function": {"name": "submit_answer", "arguments": "{\"total_eur\": 7.5}"}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
    r = parse_chat_response(data, 0.1)
    assert r.reasoning == "think" and r.tool_calls[0].arguments == {"total_eur": 7.5}
    assert r.tool_calls[0].source == "native"


def test_parse_response_bad_json_args():
    data = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "x", "function": {"name": "send_message", "arguments": "{bad"}}]}}]}
    r = parse_chat_response(data, 0)
    assert r.tool_calls[0].parse_error
