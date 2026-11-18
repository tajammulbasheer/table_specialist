from ts.eval.self_consistency import result_key, vote


def test_result_key_ignores_order_and_drops_failures():
    assert result_key([[1, "a"], [2.0000001, "b"]]) == result_key([[2.0, "b"], [1, "a"]])
    assert result_key("SQLExecutionError") is None and result_key([]) is None


def test_vote_takes_largest_group_and_prefers_earlier():
    a, b = ("a",), ("b",)
    assert vote([a, b, b, a, b]) == 1
    assert vote([a, b, a, b]) == 0          # tie: the earlier answer (greedy first)
    assert vote([None, b, None, b, a]) == 1  # failed or empty answers do not vote
    assert vote([None, None]) == 0           # nothing usable: keep the greedy answer


def test_instruct_training_examples_use_the_repo_chat_form():
    from ts import loop
    example = loop.as_chat({"prompt": "P", "completion": "C"})
    assert example["prompt"][1] == {"role": "user", "content": "P"}
    assert example["completion"] == [{"role": "assistant", "content": "C"}]
    assert example["prompt"][0]["role"] == "system"
