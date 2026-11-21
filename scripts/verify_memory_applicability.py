from audit_memory_applicability import bag_cosine, destinations, evaluate, load_sources, target_status, task_description


def main():
    assert task_description("Task: move to the blue box.\nThis room is called kitchen. red box") == "Task: move to the blue box."
    assert destinations("the blue box and the blue box") == ["blue"]
    query = {"id": "query", "family": "find-living-thing", "text": "Task: move to the blue box.", "ordinal": 1, "task_id": "test:1:train"}
    rule = {"id": "rule", "family": query["family"], "text": "Move to the red box.", "source_ordinal": 0, "source_task_id": "test:0:train"}
    assert target_status(query, rule) == "explicit_destination_mismatch"
    assert target_status(query, {**rule, "text": "Avoid invalid actions."}) == "unknown"
    assert target_status(query, {**rule, "text": "Move to the blue box."}) == "same_destination"
    assert target_status(query, {**rule, "family": "other"}) == "not_comparable"
    assert abs(bag_cosine("a b", "a b") - 1) < 1e-10
    assert bag_cosine("a", "b") == 0
    rows = evaluate([query], [rule])
    assert len(rows) == 12
    for row in rows:
        assert row["selected_count"] == (0 if row["route"] == "family_explicit_scope_top2" else 1)
    blocked = evaluate([query], [{**rule, "source_ordinal": 1, "source_task_id": query["task_id"]}])
    assert all(not row["selected"] for row in blocked)
    queries, rules = load_sources()
    assert len(queries) == 12 and len(rules) == 8
    assert all("This room is called" not in query["text"] for query in queries)
    real = evaluate(queries, rules)
    assert len(real) == 144
    for row in real:
        assert row["selected_count"] <= 2
        if row["route"] == "family_explicit_scope_top2":
            assert row["mismatch_count"] == row["cross_family_count"] == 0
        if row["regime"] == "chronological_availability" and row["query"]["ordinal"] == 0:
            assert not row["ranked_available_bank"]
    print("APPLICABILITY_WITNESS: boundary, slots, unknowns, chronology, source exclusion,144diagnostic rows; noAPI or efficacy")


if __name__ == "__main__":
    main()
