from soc_agent import new_findings

seen = set()
listing = {"findings": [{"id": "a"}, {"id": "b"}, {"title": "no id"}]}
assert [f["id"] for f in new_findings(listing, seen)] == ["a", "b"]
assert new_findings(listing, seen) == []  # 두 번째 폴링에서는 중복 보고 안 함
assert [f["id"] for f in new_findings({"findings": [{"id": "c"}]}, seen)] == ["c"]
assert new_findings({"count": 0}, seen) == []
print("ok")
