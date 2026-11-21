from dataclasses import asdict, dataclass
import re

from .text import TokenCounter


def parse_operations(response):
    pattern = r'((?:REMOVE|EDIT|ADD|AGREE)(?: \d+|)): (?:[a-zA-Z\s\d]+: |)(.*)'
    operations = []
    for operation, text in re.findall(pattern, response):
        text = text.strip()
        if text and not any(word in text for word in ("ADD", "AGREE", "EDIT")) and text.endswith("."):
            operations.append(("ADD" if "ADD" in operation else operation.strip(), text))
    return operations


@dataclass
class Insight:
    identifier: int
    text: str
    importance: int = 2


class InsightBank:
    def __init__(self, rules=(), max_num_rules=20, max_items=None, max_tokens=None, counter=None):
        for value in (max_num_rules, max_items, max_tokens):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError("capacities must be positive integers")
        if max_num_rules is None:
            raise ValueError("max_num_rules is required")
        self.max_num_rules = max_num_rules
        self.max_items = max_items
        self.max_tokens = max_tokens
        self.counter = counter or TokenCounter()
        self.rules = []
        self.next_identifier = 0
        for text, importance in rules:
            if not isinstance(text, str) or not text.strip() or isinstance(importance, bool) or not isinstance(importance, int) or importance < 1:
                raise ValueError("initial insights require nonempty text and positive integer importance")
            self.rules.append(Insight(self.next_identifier, text, importance))
            self.next_identifier += 1
        self.initial_capacity_removals = self._enforce_capacity()

    def snapshot(self):
        return [asdict(rule) for rule in self.rules]

    def pairs(self):
        return [(rule.text, rule.importance) for rule in self.rules]

    def _match(self, text):
        return next((index for index, rule in enumerate(self.rules) if rule.text in text), None)

    def _enforce_capacity(self):
        removed = []
        while self.rules and ((self.max_items is not None and len(self.rules) > self.max_items) or
                              (self.max_tokens is not None and sum(self.counter.count(rule.text) for rule in self.rules) > self.max_tokens)):
            worst = min(self.rules, key=lambda rule: (rule.importance, rule.identifier))
            removed.append(asdict(worst))
            self.rules.remove(worst)
        return removed

    def apply(self, response):
        before = self.snapshot()
        operations = parse_operations(response)
        accepted = []
        rejected = []
        normalized = []
        list_full = len(self.rules) >= self.max_num_rules + 5
        for operation, text in operations:
            pieces = operation.split()
            kind = pieces[0]
            number = int(pieces[1]) if len(pieces) > 1 else None
            match = self._match(text)
            reason = None
            if kind == "ADD" and match is not None:
                reason = "existing_substring"
            elif kind == "EDIT":
                if match is not None:
                    normalized.append({"original": [operation, text], "effective": [f"AGREE {match+1}", self.rules[match].text]})
                    kind, text, number = "AGREE", self.rules[match].text, match+1
                elif number is None or not 1 <= number <= len(self.rules):
                    reason = "invalid_edit_index"
            elif kind in ("REMOVE", "AGREE") and match is None:
                reason = "no_matching_text"
            if reason:
                rejected.append({"operation": operation, "text": text, "reason": reason})
            else:
                accepted.append((kind, number, text))
        for kind in ("REMOVE", "AGREE", "EDIT", "ADD"):
            for operation_kind, number, text in accepted:
                if kind != operation_kind:
                    continue
                if kind == "ADD":
                    self.rules.append(Insight(self.next_identifier, text))
                    self.next_identifier += 1
                elif kind == "EDIT":
                    self.rules[number-1].text = text
                    self.rules[number-1].importance += 1
                else:
                    index = self._match(text)
                    self.rules[index].importance += 1 if kind == "AGREE" else (-3 if list_full else -1)
        count_removals = [asdict(rule) for rule in self.rules if rule.importance <= 0]
        self.rules = [rule for rule in self.rules if rule.importance > 0]
        self.rules.sort(key=lambda rule: rule.importance, reverse=True)
        capacity_removals = self._enforce_capacity()
        return {"raw_response": response, "parsed_operations": operations, "accepted_operations": accepted,
                "rejected_operations": rejected, "normalized_operations": normalized, "list_full": list_full,
                "before": before, "after": self.snapshot(), "count_removals": count_removals,
                "capacity_removals": capacity_removals}
