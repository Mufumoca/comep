from dataclasses import asdict
from itertools import combinations
import csv
import json
from pathlib import Path
import platform
import time

import numpy as np

from . import __version__
from .core import Config, MemoryManager, sigmoid
from .text import LocalEncoder, TokenCounter


LEGACY_SCENARIOS = ("additive", "poison", "conflict", "synergy", "mixed", "higher-order")
AGENT_SCENARIOS = (
    "agent-additive",
    "agent-poison",
    "agent-explicit-conflict",
    "agent-implicit-conflict",
    "agent-synergy",
    "agent-composite",
)
SCENARIOS = LEGACY_SCENARIOS + AGENT_SCENARIOS


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def new_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"refusing to overwrite nonempty experiment directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class SyntheticWorld:
    def __init__(self, seed, scenario="mixed", size=16, poison_fraction=0.2):
        if scenario not in LEGACY_SCENARIOS or size < 8 or not 0 <= poison_fraction <= 1:
            raise ValueError("invalid synthetic world configuration")
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.rules = []
        self.truth = {}
        self.pairs = {}
        topics = ("navigation", "cleaning", "heating", "sorting")
        permutation = self.rng.permutation(size)
        poisoned = set(self.rng.choice(size, int(size * poison_fraction), replace=False)) if scenario in ("poison", "mixed") else set()
        for index in range(size):
            identifier = f"memory-{int(permutation[index]):03d}"
            topic = index % len(topics)
            effect = -1.8 if index in poisoned else 0.7
            if index % 7 == 6 and index not in poisoned:
                effect = 0.0
            text = f"For {topics[topic]} tasks check the state before action procedure {index}."
            self.rules.append({"id": identifier, "text": text})
            self.truth[identifier] = {"topic": topic, "effect": effect, "poison": index in poisoned, "synergy_member": False}
        if scenario in ("conflict", "mixed"):
            for offset in (0, 1):
                self.pairs[tuple(sorted((self.rules[offset]["id"], self.rules[offset + 4]["id"])))] = -3.5
        if scenario in ("synergy", "mixed"):
            first, second = self.rules[2]["id"], self.rules[6]["id"]
            self.truth[first]["effect"] = -0.2
            self.truth[second]["effect"] = -0.2
            self.truth[first]["synergy_member"] = True
            self.truth[second]["synergy_member"] = True
            self.pairs[tuple(sorted((first, second)))] = 3.5
        self.triple = {self.rules[index]["id"] for index in (0, 4, 7)}
        self.topics = topics

    def tasks(self, episodes):
        contexts = self.rng.integers(0, len(self.topics), size=episodes)
        uniforms = self.rng.random(episodes)
        return [(int(context), 0, float(uniform)) for context, uniform in zip(contexts, uniforms)]

    def query(self, context, variant=0):
        return f"For {self.topics[context]} tasks check the state before action"

    def probability(self, context, selected):
        gates = {identifier: 1.0 if self.truth[identifier]["topic"] == context else 0.15 for identifier in selected}
        latent = (-1.2, -0.5, -1.7, -0.8)[context]
        latent += sum(gates[identifier] * self.truth[identifier]["effect"] for identifier in selected)
        latent += sum(gates[first] * gates[second] * self.pairs.get(tuple(sorted((first, second))), 0.0) for first, second in combinations(selected, 2))
        if self.scenario == "higher-order" and self.triple <= set(selected):
            latent += -4.0
        return sigmoid(latent)

    def export_truth(self):
        return {
            "scenario": self.scenario,
            "rules": self.truth,
            "pairs": [{"identifiers": list(pair), "effect": effect} for pair, effect in self.pairs.items()],
            "higher_order_triple": sorted(self.triple) if self.scenario == "higher-order" else [],
        }


AGENT_TASK_FAMILIES = (
    {
        "name": "pick_clean_then_place",
        "base_logit": -0.95,
        "queries": (
            "Task: clean a dirty mug and put it in the coffeemachine in the kitchen.",
            "Task: wash a soiled pan at the sinkbasin and place it on the countertop.",
            "Task: rinse a plate in the kitchen sinkbasin and store it inside the cabinet.",
        ),
    },
    {
        "name": "pick_heat_then_place",
        "base_logit": -1.10,
        "queries": (
            "Task: heat an egg in the microwave and put the hot egg on the diningtable.",
            "Task: warm a mug or potato using the kitchen microwave and place it on the countertop.",
            "Task: heat a slice of bread in the microwave and set it on the sidetable.",
        ),
    },
    {
        "name": "pick_cool_then_place",
        "base_logit": -1.05,
        "queries": (
            "Task: cool a fresh apple in the fridge and place the chilled apple on the diningtable.",
            "Task: chill a tomato or bowl inside the kitchen fridge and put it on the countertop.",
            "Task: cool a lettuce head in the fridge before placing it on the garbagecan counter.",
        ),
    },
    {
        "name": "look_at_obj_in_light",
        "base_logit": -0.90,
        "queries": (
            "Task: examine a creditcard or alarmclock under the light of a desklamp in the bedroom.",
            "Task: find a book on the desk or shelf and inspect it under an illuminated desklamp.",
            "Task: hold a keychain or cd and look at it beneath a switched-on desklamp.",
        ),
    },
    {
        "name": "pick_two_obj_and_place",
        "base_logit": -1.15,
        "queries": (
            "Task: find two pillows or remotecontrols in the livingroom and put both on the sofa.",
            "Task: collect two spraybottles or soapbars in the bathroom and place both in the cabinet.",
            "Task: gather two shakers or mugs and place both items together on the diningtable.",
        ),
    },
    {
        "name": "pick_and_place_simple",
        "base_logit": -0.75,
        "queries": (
            "Task: pick up a pencil or pen from the desk and put it inside the drawer.",
            "Task: locate a clean towel on the rack and transfer it into the bathroom basin cabinet.",
            "Task: move a tissuebox from the sidetable and place it onto the dresser.",
        ),
    },
)


def _build_agent_catalog(scenario: str) -> tuple[list[dict], list[tuple[int, int, float]]]:
    """Construct realistic household-agent memory rules and pairwise ground-truth interactions."""
    entries: list[dict] = []
    pair_specs: list[tuple[int, int, float]] = []

    def add_rule(topic: int, text: str, effect: float, poison: bool = False, role: str = "standard", cross_topics: tuple[int, ...] = ()):
        idx = len(entries)
        entries.append({
            "topic": topic,
            "cross_topics": cross_topics,
            "text": text,
            "effect": effect,
            "poison": poison,
            "role": role,
            "synergy_member": role == "synergy",
        })
        return idx

    if scenario == "agent-additive":
        templates = [
            # Pure zero-interaction, zero-poison negative control (strong, moderate, weak, neutral verbose distractor)
            (0, "When cleaning kitchen mugs, pans, or plates, stand at the sinkbasin and issue clean item with sinkbasin directly.", 1.35, False, "strong"),
            (0, "Dirty mugs, plates, and pans usually sit on the stoveburner, diningtable, or sinkbasin countertop.", 0.90, False, "moderate"),
            (0, "For kitchen cleaning tasks, inspect open countertops before opening closed lower drawers.", 0.25, False, "weak"),
            (0, "For kitchen mug and pan cleaning tasks, note the general room layout around the sinkbasin and coffeemachine.", 0.0, False, "neutral"),
            (1, "To heat eggs, mugs, or bread in the microwave, stand at the closed microwave and run heat directly.", 1.40, False, "strong"),
            (1, "Eggs, potatoes, and bread slices to warm are usually inside the kitchen fridge or on the countertop.", 0.90, False, "moderate"),
            (1, "Confirm your inventory holds the food item before walking to the kitchen microwave.", 0.25, False, "weak"),
            (1, "For kitchen microwave heating of eggs or bread, observe the surrounding diningtable and countertop area.", 0.0, False, "neutral"),
            (2, "To cool apples, tomatoes, or bowls, stand at the kitchen fridge and run cool item with fridge.", 1.40, False, "strong"),
            (2, "Produce for fridge cooling tasks usually starts on the diningtable, countertop, or garbagecan counter.", 0.85, False, "moderate"),
            (2, "After chilling an item in the fridge, walk straight to the target diningtable or countertop.", 0.25, False, "weak"),
            (2, "For kitchen fridge cooling of apples or tomatoes, take note of nearby kitchen receptacles.", 0.0, False, "neutral"),
            (3, "To inspect a creditcard, book, or keychain in light, hold the item and toggle on a nearby desklamp.", 1.35, False, "strong"),
            (3, "Creditcards, alarmclocks, and keychains in the bedroom are usually on the desk, dresser, or sidetable.", 0.85, False, "moderate"),
            (3, "Keep holding the target item when switching on the bedroom desklamp so examination triggers.", 0.25, False, "weak"),
            (3, "For bedroom desklamp inspection of creditcards or books, note the furniture arrangement near the desk.", 0.0, False, "neutral"),
            (4, "Deliver the first pillow, spraybottle, or mug to the target receptacle before searching for the second matching item.", 1.45, False, "strong"),
            (4, "Remember where you placed the first item so you do not pick up the delivered object again.", 0.90, False, "moderate"),
            (4, "Scan sofas, armchairs, and bathroom counters systematically when gathering two matching items.", 0.25, False, "weak"),
            (4, "When gathering two pillows, spraybottles, or shakers, observe the room receptacles along the way.", 0.0, False, "neutral"),
            (5, "For simple pick and place of a pencil, towel, or tissuebox, check desks and racks then open closed target drawers.", 1.25, False, "strong"),
            (5, "Open a closed drawer or cabinet door before issuing put item in/on that receptacle.", 0.80, False, "moderate"),
            (5, "Check exact object numbering in admissible actions when picking up pens, pencils, or towels.", 0.25, False, "weak"),
            (5, "When moving a pencil, towel, or tissuebox into a drawer or dresser, note the nearby room surfaces.", 0.0, False, "neutral"),
        ]
        for t, txt, eff, psn, role in templates:
            add_rule(t, txt, eff, psn, role)

    elif scenario == "agent-poison":
        templates = [
            # Guilt-by-association poison: high-relevance failure-distilled rules (-1.75) co-occurring with valid domain rules
            (0, "When cleaning kitchen mugs, pans, or plates, stand at the sinkbasin and issue clean item with sinkbasin directly.", 1.40, False, "strong"),
            (0, "Dirty mugs, plates, and pans usually sit on the stoveburner, diningtable, or sinkbasin countertop.", 0.85, False, "moderate"),
            (0, "For kitchen cleaning tasks, inspect open countertops before opening closed lower drawers.", 0.25, False, "weak"),
            (0, "When cleaning dirty mugs, pans, or plates at the kitchen sinkbasin, dump the vessel in the garbagecan first.", -1.80, True, "poison"),
            (1, "To heat eggs, mugs, or bread in the microwave, stand at the closed microwave and run heat directly.", 1.45, False, "strong"),
            (1, "Eggs, potatoes, and bread slices to warm are usually inside the kitchen fridge or on the countertop.", 0.85, False, "moderate"),
            (1, "Confirm your inventory holds the food item before walking to the kitchen microwave.", 0.25, False, "weak"),
            (1, "When heating eggs, mugs, or bread in the kitchen microwave, place the raw item on the diningtable and stop.", -1.80, True, "poison"),
            (2, "To cool apples, tomatoes, or bowls, stand at the kitchen fridge and run cool item with fridge.", 1.45, False, "strong"),
            (2, "Produce for fridge cooling tasks usually starts on the diningtable, countertop, or garbagecan counter.", 0.85, False, "moderate"),
            (2, "After chilling an item in the fridge, walk straight to the target diningtable or countertop.", 0.25, False, "weak"),
            (2, "When cooling apples, tomatoes, or bowls in the kitchen fridge, heat the produce in the microwave instead.", -1.80, True, "poison"),
            (3, "To inspect a creditcard, book, or keychain in light, hold the item and toggle on a nearby desklamp.", 1.40, False, "strong"),
            (3, "Creditcards, alarmclocks, and keychains in the bedroom are usually on the desk, dresser, or sidetable.", 0.85, False, "moderate"),
            (3, "Keep holding the target item when switching on the bedroom desklamp so examination triggers.", 0.25, False, "weak"),
            (3, "When examining a creditcard, book, or keychain under a bedroom desklamp, lock the item inside a closed drawer.", -1.80, True, "poison"),
            (4, "Deliver the first pillow, spraybottle, or mug to the target receptacle before searching for the second matching item.", 1.45, False, "strong"),
            (4, "Remember where you placed the first item so you do not pick up the delivered object again.", 0.85, False, "moderate"),
            (4, "Scan sofas, armchairs, and bathroom counters systematically when gathering two matching items.", 0.25, False, "weak"),
            (4, "When collecting two pillows, spraybottles, or mugs for the sofa or cabinet, repeatedly swap both items on the floor.", -1.85, True, "poison"),
            (5, "For simple pick and place of a pencil, towel, or tissuebox, check desks and racks then open closed target drawers.", 1.30, False, "strong"),
            (5, "Open a closed drawer or cabinet door before issuing put item in/on that receptacle.", 0.80, False, "moderate"),
            (5, "Check exact object numbering in admissible actions when picking up pens, pencils, or towels.", 0.25, False, "weak"),
            (5, "When moving a pencil, towel, or tissuebox into a desk drawer or cabinet, discard the object in the trashcan.", -1.80, True, "poison"),
        ]
        for t, txt, eff, psn, role in templates:
            add_rule(t, txt, eff, psn, role)

    elif scenario == "agent-explicit-conflict":
        specs = [
            # Family 0: explicit contradiction on sinkbasin cleaning
            (0, "Always clean dirty mugs, pans, and plates directly at the sinkbasin without opening or toggling the faucet separately.", 1.45, False, "valid"),
            (0, "Never run clean directly at the sinkbasin; forbid direct cleaning and always skip the sinkbasin command.", -1.45, True, "explicit_contradiction"),
            (0, "Check the stoveburner and diningtable first when searching for dirty mugs, pans, or plates.", 0.55, False, "helper"),
            (0, "Place clean vessels into the target coffeemachine or countertop as soon as cleaning completes.", 0.35, False, "helper"),
            # Family 1: explicit contradiction on microwave door
            (1, "Always keep the microwave door closed and issue heat item with microwave directly when warming eggs, mugs, or bread.", 1.50, False, "valid"),
            (1, "Never use heat on a closed microwave; always open the microwave door first and avoid closed-door microwave heating.", -1.50, True, "explicit_contradiction"),
            (1, "Look inside the fridge and on kitchen countertops when locating an egg or potato to heat.", 0.55, False, "helper"),
            (1, "Carry the heated food item directly to the diningtable or countertop after microwave heating.", 0.35, False, "helper"),
            # Family 2: explicit contradiction on fridge cooling
            (2, "Always cool apples, tomatoes, and bowls inside the kitchen fridge using the cool command.", 1.45, False, "valid"),
            (2, "Never open or use the fridge to cool an apple or tomato; avoid the fridge completely and leave produce warm.", -1.50, True, "explicit_contradiction"),
            (2, "Check diningtables and kitchen counters first to pick up apples or tomatoes for cooling.", 0.55, False, "helper"),
            (2, "Verify the chilled item is placed onto the requested table after using the fridge.", 0.35, False, "helper"),
            # Family 3: explicit contradiction on desklamp examination
            (3, "Always hold the creditcard, book, or keychain in inventory while turning on the desklamp to examine it.", 1.45, False, "valid"),
            (3, "Never hold any item when using a desklamp; drop the creditcard or book first and skip examining under light.", -1.45, True, "explicit_contradiction"),
            (3, "Search desks, dressers, and sidetables first for small bedroom objects like creditcards or alarmclocks.", 0.50, False, "helper"),
            (3, "Locate the desklamp on the desk or sidetable before picking up the object to inspect.", 0.35, False, "helper"),
            # Family 4: explicit contradiction on two-object delivery
            (4, "Always place the first pillow, mug, or spraybottle at the destination receptacle before picking up the second object.", 1.50, False, "valid"),
            (4, "Never put down the first object before finding the second; forbid single-item trips and always hold both at once.", -1.55, True, "explicit_contradiction"),
            (4, "Track which sofa or cabinet already received your first item so you pick a distinct second instance.", 0.50, False, "helper"),
            (4, "Check armchairs, sofas, and bathroom counters systematically for matching pairs of items.", 0.35, False, "helper"),
            # Family 5: explicit contradiction on enclosed drawers
            (5, "Always open a closed desk drawer or bathroom cabinet before trying to put a pencil or towel inside it.", 1.35, False, "valid"),
            (5, "Never open closed drawers or cabinets before placing items; skip opening doors and force put directly.", -1.40, True, "explicit_contradiction"),
            (5, "Look on the desk surface and sidetable first for pens, pencils, and tissueboxes.", 0.45, False, "helper"),
            (5, "Match exact target receptacle names when transferring towels or stationery.", 0.30, False, "helper"),
        ]
        for fam in range(6):
            base = fam * 4
            for t, txt, eff, psn, role in specs[base : base + 4]:
                add_rule(t, txt, eff, psn, role)
            pair_specs.append((base, base + 1, -2.5))

    elif scenario == "agent-implicit-conflict":
        # Two individually effective, positively phrased workflows per family (<=24 tokens each) that clash via inventory/state/step budget at runtime
        for fam, (wf_a, wf_b, helper_c, helper_d) in enumerate([
            (
                "For dirty mugs, pans, or plates, carry the vessel to sinkbasin and run clean while holding it.",
                "For dirty mugs, pans, or plates at sinkbasin, place the vessel inside sinkbasin, cycle water, then re-take it.",
                "Check stoveburners and diningtables for dirty kitchenware before upper cabinets.",
                "Walk straight to the coffeemachine or countertop once cleaning finishes.",
            ),
            (
                "When heating an egg, mug, or bread in the microwave, carry the food item straight to microwave and heat.",
                "Before microwave heating of an egg, mug, or bread, clear diningtable clutter aside first then fetch the food.",
                "Check kitchen countertops and fridge shelves for raw eggs, bread, or potatoes.",
                "Place the heated food onto the exact diningtable or countertop named in the goal.",
            ),
            (
                "For fridge cooling of apples, tomatoes, or bowls, carry the item straight to the fridge and cool immediately.",
                "For fridge cooling of apples, tomatoes, or bowls, hold a reference bowl while surveying every fridge shelf first.",
                "Scan diningtables and kitchen counters first to locate apples or tomatoes.",
                "Deliver chilled produce directly onto the target diningtable or countertop.",
            ),
            (
                "To inspect a creditcard, book, or keychain under light, pick up the object first and carry it to a desklamp.",
                "To inspect a creditcard, book, or keychain under light, test every bedroom desklamp empty-handed first then fetch it.",
                "Look on bedroom desks and dressers for creditcards, books, or keychains.",
                "Toggle the desklamp once the target object and lamp are together.",
            ),
            (
                "When gathering two pillows, spraybottles, or mugs, deliver the first item straight to the target sofa or cabinet.",
                "When gathering two pillows, spraybottles, or mugs, stage the first item on a central table while sweeping rooms.",
                "Track the ID of the first item so you pick a distinct second object.",
                "Check sofas, armchairs, and bathroom counters for matching items.",
            ),
            (
                "When transferring a pencil, towel, or tissuebox into a drawer or cabinet, carry it there and open on arrival.",
                "When transferring a pencil, towel, or tissuebox into a drawer or cabinet, pre-open all room drawers empty-handed first.",
                "Check open desk and sidetable surfaces first for stationery or towels.",
                "Put the item cleanly inside the target drawer or cabinet.",
            ),
        ]):
            i_a = add_rule(fam, wf_a, 1.55, False, "workflow_a")
            i_b = add_rule(fam, wf_b, 1.50, False, "workflow_b")
            add_rule(fam, helper_c, 0.20, False, "weak_helper")
            add_rule(fam, helper_d, 0.15, False, "weak_helper")
            pair_specs.append((i_a, i_b, -4.10))

    elif scenario == "agent-synergy":
        # Each family has a complementary (Locator + Operator) sub-skill pair that requires BOTH rules (+3.70 together, -0.20 solo),
        # competing against 2 surface-attractive generic heuristics (+0.42 standalone) under tight retrieval limits.
        family_specs = [
            (
                "Hidden kitchen mugs, pans, and plates for cleaning are stored inside closed upper cabinets and sink drawers rather than open counters.",
                "At the sinkbasin, invoke clean item with sinkbasin directly in one command without toggling the faucet or dropping the vessel.",
                "For kitchen mug and pan cleaning tasks, check visible countertops around the coffeemachine and sinkbasin first.",
                "When cleaning kitchen items for the coffeemachine or countertop, verify the target surface is reachable.",
            ),
            (
                "Raw eggs, potatoes, and bread slices for microwave heating are kept inside the closed kitchen fridge rather than open tables.",
                "At the microwave, execute heat item with microwave directly on the closed appliance without opening the microwave door.",
                "For microwave heating tasks with eggs, mugs, or bread, look around the kitchen microwave and diningtable first.",
                "When warming food in the kitchen microwave, walk toward the diningtable or countertop afterward.",
            ),
            (
                "Apples, tomatoes, and bowls requiring cooling are tucked inside closed pantry cabinets beside the garbagecan.",
                "At the fridge, issue cool item with fridge as a single macro command without manually opening shelves or waiting.",
                "For fridge cooling tasks with apples or tomatoes, inspect the kitchen fridge exterior and diningtable first.",
                "When chilling produce for the countertop or diningtable, keep track of the destination receptacle.",
            ),
            (
                "Target creditcards, keychains, and alarmclocks in bedroom tasks are hidden inside closed desk and nightstand drawers.",
                "To complete light inspection, hold the object in inventory and toggle the desklamp directly without putting the item down.",
                "For bedroom desklamp examination of a creditcard, book, or keychain, scan the desk surface and bed first.",
                "When inspecting objects under a bedroom desklamp, stand close to the desk or sidetable.",
            ),
            (
                "Second matching pillows, spraybottles, and shakers are stored inside closed bathroom or livingroom cabinets.",
                "With single-slot inventory, deposit the first item onto the target sofa or cabinet before opening cabinets for the second item.",
                "For tasks collecting two pillows, spraybottles, or mugs, look across open sofas and diningtables first.",
                "When placing two matching items on a sofa or cabinet, check the main room area.",
            ),
            (
                "Stationery pens, pencils, and spare towels are stored inside closed sidetable drawers rather than open desktops.",
                "Open the destination drawer or cabinet door first before executing put item in/on the enclosed receptacle.",
                "For simple pick and place tasks with pencils, towels, or tissueboxes, check the open desk and dresser top first.",
                "When moving a pencil or towel into a drawer or cabinet, confirm you are in the right room.",
            ),
        ]
        for fam, (loc_txt, op_txt, dist_1, dist_2) in enumerate(family_specs):
            i_loc = add_rule(fam, loc_txt, -0.20, False, "synergy")
            i_op = add_rule(fam, op_txt, -0.20, False, "synergy")
            add_rule(fam, dist_1, 0.42, False, "mediocre_distractor")
            add_rule(fam, dist_2, 0.38, False, "mediocre_distractor")
            pair_specs.append((i_loc, i_op, 3.75))

    elif scenario == "agent-composite":
        # Family 0 (clean): Compositional synergy (Locator + Operator) + mediocre distractor + guilt-by-association poison
        s0_a = add_rule(0, "Dirty mugs, pans, and plates for sinkbasin cleaning are stored inside closed upper cabinets and lower stove drawers.", -0.18, False, "synergy")
        s0_b = add_rule(0, "At the sinkbasin, execute clean item with sinkbasin directly in one step without toggling faucets or dropping the item.", -0.18, False, "synergy")
        add_rule(0, "For kitchen sinkbasin cleaning of mugs or pans, scan open countertops near the coffeemachine first.", 0.40, False, "mediocre_distractor")
        add_rule(0, "When cleaning a mug or pan in the kitchen, throw the item into the garbagecan to reset its dirty state.", -2.10, True, "poison")
        pair_specs.append((s0_a, s0_b, 3.75))

        # Family 1 (heat): Implicit execution conflict between 2 strong positive workflows + explicit contradiction + helper
        w1_a = add_rule(1, "When heating an egg, mug, or bread slice, carry it straight to the closed microwave and issue heat immediately.", 1.50, False, "workflow_a")
        w1_b = add_rule(1, "For microwave heating tasks, clear clutter off the target diningtable and pre-open kitchen cabinets before fetching the food.", 1.42, False, "workflow_b")
        e1_c = add_rule(1, "Never heat items in a closed microwave; always leave the microwave door wide open and skip the heat command.", -1.65, True, "explicit_contradiction")
        add_rule(1, "Check inside the fridge and on kitchen counters when looking for an egg or potato to warm.", 0.35, False, "weak_helper")
        pair_specs.append((w1_a, w1_b, -3.90))
        pair_specs.append((w1_a, e1_c, -2.50))

        # Family 2 (cool): Compositional synergy (Locator + Operator) + implicit conflict distractor pair
        s2_a = add_rule(2, "Apples, tomatoes, and bowls to be cooled are stored inside closed pantry cabinets next to the kitchen counter.", -0.15, False, "synergy")
        s2_b = add_rule(2, "At the kitchen fridge, run cool item with fridge directly as an atomic command without manual door toggling.", -0.15, False, "synergy")
        w2_c = add_rule(2, "For fridge cooling of apples or tomatoes, inspect open diningtables first and carry produce directly to the fridge.", 0.55, False, "workflow_a")
        w2_d = add_rule(2, "During fridge cooling tasks, keep holding a reference bowl from the counter while surveying every fridge shelf.", 0.50, False, "workflow_b")
        pair_specs.append((s2_a, s2_b, 3.65))
        pair_specs.append((w2_c, w2_d, -3.40))

        # Family 3 (light): Implicit workflow conflict + guilt-by-association poison + weak helper
        w3_a = add_rule(3, "To inspect a creditcard, book, or keychain in light, pick up the target object first and carry it to the desklamp.", 1.48, False, "workflow_a")
        w3_b = add_rule(3, "For bedroom desklamp tasks, keep hands empty while testing every lamp in the room first before fetching the item.", 1.42, False, "workflow_b")
        add_rule(3, "When examining a creditcard or alarmclock under a desklamp, unplug and turn off all lamps in the bedroom.", -2.00, True, "poison")
        add_rule(3, "Check bedroom desks, dressers, and sidetables for small items to inspect.", 0.30, False, "weak_helper")
        pair_specs.append((w3_a, w3_b, -3.85))

        # Family 4 (two_obj): Compositional synergy (Inventory + Cabinet Locator) + explicit contradiction + mediocre distractor
        s4_a = add_rule(4, "Deposit the first pillow, spraybottle, or shaker at the target receptacle immediately to free your single inventory slot.", -0.18, False, "synergy")
        s4_b = add_rule(4, "The second matching pillow, spraybottle, or shaker is stored inside closed cabinets rather than visible surfaces.", -0.18, False, "synergy")
        e4_c = add_rule(4, "Never deposit the first object early; forbid single-item trips and always hold two objects simultaneously.", -1.65, True, "explicit_contradiction")
        add_rule(4, "For tasks gathering two pillows or spraybottles, scan open sofas and bathroom counters first.", 0.42, False, "mediocre_distractor")
        pair_specs.append((s4_a, s4_b, 3.70))
        pair_specs.append((s4_a, e4_c, -2.50))

        # Family 5 (simple): Pure-additive control family with poison inside composite bank
        add_rule(5, "For simple pick and place of a pencil, towel, or tissuebox, check open desks and racks then open closed target drawers.", 1.35, False, "strong")
        add_rule(5, "Match exact receptacle numbering before issuing put item in/on drawer, cabinet, or dresser.", 0.85, False, "moderate")
        add_rule(5, "Scan nearby sidetables if the primary desk surface does not hold the target stationery.", 0.30, False, "weak")
        add_rule(5, "Submerge every tissuebox or pencil in the bathtub before placing it inside a drawer.", -1.95, True, "poison")
    else:
        raise ValueError(f"unknown agent scenario: {scenario}")

    return entries, pair_specs


class AgentSyntheticWorld:
    """Natural-language household-agent memory benchmark with realistic explicit/implicit conflicts, synergy, and poison."""

    def __init__(self, seed, scenario="agent-composite", size=24, poison_fraction=0.2):
        if scenario not in AGENT_SCENARIOS:
            raise ValueError(f"invalid agent synthetic scenario: {scenario}")
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        entries, pair_specs = _build_agent_catalog(scenario)
        # Interleave across the 6 families so streaming arrivals introduce new rules evenly across domains
        ordered_indices = []
        for slot in range(4):
            for fam in range(6):
                idx = fam * 4 + slot
                if idx < len(entries):
                    ordered_indices.append(idx)
        permutation = self.rng.permutation(len(ordered_indices))
        self.rules = []
        self.truth = {}
        raw_to_id = {}
        for pos, raw_idx in enumerate(ordered_indices):
            entry = entries[raw_idx]
            identifier = f"mem-{int(permutation[pos]):03d}"
            raw_to_id[raw_idx] = identifier
            self.rules.append({"id": identifier, "text": entry["text"]})
            self.truth[identifier] = {
                "topic": entry["topic"],
                "cross_topics": list(entry["cross_topics"]),
                "effect": entry["effect"],
                "poison": entry["poison"],
                "role": entry["role"],
                "synergy_member": entry["synergy_member"],
            }
        self.pairs = {}
        for idx_a, idx_b, weight in pair_specs:
            if idx_a in raw_to_id and idx_b in raw_to_id:
                self.pairs[tuple(sorted((raw_to_id[idx_a], raw_to_id[idx_b])))] = float(weight)
        self.topics = tuple(fam["name"] for fam in AGENT_TASK_FAMILIES)

    def tasks(self, episodes):
        contexts = self.rng.integers(0, len(AGENT_TASK_FAMILIES), size=episodes)
        variants = self.rng.integers(0, 3, size=episodes)
        uniforms = self.rng.random(episodes)
        return [(int(ctx), int(var), float(u)) for ctx, var, u in zip(contexts, variants, uniforms)]

    def query(self, context, variant=0):
        queries = AGENT_TASK_FAMILIES[context]["queries"]
        return queries[variant % len(queries)]

    def probability(self, context, selected):
        gates = {}
        for identifier in selected:
            info = self.truth[identifier]
            if info["topic"] == context:
                gates[identifier] = 1.0
            elif context in info.get("cross_topics", ()):
                gates[identifier] = 0.75
            else:
                # Out-of-domain clutter slightly distracts the agent (-0.12 token/attention clutter penalty)
                gates[identifier] = 0.12
        latent = AGENT_TASK_FAMILIES[context]["base_logit"]
        for identifier in selected:
            g = gates[identifier]
            eff = self.truth[identifier]["effect"]
            if g < 0.5 and eff > 0:
                latent -= 0.10
            else:
                latent += g * eff
        for first, second in combinations(selected, 2):
            latent += gates[first] * gates[second] * self.pairs.get(tuple(sorted((first, second))), 0.0)
        return sigmoid(latent)

    def export_truth(self):
        return {
            "scenario": self.scenario,
            "rules": self.truth,
            "pairs": [{"identifiers": list(pair), "effect": effect} for pair, effect in self.pairs.items()],
            "higher_order_triple": [],
        }


def build_world(seed, scenario="mixed", size=16, poison_fraction=0.2):
    if scenario in AGENT_SCENARIOS:
        return AgentSyntheticWorld(seed, scenario, size=size, poison_fraction=poison_fraction)
    return SyntheticWorld(seed, scenario, size=size, poison_fraction=poison_fraction)


def run_synthetic(
    output,
    methods,
    seeds,
    episodes=200,
    scenario="mixed",
    config=None,
    size=16,
    poison_fraction=0.2,
    arrival_every=0,
    embedding_model=None,
    token_encoding=None,
):
    if episodes < 1 or not seeds or not methods or len(set(seeds)) != len(seeds) or len(set(methods)) != len(methods) or arrival_every < 0:
        raise ValueError("positive episodes, unique methods/seeds and nonnegative arrival interval required")
    config = config or Config()
    encoder = LocalEncoder(embedding_model)
    counter = TokenCounter(token_encoding)
    for method in methods:
        MemoryManager(method, config, encoder=encoder, counter=counter)
    output = new_directory(output)
    write_json(
        output / "manifest.json",
        {
            "kind": "synthetic_experiment",
            "version": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "methods": methods,
            "seeds": seeds,
            "episodes": episodes,
            "scenario": scenario,
            "size": size,
            "poison_fraction": poison_fraction,
            "arrival_every": arrival_every,
            "embedding_model": encoder.label,
            "token_counter": counter.label,
            "config": asdict(config),
            "manager_llm_calls": 0,
            "warning": "Synthetic mechanism check, not benchmark or causal evidence.",
        },
    )
    summaries = []
    for seed in seeds:
        world = build_world(seed, scenario, size, poison_fraction)
        total_rules = len(world.rules)
        tasks = world.tasks(episodes)
        write_json(output / f"truth-seed-{seed}.json", world.export_truth())
        write_json(output / f"tasks-seed-{seed}.json", [[c, u] if scenario in LEGACY_SCENARIOS else [c, v, u] for c, v, u in tasks])
        initial_count = total_rules if not arrival_every else max(8, (2 * total_rules) // 3)
        for method in methods:
            manager = MemoryManager(method, config, seed + 10000, encoder=encoder, counter=counter)
            for rule in world.rules[:initial_count]:
                manager.add(rule["id"], rule["text"], prune=False)
            manager.prune()
            introduced = {rule["id"] for rule in world.rules[:initial_count]}
            events = []
            with (output / f"{method}-seed-{seed}.jsonl").open("w", encoding="utf-8") as handle:
                for step, (context, variant, uniform) in enumerate(tasks):
                    started = time.perf_counter()
                    selection = manager.choose(world.query(context, variant), world.topics[context])
                    prediction = manager.predict(selection)
                    elapsed = time.perf_counter() - started
                    probability = world.probability(context, selection.identifiers)
                    reward = int(uniform < probability)
                    manager.observe(selection, reward)
                    added = None
                    if arrival_every and (step + 1) % arrival_every == 0:
                        index = initial_count + (step + 1) // arrival_every - 1
                        if index < total_rules:
                            rule = world.rules[index]
                            manager.add(rule["id"], rule["text"], step + 1)
                            introduced.add(rule["id"])
                            added = rule["id"]
                    pruned = {entry["memory"]["identifier"] for entry in manager.archive}
                    good = {identifier for identifier in introduced if world.truth[identifier]["effect"] > 0}
                    poison = {identifier for identifier in introduced if world.truth[identifier]["poison"]}
                    synergy_set = {identifier for identifier in introduced if world.truth[identifier].get("synergy_member", False)}
                    event = {
                        "episode": step,
                        "context": world.topics[context],
                        "selected": selection.identifiers,
                        "reward": reward,
                        "predicted_probability": prediction,
                        "oracle_probability": probability,
                        "retrieval_tokens": selection.tokens,
                        "token_counter": manager.counter.label,
                        "bank_tokens": manager.bank_tokens,
                        "active_memories": len(manager.memories),
                        "explored": selection.explored,
                        "route_seconds": elapsed,
                        "introduced": added,
                        "pruned_good_fraction": len(pruned & good) / len(good) if good else None,
                        "pruned_poison_fraction": len(pruned & poison) / len(poison) if poison else None,
                        "pruned_synergy_fraction": len(pruned & synergy_set) / len(synergy_set) if synergy_set else None,
                    }
                    handle.write(json.dumps(event, allow_nan=False) + "\n")
                    events.append(event)
            snap = manager.snapshot()
            write_json(output / f"{method}-seed-{seed}-state.json", snap)
            active_conflicts = (
                len(snap.get("nli_conflicts", []))
                if method == "evomemguard"
                else sum(1 for p in snap.get("pairs", []) if p.get("conflict"))
            )
            summaries.append({
                "method": method,
                "seed": seed,
                "episodes": episodes,
                "success_rate": sum(event["reward"] for event in events) / episodes,
                "late_success_rate": sum(event["reward"] for event in events[episodes // 2 :]) / max(1, len(events[episodes // 2 :])),
                "mean_retrieval_tokens": sum(event["retrieval_tokens"] for event in events) / episodes,
                "mean_oracle_probability": sum(event["oracle_probability"] for event in events) / episodes,
                "brier_score": sum((event["predicted_probability"] - event["reward"]) ** 2 for event in events) / episodes,
                "route_ms_mean": 1000 * sum(event["route_seconds"] for event in events) / episodes,
                "pruned_good_fraction": events[-1]["pruned_good_fraction"],
                "pruned_poison_fraction": events[-1]["pruned_poison_fraction"],
                "pruned_synergy_fraction": events[-1]["pruned_synergy_fraction"],
                "active_conflicts": active_conflicts,
                "bank_tokens": manager.bank_tokens,
                "manager_llm_calls": 0,
            })
    aggregates = []
    for method in methods:
        rows = [row for row in summaries if row["method"] == method]
        rates = [row["success_rate"] for row in rows]
        late_rates = [row["late_success_rate"] for row in rows]
        aggregates.append({
            "method": method,
            "seeds": len(rows),
            "success_rate_mean": float(np.mean(rates)),
            "success_rate_seed_se": float(np.std(rates, ddof=1) / np.sqrt(len(rates))) if len(rates) > 1 else None,
            "late_success_rate_mean": float(np.mean(late_rates)),
            "mean_retrieval_tokens": float(np.mean([row["mean_retrieval_tokens"] for row in rows])),
            "mean_active_conflicts": float(np.mean([row["active_conflicts"] for row in rows])),
        })
    write_json(output / "summary.json", {"kind": "synthetic_experiment", "runs": summaries, "aggregate": aggregates})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    return aggregates
