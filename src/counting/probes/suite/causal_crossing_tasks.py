"""Deterministic non-counting histories, used only if Stage 1 passes.

Swaps provide non-commuting state updates while preserving four objects. A
matched self-swap donor omits one update with a one-character argument change.
Each pair has identical operation histograms but different operation order.
No model response participates in construction or filtering.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

LOCATIONS = tuple("ABCD")
OBJECTS = ("red", "blue", "green", "yellow")


def execute(initial, operations):
    state = dict(initial)
    for a, b in operations:
        state[a], state[b] = state[b], state[a]
    return state


def omitted_state(initial, operations, j):
    return execute(initial, operations[:j] + operations[j + 1:])


def decision(state, obj, goal, plans):
    valid = [label for label, ops in plans.items() if execute(state, ops)[goal] == obj]
    if len(valid) != 1:
        raise ValueError("decision does not have exactly one successful plan")
    return int(valid[0])


def generate_tasks(seed=7, pairs=12):
    rng = np.random.default_rng(seed)
    tasks = []
    for pair in range(pairs):
        length = (6, 8, 10)[pair % 3]
        initial = dict(zip(LOCATIONS, map(str, rng.permutation(OBJECTS))))
        while True:
            operations = [tuple(map(str, rng.choice(LOCATIONS, 2, replace=False))) for _ in range(length)]
            permutation = [operations[int(i)] for i in rng.permutation(length)]
            if execute(initial, operations) != execute(initial, permutation):
                break
        assert Counter(operations) == Counter(permutation)
        for ordering, ops in enumerate((operations, permutation)):
            j = max(0, min(length - 2, int(round(length * (.25, .5, .75)[pair % 3])) - 1))
            true = execute(initial, ops)
            counterfactual = omitted_state(initial, ops, j)
            affected = [obj for obj in OBJECTS if next(k for k, v in true.items() if v == obj)
                        != next(k for k, v in counterfactual.items() if v == obj)]
            unaffected = [obj for obj in OBJECTS if obj not in affected]
            assert len(affected) == 2 and len(unaffected) == 2
            branches = {}
            for name, obj in (("different", affected[0]), ("same", unaffected[0])):
                correct_loc = next(k for k, v in true.items() if v == obj)
                counter_loc = next(k for k, v in counterfactual.items() if v == obj)
                goal = str(rng.choice([loc for loc in LOCATIONS if loc not in (correct_loc, counter_loc)]))
                other = counter_loc if name == "different" else str(rng.choice([loc for loc in LOCATIONS if loc not in (correct_loc, goal)]))
                labels = list(map(int, rng.choice(np.arange(10, 90), 2, replace=False)))
                plans = {str(labels[0]): [(correct_loc, goal)], str(labels[1]): [(other, goal)]}
                if rng.random() < .5:
                    plans = dict(reversed(list(plans.items())))
                branches[name] = dict(object=obj, goal=goal, plans=plans,
                                      true_answer=decision(true, obj, goal, plans),
                                      omitted_answer=decision(counterfactual, obj, goal, plans))
            assert branches["different"]["true_answer"] != branches["different"]["omitted_answer"]
            assert branches["same"]["true_answer"] == branches["same"]["omitted_answer"]
            labels = list(map(int, rng.choice(np.arange(10, 90), 2, replace=False)))
            branches["factual"] = dict(options={str(labels[0]): ops[j][1], str(labels[1]): ops[j][0]},
                                       true_answer=labels[0], omitted_answer=labels[1])
            if rng.random() < .5:
                branches["factual"]["options"] = dict(reversed(list(branches["factual"]["options"].items())))
            donor = ops.copy()
            donor[j] = (ops[j][0], ops[j][0])
            assert execute(initial, donor) == counterfactual
            tasks.append(dict(id=f"pair{pair}-order{ordering}", pair=pair, ordering=ordering,
                              initial=initial, operations=ops, donor_operations=donor, j=j,
                              true_state=true, omitted_state=counterfactual, branches=branches))
    return tasks


def branch_text(task, branch):
    if branch == "factual":
        spec = task["branches"][branch]
        choices = "\n".join(f"{label}: {loc}" for label, loc in spec["options"].items())
        return (f"\n\nWhich second location was named in operation {task['j'] + 1}?\n"
                f"{choices}\nReturn only the integer label of that location.")
    spec = task["branches"]["different" if branch == "supplied" else branch]
    supplied = ""
    if branch == "supplied":
        state = ", ".join(f"{loc}: {obj}" for loc, obj in task["true_state"].items())
        supplied = f"\n\nUse this supplied current state for the decision: {state}."
    choices = "\n".join(f"{label}: " + "; ".join(f"swap {a} and {b}" for a, b in ops)
                         for label, ops in spec["plans"].items())
    return (supplied + f"\n\nGoal: put the {spec['object']} object at location {spec['goal']}. "
            "Which single plan achieves the goal from the current state?\n" + choices
            + "\nReturn only the integer label of the successful plan.")
