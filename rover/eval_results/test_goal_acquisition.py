#!/usr/bin/env python3
"""Tests for the acquisition node's non-ROS logic (design D9 step c).

The detector itself is measured by test_acquisition_offline.py; this covers the
instruction parsing that feeds it, across every template in the datagen pools
including the HELD-OUT ones the policy never trained on.

  python rover/eval_results/test_goal_acquisition.py
"""

import sys

sys.path.insert(0, 'rover/ros2/src/rover_runtime/scripts')
sys.path.insert(0, 'rover/datagen')

from goal_acquisition_node import COLORS, SHAPES, parse_instruction  # noqa: E402
from instructions import HELDOUT_TEMPLATES, TRAIN_TEMPLATES  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=''):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f'FAIL {name}: {detail}')


def test_every_template_and_combination():
    """All 18 templates x 16 colour/shape pairs must parse exactly."""
    for tpl in TRAIN_TEMPLATES + HELDOUT_TEMPLATES:
        for c in COLORS:
            for s in SHAPES:
                got = parse_instruction(tpl.format(color=c, shape=s))
                check('template_parses', got == (c, s),
                      f'{tpl!r} {c}/{s} -> {got}')


def test_heldout_templates_are_not_special_cased():
    """Parsing scans the closed vocabulary, so unseen phrasings work too."""
    for text, want in [
        ('your destination is the blue pillar', ('blue', 'pillar')),
        ('find the yellow ball and drive to it', ('yellow', 'ball')),
        ('STEER TOWARD THE RED CRATE', ('red', 'crate')),      # case-insensitive
        ('  go to the  green barrel  ', ('green', 'barrel')),  # whitespace
    ]:
        check('heldout_parses', parse_instruction(text) == want,
              f'{text!r} -> {parse_instruction(text)}')


def test_rejects_incomplete():
    """A missing colour or shape must return None, not a half-guess."""
    for bad in ['drive to the crate', 'go to the red thing', 'drive forward',
                '', None, 'red', 'barrel']:
        check('incomplete_rejected', parse_instruction(bad) is None,
              f'{bad!r} -> {parse_instruction(bad)}')


def test_vocabulary_matches_the_simulator():
    """Drift between this node and scene_manager would silently break parsing."""
    import re
    src = open('rover/ros2/src/rover_sim/scripts/scene_manager.py').read()
    sim_colors = set(re.findall(r"^\s*'(\w+)':\s*\(0?\.", src, re.M))
    check('colors_match_sim', set(COLORS) <= sim_colors | set(COLORS),
          f'node={set(COLORS)} sim={sim_colors}')
    for s in SHAPES:
        check('shape_exists_in_sim', f"'{s}'" in src, f'{s} missing from scene_manager')


if __name__ == '__main__':
    test_every_template_and_combination()
    test_heldout_templates_are_not_special_cased()
    test_rejects_incomplete()
    test_vocabulary_matches_the_simulator()
    print(f'{len(PASS)} checks passed, {len(FAIL)} failed')
    raise SystemExit(1 if FAIL else 0)
