# SPDX-License-Identifier: Apache-2.0
"""Keeping left and right apart inside one pair of AirPods.

Motion data arrives labelled "side 0" / "side 1" (a bit in the RemoteHID frame). What that bit
means physically is not documented and has not been measured on enough hardware:

  fixed     the bit names a physical bud: nothing ever swaps
  role      the bit names the bud's role (0 = primary, 1 = secondary): the labels swap whenever the
            other bud becomes primary, e.g. depending on which one left the case first
  location  the buds are told apart by the LocationID they announce their sensors with

So nothing is assumed. At pairing the user shows which bud is on the left leg, and what the AirPods
said at that moment is kept (which bud was primary: left or right, from AACP packet 0x08; the
LocationID per side). Later every hypothesis says whether the labels are now swapped compared to
pairing. While they all agree the answer is used silently. When they disagree the labels are left
as they were and the user is asked to identify the legs once more; that observation rules out the
hypotheses that were wrong, for this pair, for good.
"""
from __future__ import annotations

HYPOTHESES = ("fixed", "role", "location")


class BudSides:
    def __init__(self, base: dict | None = None):
        base = base if isinstance(base, dict) else {}
        self.has_base = bool(base)
        self.base_primary = base.get("primary") if base.get("primary") in ("left", "right") else None
        self.base_loc = _loc(base.get("loc"))
        self.ruled_out = {h for h in (base.get("ruled_out") or []) if h in HYPOTHESES}
        self.primary = None           # "left" | "right": which physical bud is primary now
        self.primary_strong = False   # from packet 0x08 (else guessed from the battery packet's order)
        self.loc = {}                 # side -> LocationID announced now

    # ------------------------------------------------------------------ what the AirPods say
    def observe_primary(self, name, strong=True):
        """-> True when it changed anything."""
        if name not in ("left", "right") or (self.primary_strong and not strong):
            return False
        changed = name != self.primary
        self.primary, self.primary_strong = name, self.primary_strong or strong
        return changed

    def observe_location(self, side, loc):
        if self.loc.get(side) == loc:
            return False
        self.loc[side] = loc
        return True

    def evidence(self):
        return {"primary": self.primary, "loc": {str(k): v for k, v in sorted(self.loc.items())}}

    # ------------------------------------------------------------------ are the labels swapped?
    def predictions(self):
        """{hypothesis: swapped since pairing?} for every hypothesis that can tell right now."""
        out = {"fixed": False}
        if self.base_primary and self.primary:
            out["role"] = self.primary != self.base_primary
        b, n = self.base_loc, self.loc
        if len(b) == 2 and b[0] != b[1] and 0 in n and 1 in n:
            if (n[0], n[1]) == (b[0], b[1]):
                out["location"] = False
            elif (n[0], n[1]) == (b[1], b[0]):
                out["location"] = True
        return {h: v for h, v in out.items() if h not in self.ruled_out}

    def resolve(self):
        """(swap, certain): swap the side labels to get back the ones of pairing time; certain is
        False when the remaining hypotheses disagree (then swap is False: labels left as they are)."""
        if not self.has_base:
            return False, True
        answers = set(self.predictions().values())
        if len(answers) == 1:
            return answers.pop(), True
        return False, False

    # ------------------------------------------------------------------ learning from a new pairing
    def learn(self, observed_swap: bool):
        """The user identified the legs again and the labels were (not) swapped compared to the last
        pairing: drop the hypotheses that said otherwise."""
        wrong = {h for h, v in self.predictions().items() if v != bool(observed_swap)}
        if len(wrong) < len(self.predictions()):       # never rule out everything on one observation
            self.ruled_out |= wrong
        return wrong

    def to_config(self):
        """What to keep: what the AirPods say now becomes the reference of the new pairing."""
        return {"primary": self.primary, "loc": {str(k): v for k, v in sorted(self.loc.items())},
                "ruled_out": sorted(self.ruled_out)}


def _loc(d):
    out = {}
    for k, v in (d or {}).items():
        try:
            out[int(k)] = int(v)
        except (TypeError, ValueError):
            pass
    return out


def split(source):
    """'AA:BB#1' -> ('AA:BB', 1)"""
    addr, _, side = str(source).rpartition("#")
    return addr, int(side) if side.isdigit() else 0


def partner(source, sources):
    """The other bud of the same pair, if it is streaming too."""
    addr, side = split(source)
    other = f"{addr}#{1 - side}"
    return other if other in sources else None


def left_side(roles, addr):
    """Side label of the bud of `addr` that is on a left leg (None when the pair is not worn as a
    left/right couple, e.g. one bud on the waist)."""
    sides = {r.split("_")[0]: split(s)[1] for r, s in roles.items()
             if s and split(s)[0] == addr and r.split("_")[0] in ("left", "right")}
    return sides.get("left") if len(sides) == 2 else None


def update_buds(old_buds, old_roles, new_roles, evidence, forget=False):
    """Config 'buds' after a (re-)identification. evidence: {addr: BudSides.evidence()} as seen while
    identifying (labels as received, not corrected)."""
    out = dict(old_buds or {})
    for addr, ev in (evidence or {}).items():
        b = BudSides(None if forget else (old_buds or {}).get(addr))
        b.observe_primary(ev.get("primary"))
        for side, loc in _loc(ev.get("loc")).items():
            b.observe_location(side, loc)
        was, now = left_side(old_roles or {}, addr), left_side(new_roles, addr)
        if b.has_base and was is not None and now is not None:
            b.learn(now != was)
        if now is None:
            out.pop(addr, None)          # not worn as a left/right couple: nothing to keep apart
        else:
            out[addr] = b.to_config()
    return out
