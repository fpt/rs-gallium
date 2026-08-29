"""Compare two `gallium::layers` traces stage by stage.

The question it answers is the one `docs/CANDLE_BACKEND.md` §6b and §6c both end
on: two devices produce prefill logits ~0.2-0.3 apart, and the final logits
cannot say whether one operator did it or twenty-four layers of rounding did.
Those two look identical at the head and completely different one stage at a
time.

    GALLIUM_DEVICE=cpu   RUST_LOG=gallium::layers=trace gallium ... 2> cpu.log
    GALLIUM_DEVICE=metal RUST_LOG=gallium::layers=trace gallium ... 2> metal.log
    uv run python scripts/layer_diff.py cpu.log metal.log

The first log is the reference. Only the first forward pass in each file is
compared: a prefill is the shape where the two disagree, and every decode step
after it starts from a KV cache the two no longer share, so comparing those
would measure the drift already accumulated rather than where it came from.

Reading the output:

  step      a jump at one stage is one operator, and the stage names it
  ramp      a ratio that climbs smoothly is compounding, and no operator is
            singly at fault
  stage 0   the embedding lookup, before any arithmetic -- a difference here is
            the GGUF read and nothing below it means anything
"""

import re
import sys

STAGE = re.compile(
    r"stage=(\d+)\s+rms=(\S+)\s+absmax=(\S+)\s+mean=(\S+)\s+last=\[([^\]]*)\]"
)


def load(path):
    """The first forward pass in a log, as {stage: (rms, absmax, mean, [last])}.

    A stage number that repeats has started the next forward pass, so reading
    stops there rather than letting later passes overwrite the first.
    """
    out = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = STAGE.search(line)
            if not m:
                continue
            stage = int(m.group(1))
            if stage in out:
                break
            out[stage] = (
                float(m.group(2)),
                float(m.group(3)),
                float(m.group(4)),
                [float(v) for v in m.group(5).split(",") if v],
            )
    return out


def rel(a, b):
    """Relative difference of two scalars that describe the same quantity."""
    scale = max(abs(a), abs(b), 1e-30)
    return abs(a - b) / scale


def channel_dev(ref_row, other_row, scale):
    """Largest sampled-channel disagreement, as a fraction of the signal.

    Deliberately *not* per-element relative error. A hidden state has channels
    near zero, and dividing a difference by one of those produces numbers above
    1.0 that say nothing about the state as a whole -- they dominate any
    aggregate they are mixed into, and the verdict then tracks whichever near-
    zero channel the fixed sample happened to land on. Normalising by the row's
    own magnitude keeps the number bounded and comparable across stages.
    """
    scale = max(abs(scale), 1e-30)
    return max((abs(a - b) for a, b in zip(ref_row, other_row)), default=0.0) / scale


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    ref_path, other_path = argv[1], argv[2]
    ref, other = load(ref_path), load(other_path)
    if not ref or not other:
        print(f"no `stage=` lines in {ref_path if not ref else other_path} -- "
              "was RUST_LOG=gallium::layers=trace set, and stderr captured?")
        return 1

    shared = sorted(set(ref) & set(other))
    missing = sorted(set(ref) ^ set(other))
    if missing:
        print(f"note: stages in only one log: {missing}")

    print(f"reference: {ref_path}")
    print(f"compared : {other_path}\n")
    print(f"{'stage':>5}  {'rms rel':>10}  {'absmax rel':>10}  "
          f"{'chan/signal':>12}  {'growth':>8}")

    # Below this, two floats are the same number as far as this comparison is
    # concerned, and a ratio against it is noise amplified into a verdict.
    floor = 1e-9

    prev = None
    entered = None
    worst_step = (0.0, None)
    for stage in shared:
        r, o = ref[stage], other[stage]
        rms, absmax = rel(r[0], o[0]), rel(r[1], o[1])
        chan = channel_dev(r[3], o[3], r[1])
        # The pointwise number leads: two tensors can carry the same rms and
        # the same absmax and still differ everywhere, so an aggregate alone
        # cannot see a divergence -- while the sample cannot miss one that
        # reaches the channels it watches. The aggregates stay as the backstop
        # for a difference spread too thin for 16 channels to catch.
        here = max(chan, rms)
        if entered is None and here > floor:
            entered = stage
        growth = ""
        if prev is not None and prev > floor:
            ratio = here / prev
            growth = f"{ratio:6.2f}x"
            if ratio > worst_step[0]:
                worst_step = (ratio, stage)
        print(f"{stage:>5}  {rms:>10.2e}  {absmax:>10.2e}  {chan:>12.2e}  {growth:>8}")
        prev = here

    last = shared[-1]
    end = max(channel_dev(ref[last][3], other[last][3], ref[last][1]),
              rel(ref[last][0], other[last][0]))
    print()

    if entered is None:
        print(f"the two logs agree at every stage to better than {floor:.0e}.")
        return 0
    if entered == shared[0]:
        print(f"stage {entered} already differs -- that is the first stage recorded, "
              "so this compares two different inputs and nothing below it is about "
              "arithmetic.")
        return 0

    print(f"the difference enters at stage {entered}, i.e. in block {entered - 1}.")
    ratio, at = worst_step
    if at is None:
        print("only one stage differs: it is that operator alone.")
    elif ratio > 4.0:
        print(f"largest single-stage jump: {ratio:.1f}x at stage {at}. A step that "
              f"size is an operator, and it is in block {at - 1}.")
    else:
        print(f"no stage after it multiplies the difference by more than {ratio:.1f}x: "
              "this is compounding through the residual stream, not one operator.")
    print(f"end to end: reaches {end:.2e} by stage {last}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
