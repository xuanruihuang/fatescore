"""How many state-fate clones the other arms of the LARRY deposit hold.

The analysis in the manuscript uses one arm of GSE140802, the in vitro time
course, and takes its splicing layer from a public requantification. The other
arms -- the in vivo transplant and the cytokine perturbation -- carry the same
barcoding and the same clone matrices, but no public splicing layer, so using
them means realigning the raw reads. Before spending that, count what they
would give: how many clones span an early state sample and a later fate sample,
how many early cells sit in such clones, and whether a clone's later cells fall
in more than one recipient mouse, which is what an independent second read-out
of the same clone would need.

Reads only the per-sample metadata table and clone matrix from GEO, both of
which are a few megabytes; it does not touch the count matrices. Standard
library only, so it runs under any python 3 without the analysis environment.

Usage:
  python 26_larry_arms_feasibility.py <meta.txt.gz> <clone_matrix.mtx.gz> \
      <out.csv> [--group-col "Recipient mouse"] [--early-timepoint 2]
"""
import argparse
import csv
import gzip
import os
import sys
from collections import Counter, defaultdict


def opener(path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.endswith(".gz") \
        else open(path, "rt", encoding="utf-8", newline="")


def read_meta(path):
    with opener(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def read_mtx_rows(path, n_cells):
    """Return {clone index: set of cell indices} for a coordinate Matrix Market
    file, plus its declared shape. The deposit writes the clone matrix with
    clones as rows and cells as columns for some arms and the other way round
    for others, so the orientation is read off the declared shape."""
    cells_of = defaultdict(set)
    shape = None
    flip = None
    with opener(path) as fh:
        for line in fh:
            if line.startswith("%"):
                continue
            parts = line.split()
            if not parts:
                continue
            if shape is None:
                shape = (int(parts[0]), int(parts[1]))
                if shape[1] == n_cells:
                    flip = False          # clones x cells
                elif shape[0] == n_cells:
                    flip = True           # cells x clones
                else:
                    sys.exit("clone matrix is %d x %d, neither side matches the "
                             "%d cells in the metadata" % (shape + (n_cells,)))
                continue
            i, j = int(parts[0]) - 1, int(parts[1]) - 1
            v = float(parts[2]) if len(parts) > 2 else 1.0
            if v != 0.0:
                cells_of[i if not flip else j].add(j if not flip else i)
    return cells_of, shape, flip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("meta")
    ap.add_argument("clones")
    ap.add_argument("out")
    ap.add_argument("--group-col", default="Recipient mouse",
                    help="the column naming the independent compartment a late "
                         "cell was collected from; a clone whose late cells "
                         "span two of them has a second read-out")
    ap.add_argument("--early-timepoint", default="2",
                    help="the value of the time-point column marking the state "
                         "sample")
    ap.add_argument("--time-col", default="Time point")
    ap.add_argument("--type-col", default="Cell type annotation")
    ap.add_argument("--mature-types", default="",
                    help="comma-separated annotations that count as a called "
                         "fate, in the fixed order used to break ties; empty "
                         "skips the counts under the published rule")
    ap.add_argument("--min-siblings", type=int, default=3,
                    help="the published rule's threshold on how many mature "
                         "late siblings a clone needs before its fate is called")
    a = ap.parse_args(argv)

    meta = read_meta(a.meta)
    n_cells = len(meta)
    cells_of, shape, flip = read_mtx_rows(a.clones, n_cells)
    n_clones = shape[1] if flip else shape[0]

    time = [r[a.time_col] for r in meta]
    ctype = [r.get(a.type_col, "NA") for r in meta]
    has_group = a.group_col in meta[0]
    group = [r[a.group_col] for r in meta] if has_group else ["NA"] * n_cells
    early = [t == a.early_timepoint for t in time]

    rows = []

    def add(k, v):
        rows.append((k, v))

    add("cells in the deposit", n_cells)
    add("clone matrix shape", "%d x %d (%s)"
        % (shape + ("cells x clones" if flip else "clones x cells",)))
    add("clones declared in the clone matrix", n_clones)
    add("clones with at least one cell", len(cells_of))
    add("cells with a clone",
        len(set().union(*cells_of.values())) if cells_of else 0)
    for t, c in sorted(Counter(time).items(), key=lambda x: x[0]):
        add("cells at time point %s" % t, c)
    add("early cells (time point %s)" % a.early_timepoint, sum(early))
    if has_group:
        add("distinct %s among late cells" % a.group_col,
            len(set(g for i, g in enumerate(group)
                    if not early[i] and g not in ("NA", ""))))

    n_sf = 0
    early_in_clone = 0
    early_in_sf = 0
    two_group = 0
    groups_per_clone = Counter()
    fate_types = Counter()
    for j, cells in cells_of.items():
        e = [i for i in cells if early[i]]
        l = [i for i in cells if not early[i]]
        early_in_clone += len(e)
        if e and l:
            n_sf += 1
            early_in_sf += len(e)
            fate_types.update(ctype[i] for i in l)
            g = set(group[i] for i in l if group[i] not in ("NA", ""))
            groups_per_clone[len(g)] += 1
            if len(g) >= 2:
                two_group += 1

    add("early cells in any clone", early_in_clone)
    add("clones spanning an early and a late sample", n_sf)
    add("early cells in those clones", early_in_sf)
    add("of those clones, late cells in >=2 %s" % a.group_col, two_group)
    for k in sorted(groups_per_clone):
        add("clones whose late cells span %d %s" % (k, a.group_col),
            groups_per_clone[k])
    for t, c in fate_types.most_common(16):
        add("late cells annotated %s (in state-fate clones)" % t, c)

    # The same arm under the rule the manuscript actually uses: a clone's fate
    # is called only from at least --min-siblings mature late cells, and the
    # comparison against a sister oracle needs that many in each of two
    # compartments separately.
    mature = [t for t in a.mature_types.split(",") if t]
    if mature:
        rank = {t: i for i, t in enumerate(mature)}
        called = 0
        early_called = 0
        by_fate = Counter()
        two_comp = 0
        two_comp_each = 0
        for j, cells in cells_of.items():
            e = [i for i in cells if early[i]]
            if not e:
                continue
            lm = [i for i in cells if not early[i] and ctype[i] in rank]
            if len(lm) < a.min_siblings:
                continue
            called += 1
            early_called += len(e)
            cnt = Counter(ctype[i] for i in lm)
            top = max(cnt.values())
            by_fate[min((t for t in cnt if cnt[t] == top), key=rank.get)] += 1
            per_comp = Counter(group[i] for i in lm if group[i] not in ("NA", ""))
            if len(per_comp) >= 2:
                two_comp += 1
                if sum(1 for v in per_comp.values() if v >= a.min_siblings) >= 2:
                    two_comp_each += 1
        add("clones with >=%d mature late siblings and an early cell"
            % a.min_siblings, called)
        add("early cells in those clones", early_called)
        add("of those, late mature cells in >=2 %s" % a.group_col, two_comp)
        add("of those, >=%d mature late siblings in each of 2 %s"
            % (a.min_siblings, a.group_col), two_comp_each)
        for t, c in by_fate.most_common():
            add("clones whose called fate is %s" % t, c)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["quantity", "value"])
        for k, v in rows:
            w.writerow([k, v])
    for k, v in rows:
        print("%-64s %s" % (k, v))


if __name__ == "__main__":
    main()
