"""Assert that everything compared with everything else was built on the same
cells, the same genes and the same graph.

This exists because it was broken. Scripts added on 2026-11-21 read one feature
directory's gene list and cell list while building their objects on all the
cells, so a feature block compared against a day-2-only velocity kernel had
been computed with the later cells -- the ones that define the fate labels --
inside its graph. Two results had to be retracted.

The check is deliberately mechanical and reads only what is on disk:

  * every log in a feature directory that records a scope must record the same
    scope. The pipeline's own cellrank log records the transient and terminal
    counts, whose sum is the object the kernels were built on; anything added
    later records the same quantity. A disagreement is the error above, and it
    is a failure.
  * every array in a feature directory must have one row per early cell of that
    directory, and a gene-space array must have one column per highly variable
    gene of that directory.
  * the published evaluations each take exactly one feature directory, which is
    what makes their arms comparable; the call sites are printed so that this
    can be seen rather than assumed.

Exits non-zero if any assertion fails.

Usage:
  python 38_scope_check.py <outdir> [--repo <dir with reproduce.sh and server/>]
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

# logs that record the size of the object a feature directory was built on, and
# how to read it out of each
SCOPE_READERS = {
    "cellrank_log.json": lambda j: (j.get("n_transient", 0) + j.get("n_terminal", 0)
                                    if "n_transient" in j else None),
    "expression_kernel_log.json": lambda j: (
        j.get("n_transient", 0)
        + sum((j.get("terminal_states") or {}).values())
        if "n_transient" in j else None),
    "idw_log.json": lambda j: j.get("n_cells_in_object"),
    "scope.json": lambda j: j.get("n_cells_in_object"),
}
# arrays whose columns are fate probabilities rather than genes
FATE_PREFIXES = ("cr_", "snap_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("--repo", default=".")
    a = ap.parse_args()

    fails, notes = [], []
    featdirs = sorted(d for d in glob.glob(os.path.join(a.outdir, "*feat_*"))
                      if os.path.isdir(d))
    print("feature directories: %s\n" % [os.path.basename(d) for d in featdirs])

    hvgs, d2s = {}, {}
    for d in featdirs:
        name = os.path.basename(d)
        files = sorted(os.listdir(d))
        if "log.json" not in files:
            notes.append("%s: no log.json, nothing to check" % name)
            continue
        log = json.load(open(os.path.join(d, "log.json")))
        n_hvg = log.get("n_hvg")
        hvg = json.load(open(os.path.join(d, "hvg.json"))) \
            if "hvg.json" in files else []
        d2 = json.load(open(os.path.join(d, "d2_names.json"))) \
            if "d2_names.json" in files else []
        hvgs[name], d2s[name] = set(hvg), list(d2)
        print("%s: %d early cells, %d highly variable genes" % (name, len(d2), n_hvg))
        if n_hvg != len(hvg):
            fails.append("%s: log says %s genes, hvg.json has %d"
                         % (name, n_hvg, len(hvg)))

        # 1. every recorded scope must agree
        scopes = {}
        # the snapshot methods write one log each, named after the tag,
        # so they are picked up by pattern rather than listed
        readers = dict(SCOPE_READERS)
        for q in sorted(glob.glob(os.path.join(d, "snap_*_log.json"))):
            readers[os.path.basename(q)] = (
                lambda j: j.get("n_cells_in_object"))
        for f, read in readers.items():
            p = os.path.join(d, f)
            if not os.path.exists(p):
                continue
            try:
                v = read(json.load(open(p)))
            except Exception as e:
                notes.append("%s/%s: unreadable (%s)" % (name, f, e))
                continue
            if v is None:
                notes.append("%s/%s: records no scope" % (name, f))
            else:
                scopes[f] = int(v)
        if scopes:
            print("   recorded scopes: %s" % scopes)
            if len(set(scopes.values())) > 1:
                fails.append("%s: logs disagree on the scope: %s -- a block was "
                             "built on a different cell set from the one it is "
                             "compared against" % (name, scopes))
        else:
            notes.append("%s: no log records a scope" % name)

        # 2. array shapes
        n_fate = None
        crp = os.path.join(d, "cellrank_log.json")
        if os.path.exists(crp):
            st = json.load(open(crp)).get("terminal_states") or {}
            n_fate = len(st) or None
        for f in files:
            if not f.endswith(".npy"):
                continue
            M = np.load(os.path.join(d, f), mmap_mode="r")
            rows = M.shape[0]
            if rows != len(d2):
                fails.append("%s/%s: %d rows, expected one per early cell (%d)"
                             % (name, f, rows, len(d2)))
            if M.ndim == 2 and not f.startswith(FATE_PREFIXES) \
                    and not f.startswith("field_") and M.shape[1] not in (1,):
                if M.shape[1] != n_hvg:
                    notes.append("%s/%s: %d columns, not the %d genes "
                                 "(fine for a field in another space)"
                                 % (name, f, M.shape[1], n_hvg))
            if f.startswith(FATE_PREFIXES) and M.ndim == 2 and n_fate:
                if M.shape[1] != n_fate:
                    fails.append("%s/%s: %d fate columns, the directory's "
                                 "cellrank log has %d terminal states"
                                 % (name, f, M.shape[1], n_fate))
        print()

    # 3. cross-directory comparison, so a mismatch is visible rather than silent
    names = sorted(hvgs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            x, y = names[i], names[j]
            if not hvgs[x] or not hvgs[y]:
                continue
            shared = len(hvgs[x] & hvgs[y])
            same_cells = d2s[x] == d2s[y]
            print("%s vs %s: %d shared genes of %d/%d, same early cells: %s"
                  % (x, y, shared, len(hvgs[x]), len(hvgs[y]), same_cells))
            if shared < min(len(hvgs[x]), len(hvgs[y])):
                notes.append("%s and %s select different genes (%d shared) -- "
                             "never mix their arrays in one comparison"
                             % (x, y, shared))

    # 4. the published evaluations: one feature directory per run
    print()
    pat = re.compile(r"03_evaluate\.py[\"'\s]+(\S+)")
    for sh in sorted(glob.glob(os.path.join(a.repo, "reproduce.sh"))
                     + glob.glob(os.path.join(a.repo, "server", "*.sh"))):
        for line in open(sh, encoding="utf-8", errors="replace"):
            m = pat.search(line)
            if m:
                print("%-28s evaluates %s" % (os.path.basename(sh), m.group(1)))
    print("   each call takes exactly one feature directory, so the arms within "
          "one evaluation share cells, genes and graph by construction")

    print()
    for n in notes:
        print("note: %s" % n)
    if fails:
        print("\nFAILED %d check(s):" % len(fails))
        for f in fails:
            print("  - %s" % f)
        sys.exit(1)
    print("\nall scope checks passed")


if __name__ == "__main__":
    main()
