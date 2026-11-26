"""Ensembl gene IDs -> HGNC symbols via mygene.info, for the labelling data of 91.

dynamo converts Ensembl IDs to symbols with pyensembl, which needs an Ensembl
download; the same mapping is taken from mygene.info through the server's proxy.

Usage: 91b_gene_symbols.py <h5ad> <out.tsv>
"""
import json
import subprocess
import sys

import anndata as ad

a = ad.read_h5ad(sys.argv[1], backed="r")
ids = [i.split(".")[0] for i in a.var_names]
out = {}
for k in range(0, len(ids), 1000):
    q = ",".join(ids[k:k + 1000])
                        "-d", "q=" + q, "-d", "scopes=ensembl.gene", "-d", "fields=symbol", "-d", "species=human",
                        "https://mygene.info/v3/query"], capture_output=True, text=True)
    for h in json.loads(r.stdout):
        if "symbol" in h:
            out[h["query"]] = h["symbol"]
with open(sys.argv[2], "w") as f:
    f.write("gene_id\tsymbol\n")
    for i in ids:
        f.write("%s\t%s\n" % (i, out.get(i, "")))
print(len(ids), "ids", len(out), "symbols")
