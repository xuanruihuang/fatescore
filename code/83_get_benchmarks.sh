#!/bin/bash
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}   # repository root
# Standard RNA-velocity benchmark data sets, from the URLs scVelo's own
# scv.datasets functions use (scvelo 0.3.4): pancreatic endocrinogenesis, dentate
# gyrus (10X43_1), gastrulation erythroid, human bone marrow.
# GitHub is reached through the server's local proxy; figshare directly.
set -u
D=$ROOT/data/bench
mkdir -p "$D"
get() {  # url dest proxy
  if [ -s "$D/$2" ]; then echo "have $2"; return; fi
  curl -L --retry 3 --fail $3 -o "$D/$2.part" "$1" && mv "$D/$2.part" "$D/$2"
  rc=$?; echo "$2 rc=$rc size=$(stat -c %s "$D/$2" 2>/dev/null)"
}
get "https://github.com/theislab/scvelo_notebooks/raw/master/data/Pancreas/endocrinogenesis_day15.h5ad" pancreas.h5ad "$P"
get "https://github.com/theislab/scvelo_notebooks/raw/master/data/DentateGyrus/10X43_1.h5ad" dentategyrus.h5ad "$P"
get "https://ndownloader.figshare.com/files/27686871" gastrulation_erythroid.h5ad ""
get "https://ndownloader.figshare.com/files/27686835" bonemarrow.h5ad ""
md5sum "$D"/*.h5ad
