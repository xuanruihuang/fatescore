# Data for the velocity / clonal-fate analysis

Not committed to the repository; kept on the compute server at
`<ROOT>/data/`.

| file | source | bytes | md5 | licence |
|---|---|---|---|---|
| `larry_invitro_adata_sub_raw.h5ad` | figshare article 20780344 (Qian Qin, Pyro-Velocity release, 2022-09-02), https://ndownloader.figshare.com/files/37028569 | 1525338690 | d618305c0ee0e41570b1147e63df332a | CC BY 4.0 |
| `celltag/morris_reprogramming.h5ad` | figshare article 13105481 (Marius Lange and Allon Klein, MEF to induced endoderm progenitors reprogramming, 2021-06-28), deposited as `2020-11-14_MK_morris_reprogramming.h5ad`, https://ndownloader.figshare.com/files/25503773 | 1594016831 | 10e7290862ae6a097c37e1e22eb496fe | CC BY 4.0 |
| `celltagmulti_clones/hsc_master_v2.csv` | the CellTag-multi authors' own clone table, `clone_tables/hsc.rna&atac.r1&2_master_v2.csv` in github.com/morris-lab/CellTag-multi-2023 | 4289676 | 259a3ac68b0b3bfce20f166243fdf54c | repository has no licence file |
| `celltagmulti_clones/sra_runinfo_PRJNA894147.csv` | NCBI run info for the CellTag-multi BioProject, https://trace.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?acc=PRJNA894147, retrieved 2026-11-06 | 37297 | 2c33a65e40d95613c5d139935229ad8e | public domain (NCBI) |

Original experiment: Weinreb et al. 2020, Science 367:eaaw3381 (LARRY), GEO
GSE140802. GEO distributes total counts only; the first file above carries the
spliced/unspliced split and the clone matrix used by the Pyro-Velocity and
TSvelo papers (49,302 cells).

The second experiment: Biddy et al. 2018, Nature 564:219--224 (CellTag), doi
10.1038/s41586-018-0744-4. The second file above is the CellRank group's
splicing-aware requantification, 104,679 cells by 22,630 genes, carrying
`reprogramming_day`, `timecourse`, `cell_type` and six columns of CellTag
calls. Both checksums were read from the figshare API's `computed_md5` field
and verified against the downloaded files.

The last two files are not analysis data. They support one sentence about what
a third experiment could and could not supply: CellTag-multi, Jindal et al.
2024, Nature Biotechnology 42:946--959, doi 10.1038/s41587-023-01931-4, GEO
GSE216521. The clone table is the authors' own and is recounted by
`code/25_celltagmulti_clones.py` under the definition this question needs,
because the paper's 2,227 state--fate clones pool two assays run on separate
aliquots of cells. The run-info file is the size and strategy of the raw
sequence data behind that series: the GEO supplementary archive carries 68
filtered count matrices and no alignments, while the 68 raw libraries (34
scRNA-seq, 34 scATAC-seq, 636 GB) are in SRA under BioProject PRJNA894147.
Neither file was downloaded into the analysis data directory; both are kept
beside it at `<ROOT>/celltagmulti_clones/`.
