# V10 publication package

## Main manuscript
- `Main_Figures`: numbered PNG/PDF/SVG files selected for the revised paper.
- `Main_Figure_Data`: one CSV per selected figure, including the V8 core and
  the Nordic/reviewer extensions.
- `Main_Tables`: at most seven numbered tables.
- `main_figures_manifest.csv` and `main_tables_manifest.csv`: source-to-output audit trail.

## Appendix
All non-selected figures, tables, and raw figure CSVs are retained under
`04_Appendix`; nothing is deleted from the numerical study folders.

## Rebuilding figures without recomputation
Run:
`python geo_transition_network_v10.py --figures-only --package-only`

The reviewer figures are recreated from `Figure_Data`. Core V8 figure datasets
are preserved in `00_Manuscript/Main_Figure_Data` and can be used by any
standalone plotting script without rerunning the Monte Carlo experiments.
