# PLAsTiCC course data

These files are converted copies of *Unblinded Data for PLAsTiCC
Classification Challenge*, created by the PLAsTiCC Team and PLAsTiCC
modelers and published at <https://doi.org/10.5281/zenodo.2539456>.
The accompanying paper is PLAsTiCC Team et al., arXiv:1810.00001.

The source data are licensed under the [Creative Commons Attribution 4.0
International license](https://creativecommons.org/licenses/by/4.0/).
The `tiny300` files are an instructor-created teaching subset containing all
observations for a deterministic, stratified sample of 300 training objects.

The repository's MIT license applies only to original code and teaching
materials. It does not replace or modify the PLAsTiCC data license.

The four committed Parquet files are flat tables. Metadata has one row per
astronomical object; observations has many time-series measurements per
object. `manifest.json` records their provenance, schemas, counts, hashes, and
the exact subset and anchor-selection rules.

