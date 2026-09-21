# Resource 2 - Bibliographic Audit

**Article:** Explainable and Uncertainty-Aware Computational Geodetic Inversion: A Review and Framework for Multi-Source Earth Observation  
**Journal:** Computers & Geosciences  
**Authors:** Mohit Sheode and D. Kishan

## Purpose

This directory contains the derived bibliographic audit supporting manuscript Tables 1 and 2 and the narrative technical review. The files are reviewer-readable derived indexes and audit summaries; they do **not** redistribute the proprietary raw Scopus or Dimensions exports.

## Files

- `bibliographic_audit.xlsx` - cited-reference index, author-year thematic map, Table 1 support, summary counts, and package guide.
- `scopus_derived_index_151.csv` - derived Scopus index containing 151 records.
- `dimensions_derived_index_500.csv` - derived Dimensions index containing 500 records.
- `doi_overlap_95.csv` - 95 records matched across the two sources by normalized DOI.
- `combined_unique_index_556.csv` - approximate 556-record combined index after DOI/title deduplication.
- `table1_summary_counts.csv` - machine-readable values supporting manuscript Table 1.
- `thematic_map_author_year.csv` - author-year thematic map matching manuscript Table 2.

## Table 1 audit counts

- Scopus export records: 151
- Dimensions export records: 500
- DOI overlap: 95
- Combined unique pool: approximately 556
- Scopus records from 2020 onward: 58
- Dimensions records from 2020 onward: 189
- Raw recent-emphasis total: 247 (before deduplication; includes overlap)
- Deduplicated unique records from 2020 onward: 211
- Scopus records from 2024 onward: 24
- Dimensions records from 2024 onward: 79
- Deduplicated unique records from 2024 onward: 89

## Deduplication

Records were matched by normalized DOI when available. Records without a usable DOI were compared using normalized titles. The combined count is reported as approximate because title normalization can merge or separate borderline bibliographic variants.

## Citation style

The manuscript uses author-year citations. Any `Audit ID` in the workbook is only a spreadsheet navigation identifier and is not a manuscript citation number.

## Provenance and licence

The database exports used to derive these files were dated 25 June 2026. Original raw exports are retained privately by the authors for editorial verification and are not redistributed because database-specific fields may be subject to third-party terms. These derived audit files and documentation are released under CC BY 4.0.
