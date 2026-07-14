# Data preparation

This repository includes the small question sample used by the paper, but it does not redistribute
benchmark videos or model weights.

## Included file

`videomme/full_sample.json` contains the 600-question Video-MME subset used for the main study:
six task types with 100 questions per type. Records contain the question identifier, source video,
task type, question, answer options, and answer.

For the default full-study layout, copy or link this file to:

```text
data/videomme_full/full_sample.json
```

## Default layout

Scripts use `VDG_DATA_ROOT=data` unless the environment variable is overridden.

```text
data/
|-- raw/Video-MME/
|   |-- videomme/test-00000-of-00001.parquet
|   `-- videos_chunked_*.zip
|-- videomme/                  # optional 60-question pilot
|-- videomme_full/
|   |-- full_sample.json
|   |-- videos/
|   |-- crf18/ crf23/ crf28/ crf33/ crf38/
|   `-- ablation/
|-- mvbench/
|   |-- raw/
|   |-- videos/
|   `-- mvbench_available_sample.json
`-- egoschema/
    |-- egoschema_subset.json
    |-- videos/
    `-- black/
```

Large data directories and archive files are ignored by Git.

## Official sources

- Video-MME: https://huggingface.co/datasets/lmms-lab/Video-MME
- MVBench: https://huggingface.co/datasets/OpenGVLab/MVBench
- EgoSchema: https://huggingface.co/datasets/lmms-lab/egoschema

Download each benchmark from its official source and comply with its license and terms. This
repository's MIT License applies to the repository code, not to third-party benchmark data.
