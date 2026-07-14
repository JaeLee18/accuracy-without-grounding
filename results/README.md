# Bundled results

This directory contains the prediction records and aggregate tables used by the paper. The analysis
scripts operate on these files, so most paper statistics can be reproduced without downloading
videos or model weights.

## Layout

```text
results/
|-- full_study/              # Video-MME 600-question study
|   |-- qwen2vl_results.json
|   |-- llava_results.json
|   |-- internvl2_results.json
|   |-- *_black_results.json
|   |-- *_shuffled_results.json
|   |-- *_singleframe_results.json
|   `-- optical_flow.json
|-- mvbench/                 # three-model MVBench predictions
|-- egoschema/               # three-model EgoSchema predictions and summary
|-- experiments/             # FPS, scale, and API-model aggregate results
|-- pilot_results.json
`-- CLUSTER_RESULTS.md
```

## Common record fields

Files were produced by several model-specific runners, so field names are not perfectly uniform.
The common information is:

```json
{
  "question_id": "...",
  "video_id": "...",
  "task_type": "Temporal Reasoning",
  "condition": "original",
  "prediction": "C",
  "answer": "C",
  "correct": true
}
```

Some records use `videoID`, `ground_truth`, or `is_correct`, and some retain a raw generated
response. The corresponding analysis scripts normalize the fields they consume. Null predictions
are scored as incorrect for aggregate VDG; matched McNemar comparisons require valid predictions
from both models and therefore use their reported matched sample.

## Metric naming

The final paper calls the metric Visual Dependency Gap (VDG):

```text
VDG = accuracy(original) - accuracy(black)
```

Some early scripts, filenames, and notes use Visual Grounding Gap (VGG). VDG and VGG denote the
same calculation in this repository.
