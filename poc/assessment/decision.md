# Human decision

## Human decision — 2026-09-03T03:41:22.885002Z

decision: GO
decision_source: human
decided_at: 2026-09-03T03:41:22.885002Z

### Rationale

PoC4 provides reproducible evidence that temporal and CWT representations are complementary and that multimodal fusion consistently improves over the temporal-only baseline. The best fixed fusion achieved an average MCC of approximately 0.9585 compared with 0.9474 for the temporal expert, corresponding to an improvement of about +0.011 MCC. Reliability-aware adaptive fusion further increased the average MCC to approximately 0.9634–0.9636, with an average gain of about +0.005 MCC over fixed fusion.

The adaptive improvement is promising but not yet confirmatory because its paired confidence interval overlaps zero and performance is not positive for every seed. However, the oracle MCC of approximately 0.977 indicates substantial remaining complementarity between the temporal and CWT experts.

Therefore, the project receives a GO. The current evidence is sufficient to justify a confirmatory stage using fully independent multi-seed pipelines, the untouched Te2 split, and an additional external dataset. Even if the adaptive advantage is not confirmed, the demonstrated improvement from fixed temporal–CWT fusion remains a viable basis for a publication.
