# Example output

`v1_pipeline_example.html` is a **pre-executed run** of
`notebooks/clustering/v1_harmonization_pipeline.ipynb` on the bundled All of Us + CLSA + NIH CDE
example dictionaries — so you can see the expected shape of the pipeline's output (clusters,
value sub-clusters, CDE anchors, adopt/refine/novel verdicts, EITL queue) without running it.

It's a **representative** run (ddharmon v1.0.0, generated 2026-06-02). Clustering is seeded
(`random_state=42`), but your exact cluster IDs and the adopt/refine/novel verdicts may differ
with library versions, hardware, and Claude model updates — treat the numbers as illustrative.
