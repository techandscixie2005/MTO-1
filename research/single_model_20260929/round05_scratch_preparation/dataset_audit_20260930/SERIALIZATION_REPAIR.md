# Narrow serialization repair before partition materialization

The first target-blind corpus audit completed successfully. The first materialization attempt then stopped at the aggregate-equality gate, before the private output directory or any partition array was created. `ops/materialize_attempt/FAILED.json` and its log remain intact.

Cause: the rebuilt in-memory component-size histogram had integer dictionary keys, while JSON loading returned string keys. Python dictionary equality therefore rejected equivalent scientific content. There was no identity, grouping, count, threshold or assignment discrepancy indicated by the failure.

Repair: canonical serialization first converts keys to JSON-compatible types and then sorts them consistently, including bins 1, 2, 10 and 14. The materialization gate compares the actual approved canonical receipt bytes to the canonical rebuilt result. A synthetic regression verifies both round-trip equality and rejection of a changed count. All previous grouping tests still pass: 22 checks in total.

Old text sources and review/audit records are copied and hash-verified in `repair01_original/`; its snapshot manifest SHA256 is `a644dcc26a963ebf1eeea5661f4b942a610dc8e4da9e734756388bc3e74a2f00`. The original complete audit and failed materialization operational directories retain their exact recorded paths and receipts. Current canonical receipts were moved into the snapshot before the renewed preparation, rather than silently overwritten.

Old source manifest: `1315f7fcb6112c893ad0855b2144af7cf4711baadd7e6438ae5f96075d8dd042`. Old audit: `0419a5a856e084ad4e75cc67113d3dd290adf7826d235cedc9d2cf41df78dc80`.

Repaired source manifest: `2bcd4d256179ad99f6b88b34f50bba1a99facc9389a6438535662410c6783a38`. Repaired synthetic receipt: `a8668972b37dff51ebb0d293f04d888e5b151b506c8e89e836670ad8ccd7d851`.

Only `build_partition.py` canonical serialization/comparison and the synthetic regression changed inside the scientific closure. The grouping core, settings and scientific protocol are unchanged. The operational wrapper accepts an explicit safe attempt suffix so renewed attempts keep all previous process identities/logs; it never reuses an attempt or automatically retries.

Root explicitly approved this narrow repair and renewed independent review. A new pre-corpus review, owned `repair01` audit, exact aggregate comparison with the old scientific contents, and a newly bound materialization review must precede a new materialization attempt. No model fitting, test scoring, target decoding or assignment adjustment is involved.
