# Model Update and Rollback

Written design drill using the [serving contract](results.md#exercise-7-api-contract-design-review) and the [CUDA I/O Binding evidence](results.md#ort-cuda-execution-provider-with-io-binding).

## Prompt

Version A is serving batch-1 requests. Version B has the same input and output shapes but requires a different normalization configuration. Design an update while requests are still arriving. After switching to B, inject a regression where sampled outputs pass the quality checks but p99 latency and deadline misses exceed the agreed limits.

## Deliverable

1. Define an immutable release bundle containing the model, preprocessing configuration, and input/output contract. Say how you would validate dtype, layout, shape, label ordering, and numerical parity against the correct reference for B. Set output and latency acceptance thresholds before testing.
2. Describe how you load and warm B while A serves. Budget for both engines, working buffers, and warmup interference. Explain what happens if B cannot start without violating A's memory or latency budget.
3. Draw the switch and rollback timeline. Assign each request to one complete bundle at admission. For a temporal decoder, switch at a complete window boundary and define how state is reset or migrated. State when buffers and engine references can be released and how you keep A available for rollback.
4. Define the signal that triggers rollback. Include version-tagged p99 latency, deadline misses, and error rates. Describe what happens to B's in-flight requests and how you verify that new requests return to A.

## A strong answer

Show that preprocessing and model versions cannot mix, in-flight work stays valid until completion, and a failed warmup leaves A serving. End with a concrete rollback condition and a check that freshness and output quality recover after traffic returns to A.
