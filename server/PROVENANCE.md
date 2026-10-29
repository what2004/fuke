# Playground source provenance

`vendor/playground` is a source snapshot from the user's existing Aero project:

```text
aero-hand-open-main/sim_rl/mujoco_playground
Git revision: d59156e099a8785dd58fbce2e25a6ee1f7b5800c
Package metadata: playground 0.2.0
```

The source is copied as present on disk, including any existing local changes;
the revision alone does not describe those changes. Its Apache-2.0 license is
retained in `vendor/playground/LICENSE`. Git metadata and bytecode are excluded.
The training archive's `MANIFEST.sha256` identifies every included file.

This snapshot preserves the Aero project's Playground State/wrapper interfaces;
the finger environment and training entry remain in `najiushi/rl` and do not
modify Aero's task registry. GPU dependencies use CUDA 12 and MJX-JAX.
