# ImageNet-1k extension asset condition

Date: 2026-08-01
Scope: asset provenance and readiness only; this is not method-effect evidence.

## Fixed server asset

The extension source is read-only external data already present on `pro6000`:

```text
/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted
```

It is not copied into either Git repository. FieldScope runbooks must receive the
path explicitly through `FIELDSCOPE_IMAGENET1K_ROOT`; they must not modify the
CoFiTok dataset tree.

The derived ImageFolder was exported from:

```text
Hugging Face dataset: benjamin-paine/imagenet-1k-256x256
snapshot: 1bd0400450249a7fe90c0aece37d0d03e7ea956a
raw server path: /root/autodl-tmp/CoFiTok/datasets/imagenet_1k_256x256_hf
```

The exporter copied the existing 256x256 JPEG bytes without resize, crop,
normalization, filtering, or JPEG recoding. It is a derived HF-parquet source and
must not be described as byte-identical to the official ILSVRC tar archives.

## Recorded verification facts

- train: 1,281,167 RGB JPEGs in 1,000 WNID directories;
- official validation: 50,000 RGB JPEGs in 1,000 WNID directories;
- export manifest:
  `/root/autodl-tmp/CoFiTok/datasets/imagenet_256/metadata/image_manifest.jsonl`;
- manifest bytes: 405,484,553;
- manifest SHA-256:
  `9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0`;
- exporter summary:
  `/root/autodl-tmp/CoFiTok/datasets/imagenet_256/metadata/export_summary.json`.

These facts were read from the completed CoFiTok asset record and checked against
the live server directory before any real FieldScope signal or formal metric was
available. The formal extension run must recompute the manifest SHA-256, verify
the two source split counts and 1,000 class directories, and run FieldScope's
metadata-only train/internal-val/test identity audit on its own fixed revision.

The local `D:` official-archive source was disconnected when this condition was
recorded. The server asset is self-contained for the registered extension, so the
local drive is not required while the server path and hashes continue to pass.
