# ML Building Change Detection + Georeferenced / Non-Georeferenced Upload — Design

Date: 2026-09-30 · Status: approved by user (research session `changedetection-20`), implementation handed to the refactor session.

## Problem (evidence gathered 2026-09-30)

1. **Discover (live Wayback) detections are wrong.** `istanbul_basaksehir` 2014→2026: 39 detections, 29 "demolished", every one of them a building *shadow* (different sun angle per year). The real change (construction site → blocks) was barely flagged; "existing" was always 0. Same failure on the user's own imagery (`sampla_data/2020.jpg` → `2026.jpg`, 8192×4468, Başakşehir). Root cause: `BuildingChangeDetector` is pixel differencing with a fixed 0.45 threshold; it has no radiometric normalisation, no shadow handling and no notion of "building". Parameter tuning cannot fix this.
2. **User upload fails.** 2020.jpg + 2026.jpg = 37.4 MB > `MAX_CONTENT_LENGTH` 32 MB → Flask 413 HTML → frontend `res.json()` throws → generic "hata oluştu" toast. Even if accepted, 4 full-size PNG base64 overlays of an 8192×4468 image would be sent in one JSON.
3. **Non-georeferenced images get fake coordinates** (Istanbul centre, GSD 0.5) — GeoJSON/CSV exports are fiction.
4. **Overlay colour bug:** `overlay_rgba` (RGBA) goes to `cv2.imencode` which reads BGRA → "demolished" is drawn blue, "new" yellow-green.
5. Benchmarks blend the ground-truth label into the diff map at 80% — the demo "detections" are mostly the label. No precision/recall is ever computed.

## Decisions (user-approved)

| Topic | Decision |
|---|---|
| Detection method | Pretrained deep model, not tuned classical CV. |
| Model | **ChangeStar building segmentation (ViT-B), ONNX quantized** — HF `geobase/changestar-building-segmentation-vitb`, file `onnx/model_quantized.onnx` (141 MB), pinned revision `8a6d7676d2fc9ea787b8786f2f05a65989c2606c`. Input `image` (1,3,1024,1024) ImageNet-normalised; output `building_prob` (1,1,1024,1024). **Tile size must be 1024** (fixed positional embeddings). |
| Change logic | Segment buildings in T1 and T2 separately, then match **objects** (connected components), not pixels: T2 object with no T1 support → new; T1 object with no T2 support → demolished; supported → existing; partially supported objects are split into existing + new/demolished parts. Pixel XOR was tested and rejected: parallax/misregistration produces sliver false positives on every building edge. |
| Speed options (UI) | **"Hızlı"**: resample to ~0.5 m/px (unknown GSD → longest side 2048 px). **"Derin analiz"**: original resolution, background job with progress. |
| Georef inputs | GeoTIFF tags; world files `.jgw/.jpgw/.tfw/.tifw/.pgw/.pngw/.wld` (+ optional `.prj`); manual entry in UI (WGS84 bounding box and/or GSD, EPSG for a world file without `.prj`). |
| Non-georef output | Pixel-space results only: `bounds`/`georef`/`centroid` = `null`, GeoJSON in pixel coordinates with `"georeferenced": false`, areas in m² only if a GSD was given, otherwise px. No fake coordinates. |
| Coordination | Built on top of the merged backend refactor (`model/geo.py`, `model/scenarios.py`, split `detect()`), on its own branch. The classic detector stays, reachable with `engine: "classic"`; existing snapshot tests pin that path. |

Rejected: **AdaptFormer** (`deepang/adaptformer-LEVIR-CD`, MIT) — binary change only (no new/demolished split), needs `transformers` + `einops` + remote code, missed several real changes on the user's image, ~20 s/MP. **aridbuild-seg** — model code not on the Hub.

## Measured feasibility (scratchpad prototype, CPU, quantized ONNX)

- Speed: 7–11 s per 1024² tile per date. User image deep mode ≈ 45 tiles × 2 dates ≈ 10–12 min; fast mode (2048 px) ≈ 6 tiles × 2 ≈ 1.5 min; Wayback live (512 px upsampled ×2 → 1 tile) ≈ 15 s.
- Object-level change vs. bundled labels (no GT used in detection): levir1 F1 0.91, levir2 0.99, levir3 0.91, dsifn2 0.62, dsifn1 0.31 (dsifn1 pair is poorly co-registered and its dense village merges into one T2 blob — motivates the partial-support split rule).
- Visual check on the user's 2020→2026 crops: new towers, new round structure and new halls correctly green; existing blocks correctly existing; shadows no longer flagged.

## Risks / open items for the user

- **License:** the geobase HF repo states no license; ChangeStar code (torchange) is Apache-2.0, but training data of these weights is not documented. Fine for the prototype; verify before commercial use.
- Very tall off-nadir towers can shift between dates enough to be called new+demolished.
- First run downloads 141 MB into `models/` (gitignored). Offline without cache → falls back to classic engine with a Turkish warning.
- Deep mode on 8192×4468 needs ~1.5 GB RAM.
