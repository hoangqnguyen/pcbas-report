# System Overview: BBox Filling & Ball Detection

Two independent SAM3-based pipelines augment SoccerNet PCBAS tactical data (H5, 13-col per-frame player tracking at 1920x1080, 25fps) with (a) filled bounding boxes for untracked players and (b) ball positions.

---

## 1. BBox Filling (`fill_sam3.py`)

Recovers bounding boxes for the ~22% of players who are visible on-screen but missing from the original tracking. The other ~78% of missing bboxes are out-of-frame or replay frames where no useful visual exists.

### Per-frame pipeline (frames with >= 4 known bboxes)

```
Known bboxes ──► Build foot-point homography
                  (pitch X_POS,Y_POS → image bottom-center of bbox)
                       │
                       ▼
Missing players ──► Project pitch coords → pixel coords via H
                       │
                       ├─ OUT_OF_FRAME (outside 1920x1080 ± 50px margin) → skip
                       ├─ REPLAY (<4 reference bboxes, H=None)            → skip
                       └─ IN_FRAME → candidate for SAM3 matching
                                         │
                                         ▼
                  SAM3 text="person" ──► All person masks in frame
                       │                 Filter: area>=80px, h/w>=0.7, w<=250, h<=400
                       ▼
                  Cross-validate: IoU-match (>0.2) GT bboxes to SAM3 dets → remove from pool
                       │
                       ▼
                  Match remaining dets to IN_FRAME missing players
                  by foot-point euclidean distance (<=150px threshold)
                       │
                       ▼
                  Write matched SAM3 bbox as filled ROI_X/Y/W/H
```

### Frame subsampling

SAM3 runs every 5th frame (`SAM3_SUBSAMPLE=5`). Detections propagate to +/-2 neighboring frames (re-matched against each neighbor's own homography). Reduces 26h full-scan to ~3h per dataset.

### Output format

Original 13 columns + 2 appended:

| Col 14 | Col 15 |
|--------|--------|
| `IS_INTERPOLATED`: 0=original, 1=SAM3-filled | `MISSING_CASE`: 0=had bbox, 1=in-frame missing, 2=replay/no-ref |

### Typical fill rate

~6.8% of all missing rows get filled (= the genuinely visible but untracked players). The remaining 93.2% are correctly left empty.

---

## 2. Ball Detection (`detect_ball.py`)

Detects the soccer ball across frames using SAM3, then selects the globally optimal trajectory via Viterbi dynamic programming.

### Per-cluster pipeline

Frames are grouped into clusters: either +/-12 frames around each event (training data with EVENT_ID column) or fixed 500-frame chunks (challenge data without events).

```
Read cluster frames from video
       │
       ▼
Pass 1: SAM3 text="soccer ball" per frame
        Filter each candidate:
          - area >= 15px, max size <= 50px, aspect ratio <= 2.5
          - within dynamic pitch bounds (player positions ± margins)
          - within 300px of nearest player (hard reject)
        → list of (cx, cy, w, h, conf) per frame
       │
       ▼
Pass 2: Viterbi optimal path through candidates
        States = candidates per frame + "no detection" sentinel
        Emission cost  = -confidence * 50 - player_proximity_bonus
        Transition cost = distance-weighted:
          <=120px/frame → 0.3x (smooth), <=360px → 1x, >360px → 5x (teleport)
        No-detection penalty = 30
        Backtrack to find lowest-cost global path
       │
       ▼
Pass 3: Median filter (window=5) on trajectory x,y
       │
       ▼
Pass 4: Linear interpolation across gaps <= 4 frames
       │
       ▼
Output: {frame: (cx, cy, w, h) | None}
```

### Player-aware filtering

Visible player foot-points (from tactical data bboxes) define a dynamic pitch region. Candidates outside this region or >300px from any player are rejected. A proximity bonus (up to 40 points for <80px from nearest player) biases the Viterbi path toward physically plausible positions.

### Output format

H5 with one dataset per game-half, each an Nx5 float32 array `[frame, cx, cy, w, h]`. Only frames with detected balls are stored (gzip-compressed).

---

## 3. Infrastructure

### Watchdog runners

Shell scripts (`run_train_fill.sh`, `run_train_ball.sh`, `run_challenge_ball.sh`) wrap each pipeline in a restart loop. Both `fill_sam3.py` and `detect_ball.py` have built-in resume support (check existing H5 keys on startup, skip completed game-halves). If SAM3 crashes (GPU OOM, etc.), the watchdog restarts after 10s and processing continues from the last completed half.

### Rendering

`render_ball_demos.py` reads pre-computed ball H5 + tactical data + video to produce annotated MP4 clips (no SAM3 needed). `detect_ball.py --render` can also render during detection. Clips show: player bboxes (gray/green for actor), ball crosshair + circle, fading yellow trajectory trail, and HUD overlay.

### SAM3 configuration

| Parameter | BBox filling | Ball detection |
|-----------|-------------|----------------|
| Confidence threshold | 0.3 | 0.15 |
| Text prompt | "person" | "soccer ball" |
| Half precision | yes | yes |
| Weights | sam3.pt (3.4 GB) | sam3.pt (3.4 GB) |

Ball detection uses a lower confidence threshold (0.15 vs 0.3) because balls are small and often low-confidence; the Viterbi pass handles false positives.

---

## 4. Data flow summary

```
tactical_data.h5 (13-col) ─┬─► fill_sam3.py ──► *_sam3filled.h5 (15-col)
                            │
videos/*.mp4 ───────────────┤
                            │
sam3.pt ────────────────────┼─► detect_ball.py ──► ball_detections_*.h5 (5-col)
                            │
                            └─► render_ball_demos.py ──► clips/*.mp4
```
