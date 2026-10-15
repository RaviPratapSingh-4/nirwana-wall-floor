# Wall and Floor Mask Generator

Given an interior photo, this produces two pixel masks: one for the **walls** and one for the **floor**.

It was built for a 24-hour take-home assignment with one hard rule: **no pre-trained or fine-tuned segmentation model, and no segmentation API.** So everything here is classical computer vision: superpixels, k-means clustering, Gaussian colour models and some simple geometry.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xVDm1d9bkptrOSyD4l38QxJ9AeWfxNei?usp=sharing)

The notebook is the deliverable. It runs end to end on a fresh Colab runtime, clones the labelled images from this repo, evaluates the pipeline, and has a cell at the bottom where you can upload your own photo and get masks back.

---

## Quick start

1. Click the **Open in Colab** badge above.
2. Choose **Runtime > Run all**.
3. The last cell asks for an image. Upload any interior photo and it shows the wall and floor overlays, prints the runtime, and saves the two masks as PNGs.

No GPU is needed. Everything runs on the free CPU runtime.

The same notebook is also saved in this repo as `nirwana_wall_floor.ipynb`.

## What's in this repo

```
nirwana_wall_floor.ipynb   # the notebook (same one as the Colab link)
README.md
data/
  labeled/                 # the images I evaluate on, with ground truth
```

`data/labeled` contains:

- 6 Unsplash interior photos that I labelled by hand with [labelme](https://github.com/wkentaro/labelme) (`.jpg` plus the labelme `.json`)
- `dev01`, a sample image that was provided with the assignment, together with its ground-truth masks (`dev01.webp`, `dev01_wall.png`, `dev01_floor.png`)

The notebook turns the labelme JSON files into `_wall.png` and `_floor.png` masks when it runs, so those aren't stored here for the six hand-labelled photos.

### How I labelled

I used three labels: `floor`, `wall` and `wall_hole`.

- A rug counts as **floor**.
- Windows, curtains, cabinets, ceiling and furniture are **not wall** and **not floor**. I outlined windows and similar openings as `wall_hole` and subtracted them from the wall.
- Anything that fits none of the labels (beds, sofas, chairs, lamps) is simply left unlabelled and counts as background.

---

## How it works

The idea is to avoid per-pixel work as much as possible. The image is cut into about 300 superpixels, and all the decisions are made on those.

1. **Shrink and cut up.** Resize to 384 px on the long side, then run SLIC to get about 300 superpixels. Each superpixel gets a feature vector: mean Lab colour, brightness spread (a rough texture cue), and its position (x, y).
2. **Find the wall.** Run k-means (k = 8) on the superpixel features. The wall is the biggest cluster that sits in the upper part of the image and isn't mostly floor. A single wall often gets split across several clusters by shading, so clusters whose colour centre is close to the winning one are merged in.
3. **Learn the floor colour.** Fit a 3-component Gaussian mixture on the Lab colours of the bottom strip of the image, ignoring pixels already labelled wall. A superpixel counts as floor if it fits this model well enough **and** fits it better than a model fitted on the top of the image (which is mostly wall). Only blobs that touch the bottom edge are kept.
4. **Fill in the rugs.** Take the convex hull of each large floor blob. A dark rug has the same colour as the sofa next to it, so colour alone can't recover it; the hull assumes the floor is one roughly convex region.
5. **A safety guard.** If the predicted floor covers more than 30% of the frame, the colour seed was probably wrong (for example, a bed at the bottom of the photo), so only the bottom 15% of rows is kept.
6. **Upsample** the masks back to the original size.

## Results

Evaluated on 7 images (6 hand-labelled plus dev01). Metric is IoU against the ground-truth masks (my labels for the six photos, the provided masks for dev01).

| Version | Wall IoU | Floor IoU |
|---|---|---|
| v1: GMM floor model | 0.325 | 0.324 |
| v2: + convex hull for rugs | 0.325 | 0.362 |
| v3: + wall picked first, floor-vs-wall colour test, cluster merging | 0.606 | 0.391 |
| **Final: v3 + floor guard** | **0.606** | **0.442** |

Median runtime is roughly half a second per image on the free Colab CPU. Timings on Colab are noisy, so treat that as a ballpark.

What made a difference:

- **Merging similar wall clusters** was the biggest improvement for the wall.
- **The convex hull** fixed the dark rug on the dev01 sample (floor IoU 0.654 to 0.808), though it also swallows chair legs.
- **The floor guard** helped on photos where only a thin strip of floor is visible.

What I tried that did nothing measurable, and removed: vertical morphological closing, tuning the likelihood threshold and the height of the seed strip, and a "floor must be below the wall base" rule.

## Limitations

I'd rather be upfront about these:

- **Small evaluation set.** Seven images, and I chose the settings on those same images, so the numbers are optimistic. dev01 is an easy scene (wall IoU about 0.90) and pulls the average up.
- **Colour-only floor model.** When only a thin strip of floor is visible and the bottom of the frame is a bed, sofa or white wall base, the floor estimate floods the frame. The guard limits the damage but doesn't fix it. The next thing I'd try is a geometric cue for the wall-floor edge, such as Hough lines or vanishing points.
- **Merge distance is sensitive.** Merging too aggressively hurts easy scenes; at a larger merge distance dev01's wall IoU dropped from 0.90 to 0.58. The value I used is a compromise.
- **Convex hull assumption.** It assumes one convex floor region, so L-shaped floors and long hallways can go wrong.
- **Strong wall texture.** Floral wallpaper and panelled walls get under-segmented.
- **Not tested:** photos that contain no wall or no floor at all.

## Tools and disclosure

- **Libraries:** NumPy, OpenCV, scikit-image (SLIC), scikit-learn (k-means, Gaussian mixture), SciPy, Matplotlib.
- **Tools:** labelme for ground-truth labelling, Google Colab.
- **AI assistance:** I used Claude (Anthropic) as a mentor and pair-programmer: design discussion, debugging, and drafting some of the code. I ran all the experiments myself and the numbers above are from my own runs.
- **Not used:** any pre-trained or fine-tuned segmentation model, any segmentation API.

## Image credits

The photos in `data/labeled` are from [Unsplash](https://unsplash.com) and are used here for evaluation only. All rights remain with the original photographers; filenames keep the photographer and Unsplash ID so each photo can be traced back to its source. `dev01` was provided with the assignment and is not my own image.

## Author

Ravi Pratap Singh
