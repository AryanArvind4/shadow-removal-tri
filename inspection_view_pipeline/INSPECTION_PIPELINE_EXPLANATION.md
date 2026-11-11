# Inspection View Pipeline Explanation

This document explains the inspection-view work in simple words.

The goal of this pipeline is to make TRI PCB X-ray images cleaner and easier to
inspect for solder defects, voids, fragments, missing joints, and uneven solder
regions.

We are currently presenting two result types:

1. `V3 balanced_gray_clean`
2. `V1 inspection_soft`

We are not presenting `V3 inspection_clean`, because it looks too much like a
segmentation mask and can create false solder-like fragments.

---

## 1. Basic Principle

The X-ray image has two kinds of information mixed together:

```text
1. Slow background variation
   This is the shadow / metal sheet effect.
   It changes slowly across the image.

2. Local PCB structure
   This includes solder balls, pads, vias, traces, square grids, and defects.
   These are sharper and more local.
```

The pipeline tries to separate these two.

In simple terms:

```text
Find the smooth shadow.
Reduce that shadow.
Keep the small PCB details.
Enhance the useful details carefully.
```

This is a classical image-processing pipeline. It does not train a neural
network.

---

## 2. Why We Need Two Views

TRI has two practical needs.

### View 1: Balanced Grayscale View

This is `V3 balanced_gray_clean`.

Purpose:

```text
Keep the image grayscale.
Reduce shadow.
Preserve intensity variation inside solder regions.
Make defects/voids easier to inspect.
```

This is the main output for Well's feedback.

Well said that ideally the solder region should have a more uniform grayscale
distribution, but shadows make it uneven. So we need an output that still looks
like an X-ray image, not just a black/white mask.

### View 2: Inspection-Enhanced View

This is `V1 inspection_soft`.

Purpose:

```text
Make solder balls, pads, vias, and traces easier to visually locate.
Give a quick inspection-style enhancement.
Keep more grayscale than the aggressive dark view.
```

This is useful for human review and presentation because it clearly shows that
features are more visible after processing.

---

## 3. V1 Inspection Soft: Step-by-Step

V1 was our first inspection-view pipeline.

The output we are keeping from V1 is:

```text
inspection_view_pipeline/result/inspection_view_v1/inspection_soft
```

### Step 1: Read the X-ray Image

The image is loaded as grayscale.

Every pixel is just a brightness value:

```text
dark pixel   = low value
bright pixel = high value
```

### Step 2: Estimate the Background Shadow

The pipeline applies heavy smoothing to the image.

Why?

Because solder balls and pads are small/sharp, but the shadow is broad and slow.
When we blur heavily, the small features mostly disappear and the broad shadow
remains.

In simple words:

```text
Blur the image a lot.
The result is our estimate of the shadow/background.
```

### Step 3: Flatten the Background

After estimating the background, the pipeline reduces its effect.

It uses two correction ideas:

```text
Subtractive correction:
    corrected = image - background + average_background

Divisive correction:
    corrected = image / background * average_background
```

Subtractive correction is useful when the shadow behaves like an added dark
layer.

Divisive correction is useful when the shadow behaves like uneven illumination.

V1 mixes both because the TRI images do not behave perfectly like only one type.

### Step 4: Extract Local Features

The pipeline then searches for local structures.

It uses:

```text
local contrast
top-hat filtering
edge/gradient strength
```

In simple words:

```text
Look for pixels that stand out from their local neighborhood.
```

This helps highlight:

- solder balls
- solder pads
- square pads
- vias
- traces
- local defect-like regions

### Step 5: Create the Soft Inspection View

The `inspection_soft` result blends the flattened grayscale image with the
extracted feature image.

This means:

```text
Original structure is still visible.
Features are enhanced.
The image does not become fully black/white.
```

That is why `inspection_soft` is better for presentation than the more aggressive
`inspection_dark`.

---

## 4. V3 Balanced Gray Clean: Step-by-Step

V3 was created after TRI/Well feedback.

The output we are keeping from V3 is:

```text
inspection_view_pipeline/result/inspection_view_v3/balanced_gray_clean
```

V3 focuses more on preserving grayscale information.

### Step 1: Read the Image

Same as V1, the image is read as grayscale.

### Step 2: Work at a Smaller Internal Size

Large J1/J4 images are big and slow to process.

So V3 temporarily resizes large images to a smaller working size, processes
them, and then resizes the result back.

Why this is acceptable:

```text
The shadow is broad and slow.
We do not need full resolution to estimate the broad shadow field.
```

### Step 3: Estimate the Broad Background

V3 estimates the background/shadow by:

```text
1. applying median filtering
2. applying strong Gaussian blur
```

Median filtering helps reduce small bright/dark features.

Gaussian blur makes a smooth background field.

The result should mostly contain:

```text
shadow pattern
illumination pattern
broad panel brightness change
```

It should not contain detailed solder balls or square pads.

### Step 4: Shadow Flattening

V3 reduces the shadow using a mix of additive and divisive correction.

The idea is:

```text
If one region is dark only because of shadow, lift it.
If one region is bright only because of illumination, reduce it.
```

This makes the image more even.

### Step 5: Suppress Horizontal Bands

J1 and J4 have strong horizontal line/seam artifacts.

V3 estimates the median brightness of each image row.

Then it separates:

```text
slow row trend
fast row band artifact
```

It subtracts part of the row-band artifact.

In simple words:

```text
If a whole row is weirdly bright or dark, reduce that row effect.
```

This helps reduce horizontal seams without completely destroying solder balls.

### Step 6: Compute Local Feature Strength

V3 computes a feature map using:

```text
local contrast
top-hat features
edge strength
```

This tells the pipeline where useful structures probably are.

The feature map is not the final output. It is used as a guide.

### Step 7: Local Contrast Boost

V3 applies a mild local contrast enhancement using CLAHE.

CLAHE means:

```text
Contrast Limited Adaptive Histogram Equalization
```

Simple explanation:

```text
It improves contrast locally in small regions,
but limits the strength so noise does not explode too much.
```

We use it carefully because too much CLAHE makes U29/U30 look noisy.

### Step 8: Blend Grayscale + Feature Enhancement

This is the most important part.

V3 does not simply make features white and background black.

Instead:

```text
Most of the output comes from the cleaned grayscale image.
A smaller part comes from the feature enhancement.
```

That is why `balanced_gray_clean` still keeps grayscale inside solder regions.

This matters because voids and defects may appear as darker or uneven regions
inside solder pads.

If we turn everything into white blobs, those defects disappear.

---

## 5. Why We Do Not Present V3 Inspection Clean

V3 also generated an `inspection_clean` output.

But after visual review, it is not suitable for presentation.

Problems:

```text
It looks like a segmentation mask.
It creates false solder-like fragments.
It removes too much grayscale information.
It goes against Well's feedback about preserving solder-region grayscale.
```

So we should not present it.

---

## 6. Benefits of the Inspection-Enhanced View

The inspection-enhanced view is not only about "better visibility".

It has several practical benefits.

### 1. Faster Human Review

Important solder structures stand out more clearly.

This helps a reviewer quickly locate:

- solder balls
- pads
- vias
- missing-looking structures
- unusual bright/dark regions

### 2. Easier Region-of-Interest Selection

If TRI later wants automated defect detection, the inspection view can help
identify where the important solder/pad regions are.

In simple words:

```text
It can help tell the algorithm where to look.
```

### 3. Better Visual Communication

For presentation, it clearly shows progress.

The original images are heavily shadowed. The inspection-enhanced view makes the
important PCB pattern much clearer to the audience.

### 4. Possible Preprocessing for Detection

Some defect-detection algorithms work better when useful structures are more
separated from the background.

The inspection view can act as a preprocessing step for:

- solder/pad localization
- connected-component analysis
- feature extraction
- candidate defect proposal

### 5. Reduces Background Distraction

The inspection-enhanced view suppresses broad background intensity.

This means the reviewer focuses more on solder structures and less on the metal
sheet shadow.

Important note:

```text
The inspection-enhanced view is a helper view.
The balanced grayscale view is safer for final defect/void interpretation.
```

---

## 7. Metrics We Used

Because we do not have perfect clean ground-truth images, we cannot use simple
accuracy metrics like PSNR or SSIM.

Instead, we use practical image-quality metrics.

### Metric 1: Feature Density

Meaning:

```text
How much of the image contains visible enhanced structure.
```

If feature density is too low:

```text
Important solder/pad features may be missing.
```

If feature density is too high:

```text
Noise and background texture may also be enhanced.
```

So feature density is not simply "higher is better". It must be interpreted with
visual review.

V1 example averages:

```text
J1 inspection_soft:  0.893
J4 inspection_soft:  0.664
U29 inspection_soft: 0.732
U30 inspection_soft: 0.261
```

Interpretation:

```text
J1/J4/U29 have many visible enhanced features.
U30 is more selective, which can be good because U30 has large areas where we do
not want too many false features.
```

V3 balanced gray averages:

```text
J1 balanced_gray_clean:  0.952
J4 balanced_gray_clean:  0.862
U29 balanced_gray_clean: 0.899
U30 balanced_gray_clean: 0.813
```

Interpretation:

```text
Balanced grayscale keeps most image content visible.
This is expected because it is not a dark-background segmentation view.
```

### Metric 2: Contrast Score

Used mainly in V1.

Meaning:

```text
How strongly bright features stand out compared with darker background.
```

Higher contrast means features are easier to see.

V1 inspection_soft averages:

```text
J1:  3.57
J4:  4.85
U29: 5.12
U30: 6.74
```

Interpretation:

```text
U30 has strong feature/background separation.
J1 is lower because the image has many horizontal structures and background
texture, making separation harder.
```

But contrast score alone is not enough.

If contrast is pushed too hard:

```text
features become white blobs
voids disappear
fake fragments appear
```

This is why Well's feedback matters.

### Metric 3: Saturation Ratio

Used in V3.

Meaning:

```text
How many pixels are almost pure white.
```

This matters because if solder regions become pure white, they lose internal
contrast.

Well specifically pointed out a "no contrast" problem.

So lower saturation is usually better.

V3 balanced gray averages:

```text
J1:  1.32%
J4:  1.19%
U29: 0.45%
U30: 0.60%
```

Interpretation:

```text
Very little of the grayscale output is saturated.
That means internal grayscale information is mostly preserved.
```

V3 inspection_clean saturation was 0%, but that does not mean it is good.

Why?

```text
It avoided pure white, but still looked like fragments/blobs.
So saturation alone cannot judge quality.
```

### Metric 4: Background Leak

Used in V3.

Meaning:

```text
How much brightness remains in areas considered background.
```

For dark inspection views:

```text
low background leak is good
```

For balanced grayscale views:

```text
background leak is naturally higher
```

Why?

Because balanced grayscale is not supposed to make the background black. It keeps
the image as a grayscale X-ray.

V3 inspection_clean background leak:

```text
J1:  1.01%
J4:  0.48%
U29: 0.00%
U30: 0.00%
```

This means the background is very dark.

But again, the image is not good enough because it creates fake fragments.

V3 balanced_gray_clean background leak:

```text
J1:  92.40%
J4:  76.54%
U29: 78.93%
U30: 57.44%
```

This is expected because the grayscale image keeps background context.

So background leak should not be used to reject `balanced_gray_clean`.

---

## 8. How to Explain This in the Presentation

A simple presentation explanation:

```text
We observed that the shadow is mostly a smooth, low-frequency background effect,
while solder joints and PCB features are local high-frequency structures.

So we estimate the smooth background, reduce it, and then enhance local PCB
features.

Based on TRI feedback, we now provide two views:

1. Balanced grayscale restoration:
   preserves solder-region grayscale for void/defect analysis.

2. Inspection-enhanced view:
   makes solder balls, pads, vias, and traces easier to visually inspect.
```

Then show:

```text
Original image
V3 balanced_gray_clean
V1 inspection_soft
```

Recommended wording:

```text
The balanced grayscale output is the main defect-analysis image.
The inspection-enhanced output is a supporting visualization for feature review.
```

---

## 9. Current Best Result Folders

Main result:

```text
inspection_view_pipeline/result/inspection_view_v3/balanced_gray_clean
```

Supporting inspection view:

```text
inspection_view_pipeline/result/inspection_view_v1/inspection_soft
```

Do not present:

```text
inspection_view_pipeline/result/inspection_view_v3/inspection_clean
inspection_view_pipeline/result/inspection_view_v3/feature_mask
inspection_view_pipeline/result/inspection_view_v3/background_field
```

Those are useful for debugging, not final presentation images.

