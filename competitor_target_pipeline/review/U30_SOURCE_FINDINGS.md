# U30 source investigation

The eight concise U30 files are grayscale JPEG exports, 768 x 700 pixels.
They have no EXIF tags, embedded image description, or software description.
JFIF export density information is not a detector pixel-size calibration.

No JSON/XML/CSV/TXT/YAML/DICOM/RAW sidecar candidates were found under the supplied
TRI dataset directory by the preparation script. This does not rule out metadata
elsewhere or proprietary scanner files unavailable in this workspace.

The supplied exports do not establish acquisition mode, reconstruction depth,
original bit depth, calibrated pixel size, or exposure parameters. The competitor
screen photographs suggest a slice-viewing workflow, but do not establish a
matching specimen, depth, or acquisition process. They cannot serve as aligned
ground-truth targets for our JPEGs.

## Information to obtain from TRI

1. Are our U30 images 2D projections, reconstructed slices, or processed exports?
2. Do the reference images show the same specimen and same solder layer/depth?
3. Can TRI provide a matching unprocessed input and competitor output as image
   files, instead of a photograph of a display?
4. What are the native bit depth, physical pixel size, source/detector geometry,
   exposure settings, and any preprocessing or display polarity applied?
5. Are projection images or slice stacks available, including slice spacing and
   the depth used to inspect the solder regions?
6. Which central dark/bright structures are confirmed defects versus normal
   construction or superimposed components?

U30's V5 processing intentionally retains V4 behavior. No cavities are synthesized
to resemble the competitor photograph. Inspect the matched native-resolution
U30 crop sheets before deciding whether additional processing or different source
data are needed.

The machine-readable inspection is in `u30_source_audit.json`.
