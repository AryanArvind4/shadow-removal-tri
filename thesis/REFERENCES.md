# Verified Starter References

Checked against primary sources on 2026-10-22. This is a starting bibliography,
not an exhaustive survey, novelty determination or proof of industrial validation.
Obtain and read full papers before finalizing the related-work chapter.

## R1. Non-local means

Antoni Buades, Bartomeu Coll and Jean-Michel Morel. "Non-Local Means Denoising."
Image Processing On Line, 1, 208-212, 2011.
[Primary article](https://doi.org/10.5201/ipol.2011.bcm_nlm).

Relevance: established similar-patch denoising underlying V5's OpenCV NLM stage.
The project's novelty is not NLM itself; its bounded source blending and use in
the full inspection workflow must be evaluated separately.

## R2. Retrospective shading correction

Tingying Peng et al. "A BaSiC tool for background and shading correction of optical
microscopy images." Nature Communications, 8, 14836, 2017.
[Primary article](https://doi.org/10.1038/ncomms14836).

Relevance: sequence-based sparse/low-rank background estimation, explored in the
repository's optional BaSiCPy branch. Optical-microscopy success is not evidence
of success on these exported X-ray images; confirm the backend used in each run.

## R3. Generative image editing

Chenlin Meng et al. "SDEdit: Guided Image Synthesis and Editing with Stochastic
Differential Equations." arXiv:2108.01073, 2021; revised 2022.
[Author manuscript](https://arxiv.org/abs/2108.01073).

Relevance: motivates the repository's historical diffusion exploration. Realistic
editing and preservation of inspection evidence are different evaluation goals.

## R4. Structural similarity

Zhou Wang, Alan C. Bovik, Hamid R. Sheikh and Eero P. Simoncelli.
"Image Quality Assessment: From Error Visibility to Structural Similarity."
IEEE Transactions on Image Processing, 13(4), 2004.
[Author publication page](https://eceweb.uwaterloo.ca/~z70wang/publications/ssim.html).

Relevance: reference-based structural similarity. It is not a stand-alone
defect-preservation certificate, especially without an aligned clean reference.

## R5. Marker-based watershed implementation

OpenCV. "Image Segmentation with Watershed Algorithm."
[Official documentation](https://docs.opencv.org/4.12.0/d3/db4/tutorial_py_watershed.html).

Relevance: implementation background for V5b's grid-supported segmentation branch.
This documentation is not the original watershed research paper. Add the original
method reference during the complete literature review. Cite the actual installed
OpenCV version separately from the documentation version consulted.

## R6. Histogram equalization and CLAHE baseline

OpenCV. "Histograms - 2: Histogram Equalization."
[Official documentation](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html).

Relevance: a standard baseline for local contrast, including noise amplification
and contrast limiting. Add the original CLAHE chapter to the final bibliography
after checking its publication details directly.

## R7. X-ray attenuation context

NIST. "X-Ray Mass Attenuation Coefficients - Section 2."
[Official reference](https://physics.nist.gov/PhysRefData/XrayMassCoef/chap2.html).

Relevance: exponential attenuation in an idealized narrow, monoenergetic beam.
This does not identify the formation model, reconstruction or display mapping of
TRI's supplied exports. Avoid describing those exports as optical reflectance.

## R8. Industrial context and company identity

Test Research, Inc. "TR7600 SIII" automated X-ray inspection product page.
[Official product information](https://www.tri.com.tw/en/product/product_detail-12-2-1462-21.html).

Relevance: confirms the company operates in automated PCB X-ray inspection.
It does not establish which system generated our dataset, prove any collaboration
contract, or authorize reproduction of industrial images.

## Literature Still Needed

- Industrial PCB X-ray enhancement evaluated on actual inspection tasks.
- Original robust-estimation, Retinex and morphology/watershed references.
- Stripe/ring correction and the assumptions of stack-based field estimation.
- Task-based image quality, human observer studies and uncertainty in small-feature
  inspection; sources appropriate to industrial rather than clinical conclusions.
- Current learned enhancement and constrained/non-generative alternatives,
  compared using realistic data-access and generalization assumptions.

Maintain a literature matrix: problem, data/modality, assumptions, ground truth,
algorithm, baselines, evaluation, limitations, and how it differs from this project.
No priority or state-of-the-art claim should be made from this short list alone.
