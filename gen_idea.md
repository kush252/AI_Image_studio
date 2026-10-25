## Image Enhancement Studio — Project Scope

Goal: Build an interactive image editor using classical IPCV algorithms. Users upload an image, adjust enhancement parameters through sliders, and see the output update dynamically.

Recommended stack: Python, OpenCV, NumPy, and Streamlit.

### 1. Compulsory features (MVP)

These are sufficient for a functional mini-project.

| Feature                     | Implementation                                          |
| --------------------------- | ------------------------------------------------------- |
| Image upload                | Support JPG, PNG, and grayscale images                  |
| Original vs. edited preview | Side-by-side display                                    |
| Brightness adjustment       | Pixel intensity transformation                          |
| Contrast adjustment         | Linear contrast stretching                              |
| Gamma correction            | Power-law transformation                                |
| Histogram display           | Plot intensity distribution before and after processing |
| Dynamic preview             | Update image when a slider or filter selection changes  |
| Reset button                | Restore original image and default parameter values     |
| Download                    | Save the enhanced image                                 |

Core algorithms to implement:

1. Brightness and contrast transformation
2. Gamma correction
3. Histogram equalization
4. At least one smoothing filter: averaging or median
5. At least one sharpening filter: high-pass or high-boost

Histogram equalization, smoothing, and sharpening can be separate selectable operations rather than always applied together.

### 2. Optional features

Add these only after the compulsory features work.

| Feature                                                  | Value                                                                             |
| -------------------------------------------------------- | --------------------------------------------------------------------------------- |
| Adjustable median and averaging filter kernel size       | Demonstrates neighbourhood processing                                             |
| High-boost sharpening strength slider                    | Shows the effect of sharpening intensity                                          |
| Selectable histogram equalization or contrast stretching | Enables algorithm comparison                                                      |
| Before/after histogram comparison                        | Makes the processing effect easier to explain                                     |
| PSNR and SSIM evaluation                                 | Quantitative image-quality analysis, when a suitable reference image is available |
| Preset modes: Low-light, Noisy, Sharpen                  | Convenient combinations of processing operations                                  |
| Processing pipeline                                      | Apply multiple operations in sequence                                             |
| Undo/redo                                                | More editor-like experience                                                       |
| Webcam capture                                           | Capture a photo and edit it                                                       |
| Batch processing                                         | Enhance several images together                                                   |

### 3. Interface layout

## Image Enhancement Studio

Upload an image

#### Enhancement controls

Example controls to include in the actual app

Brightness

Contrast

Gamma

Original

Enhanced

Conceptual layout, not a working editor.

### 4. Technical requirements

- Preserve the original image in memory; apply each adjustment to the original or a defined processing pipeline to avoid cumulative degradation.
- Use OpenCV and NumPy for image operations.
- Update the preview as controls change.
- Avoid excessive image resizing or recomputation that makes slider movement lag.
- Keep the processing functions separate and testable.


concepts in syllabus :
Image Transforms & Morphological Processing: Fourier Transform, IDDFT,
Frequency domain techniques - 2D-DFT, Low pass Filter (Ideal, Butterworth,
Gaussian), High pass Filter (Ideal, Butterworth, Gaussian). Hadamard Transform,
Walsh Transform. Morphological Operations: Dilation, Erosion, Opening, Closing
Geometric Transformations: Translation, Rotation, Scaling, Shearing Feature
Detection & Description: Interest or Corner Point Detectors- Harris and Hessian.
Histogram of Oriented Gradients, Scale Invariant Feature Transform(SIFT), Speeded
up Robust Features(SURF). Scale-Space Analysis- Image Pyramids
Object Segmentation & Detection: Canny Edge Detection, Difference of Gaussian (DOG),
Graph Technique, Distance Metrics, Global Thresholding, Otsu's Thresholding, Watershed
Algorithm