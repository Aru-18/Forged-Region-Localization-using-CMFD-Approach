# Forged-Region-Localization-using-CMFD-Approach

### Objective: 
  Localize the forged regions in the image created using the CMF approach

### Language Used: 
  Python 3

### Libraries Used:
  1. Open CV2
  2. NumPy
  3. PyWavelets
  4. Scikit-Learn
  5. Matplotlib

### Custom-Functions Created:
  **1. readImage:** Read the input image, resize it and convert it into grayscale form.
  **2. SWT:** Perform level 2 Stationary Wavelet Transform and extract only the LL sub-band.
  **3. keypoints:** Extraction of features and their corresponding feature descriptors using Scale-Invariant Feature Transform (SIFT).
  **4. showImage:** Display the image in a separate dialog box.
  **5. showKeypoints:** Display the input image with the extracted keypoints highlighted.
  **6. dbscan:** Perform 2-level DBSCAN to localize the forged regions.
  **7. drivercode:** For each input image, executes all the above mentioned functions in an orderly manner.

### Dataset Used:
  CoMoFoD v2_small (public)
