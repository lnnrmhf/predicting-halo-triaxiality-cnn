# Predicting Halo Triaxiality in Cosmological Simulations using a CNN

**Final project — Course: Data Science in Astrophysics**  
Institute for Astronomy (IfA), University of Vienna — July 2026

Prof. Oliver Hahn & Ass.-Prof. Dr. Sudeshna Boro Saikia

---

## Overview

The aim of this work is to predict dark matter halo geometry directly from the initial linear density field in N-body simulations using a 3D convolutional neural network (CNN). This work is part of a larger project, where Raffael Jonas investigates the prediction of structural collapse and the halo mass function. 

N-body simulations have demonstrated that dark matter halos formed through gravitational collapse are non-spherical, showing shapes that depend on halo mass, assembly history and cosmic environment. Determining halo geometries accurately is essential because halo shapes directly influence key physical properties and observations, such as subhalo orbital dynamics, spatial anisotropies in satellite galaxy distributions, two-point correlation functions, weak gravitational lensing signals and more. Mapping the origin and distribution of halo shapes therefore provides crucial insights into the non-linear growth of cosmic structure and is vital for accurate cosmological parameter inference.

While N-body simulations provide the ground truth for extracting halo geometries, running full numerical simulations and halo-finding algorithms is computationally expensive. This work applies a machine learning architecture to overcome this computational bottleneck.

Training, validation and test datasets were constructed from cosmological N-body simulations generated with DISCO-DJ (Hahn et al. 2024; List & Hahn 2024), with halos and their axis ratios (*b/a* and *c/a*) identified using the ROCKSTAR halo finder (Behroozi et al. 2012). A 3D U-Net CNN was trained on normalized, augmented sub-volumes using a Mean Squared Error (MSE) loss function.

While the model successfully identified collapsed regions on large scales, it failed to predict halo axis ratios accurately. Instead, predictions collapsed to the mean near halo centers with artificial, decreasing gradients toward halo boundaries, resulting in negative coefficient of determination values; R² = −2.45 for both axis ratios. Single-sample overfitting diagnostics confirmed that the network struggles to generalize geometric features across independent simulations. Furthermore, masking background voxels to isolate halo shapes caused predicted values to scatter tightly around the dataset mean, resulting in R² ≈ 0. The CNN under this configuration cannot extract physically meaningful halo axis ratios from initial density fields.

The full methods, results and discussion are written up in a final report: [`DataScience_final_Einramhof.pdf`](DataScience_final_Einramhof.pdf).

---

## Repository structure

```
.
├── cnn_shape.ipynb                                       # Main notebook: data loading, model, training, evaluation
├── generate_simulations.py                               # Generate N-body simulations using DISCO-DJ
├── generate_labels_shape.py                              # Assign halo axis ratios to Lagrangian grid cells
└── DataScience_final_Einramhof.pdf                       # Final report
```

---

## Methods

### Generating simulations (`generate_simulations.py`)

The differentiable software package for large-scale structure cosmology DISCO-DJ (DIfferentiable Simulations for COsmology - Done with JAX) (List et al. 2026; Hahn et al. 2024) was used to generate 110 N-body simulations. The simulation set-up closely follows the work of Buisman et al. (2025); they were set up with a box size of L = 100 Mpc/h, N = 256³ particles and a PM grid with N_g = 1024³ cells. The fiducial flat ΛCDM cosmology was used: Ω_c = 0.259605, Ω_b = 0.0488911, h = 0.67742, n_s = 0.96822, σ₈ = 0.808992. The simulation was initialized using second-order LPT at a = 0.02 (z = 49) and evolved to a = 1.0 (z = 0) using the BULLFROG N-body integrator (Rampf et al. 2025) in 100 time steps.

The initial field of the simulations was saved to serve as input for the CNN. The particle positions and velocities at z = 0 and their IDs were stored in an `.ascii` file to be further processed using the ROCKSTAR halo finder.

### Identifying halos and generating labels (`generate_labels_shape.py`)

ROCKSTAR (Robust Overdensity Calculation using K-Space Topologically Adaptive Refinement) is a halo finder based on adaptive hierarchical refinement of friends-of-friends groups in six phase-space dimensions and one time dimension (Behroozi et al. 2012). It was used to identify halos in the simulation box and assign particles to them. From the halo catalog, the virial mass, virial radius, axis ratios *b/a* and *c/a* and parent IDs were extracted for each halo.

To generate labels, an initial 3D Lagrangian position grid was constructed to map each particle back to its grid cell. Following Buisman et al. (2025), all substructures were excluded and only host halos with a minimum virial mass of M = 4 × 10¹² M⊙/h, corresponding to ~800 particles, were considered. Host halo particles were then mapped back to their original Lagrangian grid coordinate with their halo's axis ratios (*b/a* and *c/a*) stored in the voxels.

Crucially, ROCKSTAR misses ~2 × 10⁶ particles of the ~16 × 10⁶ particles and does not assign them to halos. Inspecting their positions revealed that they are clearly clustered and therefore cannot be considered background particles and ignored. Consequently, `scipy.spatial.cKDTree` was used to find the five nearest host halos of the unidentified particles. If an unassigned particle is located within the virial radius of a nearby host halo, it was assigned to that host halo and the axis ratios stored in the particle's Lagrangian grid cell.

The resulting two Lagrangian label tensors were stored as arrays of shape (256, 256, 256). The individual axis ratios are given to the model as a two-channel target rather than the triaxiality parameter T directly — this prevents the network from learning degenerate solutions, as multiple distinct axis ratio pairs can map to the same T value.

### Data processing

The 110 N-body simulations were randomly divided into 90 simulations reserved for training, 10 for validation, and 10 for testing. The entire training set was normalized using a standard z-score, with the global mean and standard deviation computed across all 90 training simulations.

Since a full volume box consists of 256³ particles, which quickly exceed the computing capabilities given the large CNN structure, a `random_crop` function was implemented, which selects a random point inside the box and returns a cropped cube of size 80³ with periodic wrapping conditions. Data augmentation via random flipping along any of the three axes and arbitrary rotations of the cube increases the effective training set size from 90 to 4320 simulations (a factor of 48), exploiting the cosmological principle and isotropy.

### CNN architecture

The CNN used in this work is heavily based on Raffael Jonas` CNN for halo mass predictions. 

The formation and final geometry of a halo are determined by many factors that span large physical scales; the architecture must therefore reflect this multiscale influence. A 4-level 3D U-Net (also called a V-Net in 3D) was implemented in PyTorch (Ronneberger et al. 2015). In the encoder, the initial density field is compressed in physical size at each layer while the feature depth increases (32 → 64 → 128 → 256 channels), starting from a sub-volume of 80³ particles. The maximum compression is achieved in the bottleneck, where the 3 × 3 × 3 convolutional kernel spans a physical scale of 9.375 Mpc/h. In the decoder, skip connections concatenate information from each encoder level with the up-sampled output of the layer below, allowing the subsequent convolution to draw on both coarse abstract features from the bottleneck and fine spatial detail preserved by the skip connection. The final 1 × 1 × 1 convolution maps the abstract features down to two output channels, corresponding to the spatial distributions of *b/a* and *c/a*, respectively.

At each layer, convolution was performed with `Conv3d` using a 3 × 3 × 3 filter and circular padding, motivated by the periodic boundary conditions of the simulations. `LeakyReLU(0.05)` was used as the activation function. Importantly, even though axis ratios are physically bounded between 0 and 1, the final layer uses a linear output rather than a restricted activation function — adding a bounded activation function caused the network to stop learning very early in the training process, as these functions cause gradients to vanish quickly near the boundaries. Through trial and error, the dropout rate was also set to zero for the final network.

Two model variants were trained:

1. **Baseline** (`model_best_background_nosigmoid_nodropout_loss.pt`): standard MSE loss computed over all voxels; background voxels (originally NaN) were mapped to 0.0 during training.
2. **Masked** (`model_best_masked_nosigmoid_nodropout.pt`): MSE loss computed over bound halo voxels only, to isolate shape prediction from collapse classification.

![`4-level U-Net structure of the CNN`](/figures/Jonas_Einramhof.pdf).


### Training

The model was optimized using Adam (Kingma & Ba 2017) with a learning rate of 10⁻⁴. The MSE loss was calculated voxel-by-voxel across the 3D grid and normalized by the total volume of the cube (80³), computed independently for each axis ratio channel with the total loss defined as their sum. Training was performed on a single NVIDIA RTX 3090 GPU with 24 GB of VRAM for a maximum of 1000 epochs (≈14 seconds per epoch). An early stopping criterion was implemented, stopping training if no improvement in validation loss was observed for 40 consecutive epochs; this was triggered after 520 epochs for the baseline model (~2 hours total) and after 52 epochs for the masked model (~25 minutes). The parallel trajectory of training and validation loss curves confirms that there is no architectural overfitting.

---

## Results

### Collapse identification (binary classification, ε = 0.05 threshold)

The high TPR of 0.976 indicates that halo structures are overall successfully identified by the network. However, the lower PPV and F₁ values highlight a high rate of false positives, showing that the model tends to bleed shape predictions into the background.

| Metric | This work | Buisman et al. (2025) | López-Cano et al. (2024) |
|--------|-----------|----------------------|--------------------------|
| TPR    | 0.976     | 0.732                | 0.838                    |
| TNR    | 0.736     | 0.956                | 0.883                    |
| PPV    | 0.524     | 0.822                | 0.838                    |
| ACC    | 0.791     | 0.907                | 0.864                    |
| F₁     | 0.682     | 0.774                | 0.838                    |

### Axis ratio regression

The model fails to reproduce the optimal diagonal relation between predicted and true values, instead predicting a broad range of values. The negative R² scores quantitatively confirm an anti-correlation and highlight the network's inability to capture the underlying geometric distribution. The single-sample overfitting experiment (R² ~ 0.9) demonstrates that the primary limitation is not the architectural capacity alone but rather a failure to generalize learned structural features across unseen simulations.

| Experiment | R² (b/a) | R² (c/a) | R² (T)  |
|------------|----------|----------|---------|
| Baseline   | −2.45    | −2.45    | −0.69   |
| Masked     | −0.01    | −0.01    | −0.03   |
| Single-simulation overfit | ~0.9 | ~0.9 | ~0.87 |

When background voxels are masked out to isolate halo shape prediction, the collapse toward the mean becomes even more pronounced — predictions collapse into narrow, horizontal bands that fail to track the ideal fit, quantitatively confirming an absence of correlation. This indicates that the network remains unable to extract genuine structural information even when constrained to halo voxels exclusively.

---

## Data

The full simulation dataset was generated and stored on a university workstation and is not included here due to its size. 

---

## References

- Behroozi, P. S., Wechsler, R. H., & Wu, H.-Y. 2012, ApJ, 762, 109 — *ROCKSTAR halo finder*
- Buisman, J., List, F., & Hahn, O. 2025 — *Differentiable Halo Mass Prediction and the Cosmology-Dependence of Halo Mass Functions*
- Hahn, O., List, F., & Porqueres, N. 2024, JCAP, 2024, 063 — *DISCO-DJ*
- Kingma, D. P. & Ba, J. 2017 — *Adam: A Method for Stochastic Optimization*
- List, F. & Hahn, O. 2024, Journal of Computational Physics, 513, 113201
- List, F., Hahn, O., Flöss, T., & Winkler, L. 2026, JCAP, 2026, 033 — *DISCO-DJ v2*
- López-Cano, D. et al. 2024 — *Characterizing Structure Formation through Instance Segmentation*
- Paszke, A. et al. 2019 — *PyTorch*
- Rampf, C., List, F., & Hahn, O. 2025, JCAP, 2025, 020 — *BULLFROG integrator*
- Ronneberger, O., Fischer, P., & Brox, T. 2015 — *U-Net: Convolutional Networks for Biomedical Image Segmentation*

Full bibliography in the accompanying paper.

---
