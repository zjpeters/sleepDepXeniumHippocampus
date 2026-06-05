---
header-includes:
  - \usepackage{mathrsfs}
geometry: margin=1in
output: pdf_document
---

### Clustering of and analysis of Xenium data

Data analysis was performed on Xenium samples from a total of 5 male mice (3 NSD, 2SD), using the stock gene panel from 10x Genomics containing a total of 297 genes, which we represent as $N_g$. The data for each sample was restricted to include only cells contained within the hippocampal formation, giving a unique number of cells for each sample, identified as $N_c$. The gene expression matrix for each mouse, with a size of $N_g \times N_c$, was then log2 normalized. Cosine distance was calculated for each cell within the hippocampal formation of a single sample by taking 1 minus the dot product of a vector of the transcriptomic data for each cell divided by the product of the norms of each vector as in Eq. 1.

$$
D_c(c_i,c_j) = 1 - \frac{c_i \cdot c_j}{\|c_i\|\|c_j\|}
$$

Where $c_i$ and $c_j$ are individual cells within the dataset. This calculation gave us a distance metric $D_c$ where 0 means two cells are identical in expression, while 1 means cells display opposite expression. 

We performed a spectral clustering of each sample by first collecting each of these distances $D_c$ into an $N_c \times N_c$ matrix for each sample called $W$ that was then min-max normalized and used as the adjacency matrix when calculating the graph Laplacian. We then used $W$ to calculate a diagonal matrix $D$ and calculated the graph Laplacian for each sample by subtracting $W$ from $D$, giving us $L$. We then calculated 250 eigenvectors and eigenvalues for $L$ and used the sorted eigenvectors as input for a k-means clustering with a $k=15$ on each sample. 

Each clustering for the individual samples was visually inspected and each of the 15 clusters was assigned an anatomically based cluster name for one of the following regions: CA1, CA2, CA3, dentate gyrus (DG), DG/CA4, or nonspecific cells. Using these labels we then ran a t-test for all cells within the CA1 and CA3 regions separately, looking for changes in gene expression across all cells in a particular region. Sidak correction was used to account for multiple comparisons, where an alpha of $p=0.05$ was adjusted based on the number of t-tests performed, i.e. the number of genes (Eq. 2). A gene was considered to be significantly differentially expressed if the p-value for the t-statistic was below the calculated $\alpha_s$ of 0.000173.

$$
\alpha_s = 1 - (1-\alpha)^{1/N_g}
$$

![Figure X](figures/xenium_figure_k15cluster_umap_rbm3ttest_gpr161ttest.svg "Figure X")

### Figure description

Figure X: Clustering, UMAP, and t-statistics of Xenium data. A Shows the clustering of a single Xenium sample within the hippocampal formation, labeled for regions of particular interest. B Is the UMAP plot of all cells from all samples used in the analysis, labeled with the regions of interest. C Displays the t-statistic for the gene Rbm3 in the CA1 and CA3 regions of the hippocampus, with a significant decrease in expression in the CA1 ($p < 5.62e^{-5}$). D Displays the t-statistic for Gpr161 in the CA1 and CA3 regions of the hippocampus, with a significant increase in expression in the CA3 ($p < 3.15e^{-24}$)
