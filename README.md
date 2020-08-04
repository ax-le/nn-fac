# Nonnegative Factorization techniques
Python code for computing some Nonnegative Factorization, using an accelerated version of Hierarchical Alternating Least Squares algorithm (HALS) with resolution of Nonnegative Least Squares problem (NNLS) [1].

This work has been done during my Research Master's (SIF, master.irisa.fr) internship in PANAMA team at IRISA/Inria Rennes, under the direction of BERTIN Nancy and COHEN Jeremy.

It has been extended during my PhD in the same Team, under the direction of BERTIN Nancy and COHEN Jeremy and BIMBOT Frederic.

## Installation

You can install it using pip:

    pip install nn-fac

You can then use code by typing:

    import nn_fac

For example, if you want to use nmf, type:

    import nn_fac.nmf

Don't hesitate to reach the author in case of problem. Comments are welcomed!

## Contents
### NNLS
This toolbox contains a NNLS resolution algorithm, developed as described in [1]. This code is based on COHEN Jeremy python code adaptation of GILLIS Nicolas MatLab code.

This toolbox also contains 4 factorization methods:
### NMF
Nonnegative Matrix Factorization [2] - Factorization of a nonnegative matrix X in two nonnegative matrices W and H, where WH approach X.

This is solved by minimizing the Frobenius norm between both matrices X and WH by NNLS.

### NTF - Nonnegative PARAFAC
Nonnegative Tensor Factorization, also called Nonnegative PARAFAC decomposition. PARAFAC decomposition consists in factorizing a tensor T in a sum of rank-one tensors [3]. By concatenating the vectors along each mode of this sum, we obtain as much factors as the number of modes of the tensor [4]. This algorithm returns these factors.

This factorization is computed as an ALS algorithm, described in [5], solved with NNLS, and using the toolbox Tensorly [6]. It returns the nonnegative factors of a nonnegative tensor T.

### Nonnegative PARAFAC2
Nonnegative Tensor Factorization admitting variability over a factor [7]. More precisely, this implemented version is based on a flexible coupling approach [8], where the coupling is enforced by a penalty term.

### NTD - Nonnegative Tucker Decomposition
Nonnegative Tucker Decomposition, which consists in factorizing a tensor T in factors (one per mode) and a core tensor, generally of smaller dimensions than T, which links linearly all factors [5]. This algorithm returns these factors and this core tensor.

This factorization is computed as an ALS algorithm, described in [5], solved with NNLS, and using the toolbox Tensorly [6]. It also uses a gradient update rule for the core.

## References
[1] N. Gillis and F. Glineur, "Accelerated Multiplicative Updates and Hierarchical ALS Algorithms for Nonnegative Matrix Factorization," Neural Computation 24 (4): 1085-1105, 2012.

[2] D. D. Lee and H. S. Seung, "Learning the parts of objects by non-negative matrix factorization," Nature, vol. 401, no. 6755, p. 788, 1999.

[3] R. A Harshman et al. "Foundations of the PARAFAC procedure: Models and conditions for an" explanatory" multimodal factor analysis," 1970.

[4] J. E. Cohen, and N. Gillis, "Dictionary-based tensor canonical polyadic decomposition," IEEE Transactions on Signal Processing, 66(7), 1876-1889, 2017.

[5]  T. G. Kolda, and B. W. Bader, "Tensor decompositions and applications," SIAM review, 51(3), 455-500, 2009.

[6] J. Kossaifi, Y. Panagakis, A. Anandkumar and M. Pantic, "TensorLy: Tensor Learning in Python," Journal of Machine Learning Research (JMLR), volume 20, number 26, 2019.

[7] R. A Harshman, "PARAFAC2: Mathematical and technical notes," UCLA working papers in phonetics 22.3044, 1972.

[8] J. E Cohen and R. Bro, "Nonnegative PARAFAC2: a flexible coupling approach," International Conference on Latent Variable Analysis and Signal Separation. Springer. 2018.
