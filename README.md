
# nn-fac: Nonnegative Factorization techniques toolbox
Python code for computing some Nonnegative Factorizations, using either an accelerated version of Hierarchical Alternating Least Squares algorithm (HALS) with resolution of Nonnegative Least Squares problem (NNLS) [1] for factorizations subject to the minimization of the Euclidean/Frobenius norm, or the Multiplicative Update [2,3] for factors, by minimizing the $\beta$-divergence [3]..

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

### MU
This toolbox contains a MU resolution algorithm, developed as described in [3] for NMF, also extended for tensorial factorizations. This code is based on an internship of Florian VOORWINDEN and has been improved with COHEN Jeremy and Valentin LEPLAT.

This toolbox also contains 4 factorization methods:
### NMF
Nonnegative Matrix Factorization [2] - Factorization of a nonnegative matrix X in two nonnegative matrices W and H, where WH approach X.

In the hals condition, this is solved by minimizing the Frobenius norm between both matrices X and WH by NNLS.

In the mu condition, this is solved by minimizing the $\beta$-divergence [3] between both matrices X and WH by Multiplicative Updates.

### NTF - Nonnegative PARAFAC
Nonnegative Tensor Factorization, also called Nonnegative PARAFAC decomposition. PARAFAC decomposition consists in factorizing a tensor T in a sum of rank-one tensors [4]. By concatenating the vectors along each mode of this sum, we obtain as much factors as the number of modes of the tensor [5]. This algorithm returns these factors.

In the hals condition, this factorization is computed as an ALS algorithm, described in [6], solved with NNLS, and using the toolbox Tensorly [7]. It returns the nonnegative factors of a nonnegative tensor T.

In the mu condition, this factorization is computed as NMF subproblems, described in [3], solved with Multiplicative Update, and using the toolbox Tensorly [7]. It returns the nonnegative factors of a nonnegative tensor T subject to the minimization of the $\beta$-divergence.

### Nonnegative PARAFAC2
Nonnegative Tensor Factorization admitting variability over a factor [8]. More precisely, this implemented version is based on a flexible coupling approach [9], where the coupling is enforced by a penalty term.

### NTD - Nonnegative Tucker Decomposition
Nonnegative Tucker Decomposition, which consists in factorizing a tensor T in factors (one per mode) and a core tensor, generally of smaller dimensions than T, which links linearly all factors [6]. This algorithm returns these factors and this core tensor.

In the hals condition, this factorization is computed as an ALS algorithm, described in [6], solved with NNLS, and using the toolbox Tensorly [7]. It also uses a gradient update rule for the core.

In the mu condition, this factorization is computed as NMF subproblems (computationally optimized with tensor contractions), described in [3], solved with the Multiplicative Update [3], and using the toolbox Tensorly [7].

## How to cite ##

You should cite the package `nn_fac`, available on HAL (https://hal.archives-ouvertes.fr/hal-02915456).

Here are two styles of citations:

As a bibtex format, this should be cited as: @softwareversion{marmoret2020nn_fac, title={nn\_fac: Nonnegative Factorization techniques toolbox}, author={Marmoret, Axel and Cohen, J{\'e}r{\'e}my}, URL={https://gitlab.inria.fr/amarmore/nonnegative-factorization}, LICENSE = {BSD 3-Clause ''New'' or ''Revised'' License}, year={2020}}

In the IEEE style, this should be cited as: A. Marmoret, and J.E. Cohen, "nn_fac: Nonnegative Factorization techniques toolbox," 2020, url: https://gitlab.inria.fr/amarmore/nonnegative-factorization.

## References
[1] N. Gillis and F. Glineur, "Accelerated Multiplicative Updates and Hierarchical ALS Algorithms for Nonnegative Matrix Factorization," Neural Computation 24 (4): 1085-1105, 2012.

[2] D.D. Lee and H.S. Seung, "Learning the parts of objects by non-negative matrix factorization," Nature, vol. 401, no. 6755, p. 788, 1999.

[3] C. Févotte, and J. Idier, "Algorithms for nonnegative matrix factorization with the β-divergence", Neural computation, 23(9), pp.2421-2456, 2011.

[4] R. A Harshman et al. "Foundations of the PARAFAC procedure: Models and conditions for an" explanatory" multimodal factor analysis," 1970.

[5] J.E. Cohen, and N. Gillis, "Dictionary-based tensor canonical polyadic decomposition," IEEE Transactions on Signal Processing, 66(7), pp.1876-1889, 2017.

[6] T.G. Kolda, and B.W. Bader, "Tensor decompositions and applications," SIAM review, 51(3), 455-500, 2009.

[7] J. Kossaifi, Y. Panagakis, A. Anandkumar and M. Pantic, "TensorLy: Tensor Learning in Python," Journal of Machine Learning Research (JMLR), volume 20, number 26, 2019.

[8] R.A. Harshman, "PARAFAC2: Mathematical and technical notes," UCLA working papers in phonetics 22.3044, 1972.

[9] J.E. Cohen and R. Bro, "Nonnegative PARAFAC2: a flexible coupling approach," International Conference on Latent Variable Analysis and Signal Separation. Springer. 2018.
