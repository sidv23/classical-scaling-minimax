# Minimax Optimality of Classical Scaling Under General Noise Conditions

Code to reproduce the simulations and figures in:

> S. Vishwanath and E. Arias-Castro, "Minimax Optimality of Classical Scaling Under General Noise Conditions," *Information and Inference: A Journal of the IMA*. [arXiv:2502.00947](https://arxiv.org/abs/2502.00947)

See the paper for methodological details, assumptions, and theoretical results. 

> [!NOTE]
> - **Requirements:** Julia 1.12 (tested with 1.12.6)


## Getting Started

1. Clone the repository:
    ```bash
    $ git clone https://github.com/sidv23/classical-scaling-minimax.git
    $ cd classical-scaling-minimax
    ```

2. Activate the Julia environment:
    ```julia
    > julia --project=. -e 'import Pkg; Pkg.instantiate()'
    ```


3. **Run the simulations**:

   ```julia
   julia simulations.jl
   ```

   This samples points on a ball, adds noise to pairwise distances, runs classical scaling, computes the reconstruction error ($L_{\text{rmse}}$ and $L_{2\to\infty}$), and writes results to `results/simulations.jld2`. 
   
   This sweeps a fairly large grid (5 sample sizes × 3 tail conditions × 5 condition numbers × 3 noise scales × 3 noise types × 5 replicates) and may take a while.

4. **Generate the figures**:

   ```julia
   julia make-plots.jl
   ```

   This reads `results/simulations.jld2` and generates the plots `plots/pxx.pdf` corresponding to the $L_{\text{rmse}}$ and $L_{2\to\infty}$ error plots.



## Citation

If you use this code, please consider citing the paper:

```bibtex
@article{vishwanath2026minimax,
  title   = {Minimax Optimality of Classical Scaling Under General Noise Conditions},
  author  = {Vishwanath, Siddharth and Arias-Castro, Ery},
  journal = {Information and Inference: A Journal of the IMA},
  year    = {2026},
  eprint  = {2502.00947},
  archivePrefix = {arXiv}
}
```
