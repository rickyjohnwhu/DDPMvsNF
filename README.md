Repository for the code to measure performance of Neural Spline Flows, Conditional Flow Matching, and Denoising Diffusion Probabilistic Models across Gaussian mixture model and Aib9 torsion angle distribution datasets.

Datasets available at: https://zenodo.org/records/14143082?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjAyYmYzODhlLWE2ZjYtNDA4NS1iNDhlLTJlNzZmMzcyNzMwZCIsImRhdGEiOnt9LCJyYW5kb20iOiI0YTE3NTE3N2Y4MThkODg0YTY4NTI4OWExMGE3NmNmNiJ9.HcFgvUV0sK8EhJm0Ow8cFn-56q8rGuSWj_LBQIcpzMZ_mAySqnJ4pJeJubxw_3Dtl2chUoHAGOaxgaRFyZRLWg

Please note: code and datasets are currently provided for completeness. A user-friendly benchmarking pipeline with both code from this repository and datasets from Zenodo is in development. The next iteration will make filepaths general, as well as migrate from the specific slurm commands the numerical experiments used.

Current instructions:

1. Download all files from Zenodo. Place inside the DDPMvsNF directory.
2. Amend filepaths to match user directories.
3. Launch numerical experiments from .py files with 'wf' prefixes. Results should appear in the data_output folder and analysis can be performed with data_analysis.ipynb.
