# Log

In-flight state only: open questions and handoff notes.

## 2026-09-30: DE-MCzs sampler

- NUTS(0.99) underestimates the upper tail when the support edge is near the data. Stationary GEV, ξ = 0.3, n = 30: 95% bound on the 100-year level 11.19 to 11.23 from NUTS, 11.30 from grid quadrature, 11.29 to 11.31 from DE-MCzs. Labs 4 and 5 sample with NUTS(0.99). Lab 5 on Los Angeles (9410660) gets stuck chains at ξ ≈ −1.9 from Turing's default starting points. Lab 5 is committed and stays unchanged; open whether later labs switch to `:demczs`.
