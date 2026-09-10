| arm | final_rl2_path | best_rl2_path | steps | steps_per_s | final_loss_interior | t@0.01 | t@0.005 | t@0.002 |
|---|---|---|---|---|---|---|---|---|
| FS-PINN, ENGD-Woodbury   | 0.0015 | 0.0010 | 55141 | 18.4 |  1.15e-06 | 6s | 20s | 165s |
| FS-PINN, Adam            | 0.0087 | 0.0049 | 586750 | 195.6 |  2.44e-06 | 246s | 2399s | never |
| EM-BSDE, Adam            | 0.1417 | 0.1412 | 762069 | 254.0 |  7.42e-04 | never | never | never |
| Heun-BSDE, Adam          | 0.0079 | 0.0067 | 677605 | 225.9 |  1.17e-05 | 212s | never | never |
| Un-EM-BSDE, Adam         | 0.0065 | 0.0061 | 756216 | 252.1 |  2.61e-05 | 232s | never | never |
