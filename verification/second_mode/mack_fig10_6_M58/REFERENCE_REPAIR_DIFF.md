# M58 reference repair: committed old-vs-new curve diff

Compared records:

- archived mislabeled N110/y_max30 curve SHA-256:
  `80a9cccaab8631da592bf712139952d295598564887934a28e633a0fe208d1d7`;
- replacement N150/y_max64 curve SHA-256:
  `ec1971fa37d349b6780902b37caaccef131d4c12b41571c4917c89bf31ea0eda`.

For each station, growth drift is
`abs(new_omega_i_max - old_omega_i_max) / abs(old_omega_i_max)` and eigenvalue
drift is `abs(c_new - c_old) / abs(c_old)` for complex phase speed `c`.

| R | old omega_i,max | new omega_i,max | relative growth drift | relative eigenvalue drift | abs alpha-peak drift |
|---:|---:|---:|---:|---:|---:|
| 240 | 8.692116915646328e-04 | 1.186948415190214e-03 | 36.5545847% | 6.9548957e-03 | 0.00125000000000001 |
| 300 | 1.232162023894936e-03 | 1.540535206507255e-03 | 25.0269994% | 6.2895159e-03 | 0.00125 |
| 400 | 1.656333696062568e-03 | 1.951720116237073e-03 | 17.8337506% | 5.4935221e-03 | 0.00125 |
| 500 | 1.953348199144082e-03 | 2.241469046046776e-03 | 14.7501017% | 4.6957457e-03 | 0 |
| 700 | 2.348908221408015e-03 | 2.627500873761061e-03 | 11.8605167% | 4.6647863e-03 | 0.00125 |
| 900 | 2.600845650433065e-03 | 2.874486463255013e-03 | 10.5212246% | 4.4134039e-03 | 0.00125 |
| 1100 | 2.778385064046344e-03 | 3.048234078522728e-03 | 9.7124412% | 4.2078277e-03 | 0.00125 |
| 1300 | 2.910089378373656e-03 | 3.176527431546483e-03 | 9.1556656% | 4.0276532e-03 | 0.00125 |
| 1500 | 3.012972783690730e-03 | 3.278544357803614e-03 | 8.8142706% | 3.9817588e-03 | 0.00125 |
| 1700 | 3.094641227300784e-03 | 3.359480145340575e-03 | 8.5579845% | 3.9436311e-03 | 0.00125 |
| 1900 | 3.163817280649578e-03 | 3.425421926765875e-03 | 8.2686395% | 3.3402694e-03 | 0 |
| 2000 | 3.193991550167661e-03 | 3.455224464920464e-03 | 8.1788856% | 3.7769591e-03 | 0.00125 |

Summary across 12 stations:

- maximum relative eigenvalue drift: `0.006954895734361942` at R=240;
- maximum relative growth drift: `0.36554584655164524` (36.5545847%) at
  R=240;
- median relative growth drift: `0.1011683288725751` (10.1168329%);
- maximum absolute growth difference: `0.0003177367236255812`;
- maximum absolute c_r difference: `0.006001717548695384`;
- maximum absolute c_i difference: `0.002447878072949979`;
- maximum absolute alpha-peak difference: `0.001250000000000015`.

The comparison metric changed from `0.08946123188554544` to
`0.02648169670599881`, an absolute change of `0.06297953517954663`. The
scientific verdict consequently changed from `acceptable` to `agrees`. That
improvement is recorded as the result, never as the motive: the repair was
required because the committed N110/y30 calculation was mislabeled N150/y64.

This is the correctly labeled N110/y30-to-N150/y64 resolution/domain shift. It
is not attributed to the GPU/backend-contract work: the replacement curve is
the fixed verifier's CPU result at the recorded effective parameters.
