# M45 reference repair: committed old-vs-new curve diff

Compared records:

- archived mislabeled N110/y_max30 curve SHA-256:
  `a8b49dad0ad4a3476b92b06d01b4483ad6b7962d8bbb3ad0c4131380f2d62237`;
- replacement N120/y_max40 curve SHA-256:
  `380ac42e785519a2b4b54b6508c6cc800a6b4373a9cb34e2093412c268ad41a1`.

For each station, growth drift is
`abs(new_omega_i_max - old_omega_i_max) / abs(old_omega_i_max)` and eigenvalue
drift is `abs(c_new - c_old) / abs(c_old)` for complex phase speed `c`.

| R | old omega_i,max | new omega_i,max | relative growth drift | relative eigenvalue drift | abs alpha-peak drift |
|---:|---:|---:|---:|---:|---:|
| 300 | 8.529766267072418e-4 | 8.802555034254461e-4 | 3.1980802% | 2.4533214e-4 | 0 |
| 400 | 1.455461691081253e-3 | 1.479838349180550e-3 | 1.6748402% | 2.1259213e-4 | 0 |
| 500 | 1.875376891396957e-3 | 1.898571480159259e-3 | 1.2367961% | 1.9420062e-4 | 0 |
| 700 | 2.429191594417527e-3 | 2.452446515077620e-3 | 0.9573111% | 1.7086491e-4 | 0 |
| 900 | 2.781707513495558e-3 | 2.805454993067861e-3 | 0.8537015% | 1.5176161e-4 | 0 |
| 1100 | 3.030196472926260e-3 | 3.052903736711313e-3 | 0.7493661% | 1.3076509e-4 | 0 |
| 1300 | 3.216508438701321e-3 | 3.236357476518990e-3 | 0.6170989% | 1.0525842e-4 | 0 |
| 1500 | 3.364801476471252e-3 | 3.379932467702541e-3 | 0.4496845% | 7.7532524e-5 | 0 |
| 1700 | 3.485562439696092e-3 | 3.494719473017124e-3 | 0.2627132% | 4.5642959e-5 | 0 |
| 1900 | 3.587381574700785e-3 | 3.590367142629377e-3 | 0.0832242% | 1.7015030e-5 | 0 |
| 2000 | 3.631991485913146e-3 | 3.631983631510647e-3 | 0.0002163% | 1.2373716e-5 | 0 |

Summary across 11 stations:

- maximum relative eigenvalue drift: `2.4533214155625913e-4` at R=300;
- maximum relative growth drift: `3.1980802127614345e-2` (3.1980802%) at R=300;
- median relative growth drift: `7.493660555654941e-3` (0.7493661%);
- maximum absolute growth difference: `2.7278876718204266e-5`;
- maximum absolute c_r difference: `1.8021177799842913e-4`;
- maximum absolute c_i difference: `1.314644661118275e-4`;
- maximum absolute alpha-peak difference: exactly `0.0`.

This is the expected, now correctly labeled N110/y30-to-N120/y40
resolution/domain shift. It is not attributed to the 13R sweep facade: the
replacement curve is the fixed driver's CPU result at the recorded effective
parameters.
