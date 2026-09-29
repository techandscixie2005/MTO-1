# Matched scratch native versus MTO direct-f validation comparison

Both arms used 100 epochs and the same fresh core, train order and normalized raw-f objective.

| Arm | Selected epoch | Validation raw-f R2 | Train raw-f R2 selected | Final100 validation R2 |
|---|---:|---:|---:|---:|
| native | 21 | 0.343782 | 0.584646 | 0.227522 |
| mto | 16 | 0.373493 | 0.517471 | 0.276909 |

Contextual eta0 R2: 0.405294; fixed equal-three R2: 0.449421.
Native minus MTO selected validation R2: -0.029711.
Molecule bootstrap 95% CI: [-0.05285466961222263, -0.007184501261827185]; positive fraction 0.004.
Connectivity-group bootstrap 95% CI: [-0.0549249420005203, -0.006530600809318478]; 6671 groups.
This is exploratory validation evidence; no test evaluation or promotion is implied.
