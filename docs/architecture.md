# Architecture
The main pass runner couples circular Cartesian ephemerides, Earth rotation, ellipsoidal visibility, a planar receive array, a physical carrier-to-noise budget, noisy beacon observations and sampled receiver policy. The independent orbit benches also support elliptic and moderate unbound two-body states and numerical J2 gravity.

At each time, all filters predict. Handover uses nominal ephemeris scores. The selected receiver commands a beam and carrier from the previous posterior. Current data service uses the previous pilot estimate and current coasting lock. The physical channel scores delivery. Only then do current beacon observations correct filters and refresh lock.

`passmodel.py` owns this order. `tracking.py` has gated covariance updates and lock timing; `service.py` has rate and satellite policy. `array.py` owns physical positions and receive normalization; `link.py` owns RF and noise units. `modem.py` independently simulates uncoded QPSK bursts.

The report layer preserves numeric arrays, masks, configuration, source revision, package versions and hashes. Rendering and replay do not solve again. See [reproduction](reproduction.md) and [model equations](models.md).
