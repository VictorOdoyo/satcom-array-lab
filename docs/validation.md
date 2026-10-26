# Validation and limits
Independent references include quarter-circle and reversible universal-variable orbits; two-body energy and angular momentum; RK4 convergence against the universal anomaly solve; an independent finite-difference gravity-potential gradient; J2 energy and axial momentum; and the approximate first-order secular node rate.

Geodetic tests include poles and inverse coordinate recovery. Frame tests check velocity rotation, inertially stationary and Earth-fixed states. The Doppler bench differentiates actual rotating-station range independently of velocity projection.

Array checks include active-element receive SNR, invariance to common weight scaling, complex constrained responses, colored-noise MVDR and thermal-noise normalization. RF checks invert the link budget, preserve sub-floor physical noise density, match coherent sinc loss to discrete tones and bracket analytic uncoded QPSK BER with Monte Carlo uncertainty.

Integrated tests cover deterministic independent random streams, sensor-based correction versus open loop, outlier rejection, beacon-loss coasting, reacquisition and rain loss. Catalog tests validate integrity and published CSV agreement. CI runs Python 3.11/3.13 on Linux/Windows and exercises a built wheel outside the checkout.

These are synthetic engineering models. There is no TLE/SGP4 operational prediction, live ephemeris, full Earth orientation model, orbit determination, atmospheric availability certification, measured antenna pattern, propagation multipath, flight terminal, hardware RF control or FEC decoder. The synthetic station is at Nairobi coordinates and zero ellipsoidal height, not a surveyed installation.
