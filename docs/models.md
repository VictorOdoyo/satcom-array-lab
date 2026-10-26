# Models and signs
WGS84 geodesy uses a=6378137 m, inverse flattening=298.257223563, mu=3.986004418e14 m³/s² and Earth angular velocity=7.292115e-5 rad/s. ECI and ECEF are related by a z rotation. Fixed-frame velocity includes minus omega cross r. GMST is anchored at the configured epoch using the USNO approximation. UTC substitutes for UT1 and TT; no Earth orientation data are loaded.

The line of sight is satellite minus station. ENU has east, north, up components. Range rate is the projection of fixed-frame satellite velocity onto this line. Positive recession gives negative Doppler: fD=-range_rate*carrier/c. Near zenith, azimuth is flagged undefined; beam control uses east/north direction cosines.

The two-body propagator solves universal Kepler anomaly with stable Stumpff functions and a bracketed root. RK4 provides an independent comparison and optional J2. J2 energy uses its matching potential; only axial angular momentum is conserved in the oblate field.

The receive manifold is exp(+j k positions dot direction). Output is wᴴ a. Signal power is |wᴴa|², thermal output noise is wᴴw, and SNR gain is their ratio. Failure masks therefore reduce broadside receive gain to the number of active elements. Spatial interference adds INR*aI*aIᴴ to identity noise. The element power pattern multiplies the array gain.

Pr=EIRP+Gr-FSPL-loss in dBW; N0=10log10(kB*Tsys) in dBW/Hz; C/N0=Pr-N0. Es/N0 subtracts 10log10(symbol rate), while Eb/N0 uses information bit rate. Physical noise density is never clamped by a plotting floor.

Residual carrier coherence loss is sinc(df*T)² with T=1 ms. QPSK symbols have unit energy and independent real/imaginary Gaussian noise variance 1/(2 Es/N0). Its uncoded BER is Q(sqrt(Es/N0)).
