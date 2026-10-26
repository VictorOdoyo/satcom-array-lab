# Receiver and scheduling
The true orbit altitude differs from the commanded ephemeris by 10 km. A fixed roll/pitch panel rotation and a carrier offset plus drift add realistic unknown biases. Two constant-velocity Kalman models track east/north direction-cosine offsets and carrier bias using synthesized noisy beacon observations. Covariance scales with elapsed seconds; updates use normalized innovation gating and Joseph covariance form.

Beacon observability requires actual visibility, sufficient C/N0 and residual carrier inside a 6 kHz acquisition range. A declared outage masks observations. The measurement abstraction assumes a beacon receiver can estimate direction and carrier; it does not simulate monopulse hardware or a beam-search waveform. The actual QPSK acquisition bench is separate.

Lock requires two seconds of accepted observations. Invalid measurements may coast for five seconds; stale lock expires before a fresh observation starts reacquisition. A measurement at this timestamp cannot justify service already offered at that timestamp.

ACM uses previous noisy pilot Es/N0, immediate downgrade, a 1 dB upgrade margin and five-second dwell. Modes and thresholds are illustrative. Offered service is symbol rate times modulation bits times code rate times 0.95 framing. Delivered service is a threshold-based physical channel proxy; no FEC decoder is present. Rates are held over each actual time interval, including a shorter final interval.

Handover compares nominal link scores with 2 dB hysteresis, 30-second dwell and five-second candidate persistence. Receivers retain separate filters and reacquire after inactivity. The chosen satellite's raw true/nominal states, commanded weights, carrier, observations, lock and rate decisions are archived.
