# Oversampled burst receiver
The standalone waveform extension transmits uncoded QPSK through a unit-discrete-energy root-raised-cosine pulse, a fractional-delay FIR, receiver-clock resampling, carrier/phase errors, IQ/DC impairment, AWGN, complex quantization and a matched filter.

Both pulse filters retain their group delay. Known-pilot correlation searches a bounded fractional timing interval. Only the pilot prefix fits carrier frequency, complex gain, conjugate leakage and DC. A bounded frequency search uses variable projection: complex linear IQ/DC coefficients are fitted at each carrier candidate. The calibrated inverse rejects singular receiver axes. Payload symbols are used only for BER/EVM scoring.

Unit-energy pulses imply matched complex noise variance N0, independent of samples per symbol. Periodogram integration is checked against mean sample power. Amplitudes and spectra use normalized digital units, not calibrated watts. Fractional sampling uses bounded cubic interpolation; clock resampling is an offline synthetic impairment.

The receiver applies one fixed burst calibration. Later clock drift and Wiener phase noise can therefore degrade the payload. Search-range failure, ADC clipping and amplifier-induced ISI remain visible. Rapp compression is memoryless AM-AM only; finite-span RRC filters retain residual truncation ISI.

See [MathWorks raised-cosine filtering](https://www.mathworks.com/help/comm/ug/raised-cosine-filtering.html), [IQ impairment context](https://www.mathworks.com/help/comm/ug/compensate-i-q-imbalance.html) and the [computed gallery](results.md).
