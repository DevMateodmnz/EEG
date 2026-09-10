# EEG fundamentals used in this project

[Back to the project README](../README.md)

This chapter records the EEG concepts needed for milestones completed so far. The executed spectral method and filter-design evidence are developed fully in [Raw frequency-domain characterization and filter design](spectral_analysis.md).

## What EEG measures

Electroencephalography (EEG) records changes in electrical potential at electrodes placed on the scalp. A recorded EEG channel represents a voltage difference involving an electrode and a reference; it is not a direct recording from an individual neuron.

Scalp EEG reflects spatially mixed activity from neural populations together with non-neural contributions and measurement noise. Therefore, this project must not describe the signals as thoughts, direct commands from the brain, or counts of active neurons.

The accessible EDF/MNE metadata has now been inspected. It does not identify a named physical reference electrode, and MNE reports that no custom re-reference has been applied. The raw-visualization milestone therefore preserves the stored signal reference and describes the physical reference as unspecified rather than guessing it.

## Electrodes and channels

An **electrode** is the physical sensor placed at a scalp location. A **channel** is the recorded signal associated with an electrode and its reference.

The selected PhysioNet recordings contain 64 EEG channels. Each channel is a time series: an ordered sequence of measurements taken at regular time intervals.

## Continuous EEG as a matrix

MNE's continuous `Raw` data can be retrieved as an array shaped:

```text
(channels, time samples)
```

Mathematically, one run can be described as:

\[
X \in \mathbb{R}^{C \times N}
\]

where:

- $C$ is the number of channels;
- $N$ is the number of sampled time points;
- $X_{c,n}$ is the voltage value for channel $c$ at sample $n$.

For each inspected run:

\[
C=64, \qquad N=20{,}000
\]

so the observed shape is `(64, 20,000)`. This does not mean there are 20,000 trials. It means that every channel has 20,000 consecutive measurements in one continuous recording.

## Sampling frequency and sample interval

The sampling frequency $f_s$ is the number of measurements recorded per second for each channel. These recordings use:

\[
f_s = 160\ \text{Hz} = 160\ \text{samples per second}
\]

The time interval between adjacent samples is the reciprocal:

\[
\Delta t = \frac{1}{f_s}
\]

Therefore:

\[
\Delta t = \frac{1}{160}\ \text{s} = 0.00625\ \text{s} = 6.25\ \text{ms}
\]

The time assigned to sample index $n$, relative to the start of the recording, is:

\[
t_n = \frac{n}{f_s}
\]

With 20,000 samples at 160 Hz, the amount of recorded data is approximately:

\[
\frac{20{,}000}{160} = 125\ \text{s}
\]

The final sample occurs one sample interval before 125 s because indexing starts at zero. The inspection script therefore labels 125 s as an approximate recording duration.

### Nyquist frequency and aliasing

The Nyquist frequency is half the sampling frequency:

\[
f_{\mathrm{Nyquist}}=\frac{f_s}{2}=80\ \text{Hz}
\]

A sampled signal can represent frequencies below this limit without aliasing, assuming adequate acquisition filtering. Frequencies above Nyquist are ambiguous: for example, samples of a 90 Hz sinusoid at 160 Hz can appear as 70 Hz because $160-90=70$. The completed PSD analysis measures no meaningful 50 Hz peak but does measure a strong, repeatable 60 Hz line. Present-day local electrical frequency was therefore not used as a substitute for inspecting the recorded data.

## Time and frequency domains

A time-domain EEG plot shows signed voltage against time. A frequency-domain description shows how amplitude or power is distributed across rates of repetition. They describe the same signal from different perspectives.

Frequency $f$ and period $T$ are reciprocals:

\[
f=\frac{1}{T}.
\]

A 10 Hz component completes ten cycles per second and has a 0.1-second period. EEG is not assumed to be a set of permanent perfect sine waves. Sine and cosine components are mathematical building blocks that allow a finite sampled signal to be decomposed and summarized.

For samples $x[n]$, the discrete Fourier transform calculates complex frequency coefficients:

\[
X[k]=\sum_{n=0}^{N-1}x[n]\exp\left(-i\frac{2\pi kn}{N}\right).
\]

Coefficient magnitude describes component size and angle describes phase. The FFT is an efficient algorithm for calculating this DFT, not a different physical operation.

For an $N$-sample segment, the Fourier frequency-bin spacing is

\[
\Delta f=\frac{f_s}{N}\approx\frac{1}{T}.
\]

Longer segments improve frequency spacing while reducing temporal localization and the number of segments available for averaging. Zero-padding makes a denser display grid but does not create true additional resolution.

## Spectral power and Welch PSD

Power is proportional to squared Fourier magnitude. Power spectral density (PSD) expresses power per unit frequency. MNE works internally with EEG in volts, so its PSD has units V²/Hz. Multiplying by $(10^6)^2=10^{12}$ converts this to µV²/Hz. Integrating PSD over a band produces power in µV².

The report's logarithmic plots use $10\log_{10}(P/P_0)$ with $P_0=1$ µV²/Hz. A 3.01 dB increase is approximately twice the power. Linear-unit values remain available in CSV tables.

Welch PSD divides a signal into segments, applies a window, calculates segment periodograms, and averages them. Averaging stabilizes the estimate relative to one periodogram, while segment length trades frequency resolution against the number and temporal localization of estimates. A taper such as Hann reduces spectral leakage caused by abruptly cutting finite segments, although no window eliminates the tradeoff completely.

This project explicitly uses 3-second/480-sample Hann segments, 240-sample overlap, mean averaging, segment-mean removal, and `n_fft=480`. That gives 0.333333 Hz bin spacing and 82 segments covering all 124.5 valid seconds without zero-padding.

## EEG bands and interpretation limits

Delta, theta, alpha, beta, and gamma are conventional frequency labels whose exact boundaries vary. They are not universal translations into mental states. This project currently uses 1–4, 4–8, 8–13, 13–30, and 30–45 Hz respectively, describing the last band only as low gamma.

An alpha-range sensorimotor rhythm is often called mu, and both mu- and beta-range activity can be relevant in motor experiments. `C3`, `Cz`, and `C4` sample central scalp locations of interest. A continuous whole-run PSD peak near these channels is descriptive only; motor-imagery evidence requires event-aligned condition comparisons.

## Filter vocabulary

A high-pass attenuates slow frequencies, a low-pass attenuates fast frequencies, a band-pass retains an intermediate range, and a notch attenuates a narrow band. Real filters have passbands, stopbands, and transitions rather than perfectly vertical cutoffs. Filter length/order, FIR versus IIR design, phase response, zero-phase versus causal application, padding, and recording-edge effects are part of the method.

Consequently, “1–40 Hz filtering” is not a reproducible specification by itself. The completed production design uses a 529-sample zero-phase Hamming/`firwin` FIR, 1 and 40 Hz passband edges, 1 and 10 Hz transitions, and `reflect_limited` padding. The direct 60 Hz notch was tested and rejected as redundant after the low-pass. See [Controlled continuous-EEG preprocessing and filter validation](preprocessing.md) for the complete response and temporal consequences.

## Voltage units

The EDF header declares the original EEG channel units as microvolts. MNE converts EEG data to SI units internally, so values returned without explicit scaling are expressed in volts. EEG amplitudes are commonly discussed and plotted in microvolts:

\[
1\ \mu\text{V} = 10^{-6}\ \text{V}
\]

The visualization workflow requests microvolts from MNE and verifies numerically that the values equal the internal volt values multiplied by $10^6$.

## Amplitude and polarity

A sample can be positive or negative because it is a signed voltage difference relative to the reference. Positive does not mean neural activation and negative does not mean neural inhibition.

**Absolute magnitude** ignores polarity: a value of $-100\ \mu\text{V}$ has an absolute magnitude of $100\ \mu\text{V}$. **Oscillation amplitude** describes the size of repeated variation and must be defined carefully, for example as peak amplitude or peak-to-peak amplitude.

A large raw deflection has several possible origins, including neural activity, eye or muscle activity, electrode behavior, movement, and environmental noise. Amplitude alone cannot determine which explanation is correct.

## Signal-quality statistics

The whole-recording mean describes average signed offset. Variance is the average squared distance from that mean, and standard deviation is the square root of variance. Standard deviation is expressed in µV and summarizes variability, while variance is expressed in µV².

Peak-to-peak amplitude is the maximum value minus the minimum value. Because only the two extremes determine it, one transient can make it large. Median, median absolute deviation (MAD), and interquartile range (IQR) are less sensitive to isolated extreme values and provide complementary evidence.

A flat technical channel is constant or almost constant for a sustained interval. Quiet physiological EEG still contains variation. A noisy channel can instead contain excessive or irregular activity caused by poor contact, muscle, movement, cables, or environmental interference. Signal-quality metrics identify evidence worth inspecting; they do not convert a statistical outlier directly into a failed electrode.

The full equations, anomaly heuristic, and executed results are documented in [Quantitative EEG signal quality and referencing decision](signal_quality.md).

## Referencing

EEG voltage is not absolute. If $V_i(t)$ is the potential at electrode $i$ and $V_r(t)$ is the reference potential, a simplified channel value is:

\[
X_i(t)=V_i(t)-V_r(t)
\]

Changing the reference changes every channel because a different time series is subtracted. The selected derived analysis strategy is an average reference:

\[
V_{i,\mathrm{new}}(t)=V_i(t)-\frac{1}{C}\sum_{j=1}^{C}V_j(t)
\]

This makes the mean across included EEG channels zero at each time point. It is an instantaneous spatial transformation, not a frequency-selective temporal filter. A bad channel can contaminate the average, so channel review must precede its final application.

For the original raw figures:

- no re-reference was applied;
- the waveforms remain under the reference stored in the EDF;
- the named physical reference is not available in the metadata inspected so far.

This preserves the raw signal for exploration while making the limitation explicit.

For the controlled comparison, average reference is applied only to an in-memory copy. The 64 broadly distributed scalp channels make it a practical later analysis reference, but finite/incomplete head coverage and the unknown original reference remain limitations. Linked mastoids are unavailable, a single electrode would be arbitrary, and REST would require a forward model not currently available.

## Raw versus derived data

The downloaded EDF is the authoritative raw recording and is never overwritten. Re-referencing, filtering, artifact handling, and epoching create derived representations that must be reproducibly generated from the raw file. The current production-derived continuous representation is regenerated in memory from immutable EDF, the JSON configuration, and reusable preprocessing functions; it is not saved as a second authoritative recording.

## Visible artifact candidates

Raw plots can suggest, but usually cannot prove, artifact types:

- eye blinks often appear as large slow frontal deflections;
- lateral eye movements can create opposite-polarity frontal patterns;
- muscle activity can appear irregular and relatively high-frequency;
- mains interference can appear as persistent narrow-frequency activity;
- electrode problems can produce flat signals, large jumps, or excessive noise;
- movement can produce large widespread transients.

The dataset has no channels labeled as EOG, so ocular interpretations cannot be confirmed from a dedicated eye channel. Artifact removal has not been performed.

## Annotations are not EEG channels

Annotations attach descriptions and time positions to the continuous recording. For this dataset they indicate rest and task onsets. They describe what was happening in the experiment; they are not additional voltage signals and are not part of the `(64, 20,000)` EEG matrix.

Epoching now uses these timing markers to extract event-aligned segments. An annotation interval is converted to a discrete event sample, and a fixed window around that event becomes an epoch. Event-relative `t=0` is the event onset; negative time precedes it and positive time follows it.

The task representation is shaped `(45, 64, 961)`: 45 trials, 64 channels, and 961 samples from `−2…+4 s` at 160 Hz. The endpoint count is inclusive, so it is `(6 s × 160 Hz) + 1`, not 960. See [Event extraction, epoching, and trial-quality assessment](epoching.md) for the complete semantics, timing, baseline, quality, and provenance decisions.

## Event-related spectral power

A time–frequency representation preserves frequency and event-relative time rather than pooling an entire recording. The current Morlet result is shaped `(trials, channels, frequencies, times) = (45, 64, 30, 241)`. ERD/ERS percent change compares each task with its paired preceding rest: negative means lower power and positive means higher power. See [Event-related spectral analysis](event_related_spectral_analysis.md).

## Sources

- MNE-Python contributors. [The `Raw` data structure](https://mne.tools/stable/generated/mne.io.Raw.html): continuous-data representation, returned array shape, and SI units.
- MNE-Python contributors. [Algorithms and implementation details](https://mne.tools/stable/documentation/implementation.html): EEG units and memory-efficient raw-data access.
- MNE-Python contributors. [`mne.set_eeg_reference`](https://mne.tools/stable/generated/mne.set_eeg_reference.html): reference transformations and average-reference behavior.
- MNE-Python contributors. [`mne.events_from_annotations`](https://mne.tools/stable/generated/mne.events_from_annotations.html): conversion from annotation onsets to event samples.
- MNE-Python contributors. [`mne.Epochs`](https://mne.tools/stable/generated/mne.Epochs.html): event-relative data extraction and baseline behavior.
- NIST/SEMATECH. [Measures of scale](https://www.itl.nist.gov/div898/handbook/eda/section3/eda356.htm): variance, standard deviation, MAD, and IQR.
- Schalk, G. (2009). [EEG Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) (version 1.0.0). PhysioNet.
