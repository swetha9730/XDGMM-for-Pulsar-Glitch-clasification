# XDGMM-for-Pulsar-Glitch-classification

We carry out a classification of pulsar glitch amplitude using Extreme Deconvolution based Gaussian Mixture Model, where the observed uncertainties in the glitch amplitude are taken into account.

The glitch amplitude data and associated error data is taken from Jodrell Bank (JBO) and ATNF catalogs. The glitches present in ATNF, which were absent in JBO, were appended to the JBO catalog. The glitches having errors present in ATNF and not in JBO were changed.

The glitches with negative amplitude were removed as we perform the XDGMM in logarithmic space. Also, glitches with missing associated errors are removed.
