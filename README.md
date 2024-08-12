[![HitCount](https://hits.dwyl.com/CharlesPlusC/ERP_tools.svg?style=flat-square&show=unique)](http://hits.dwyl.com/CharlesPlusC/PODDensity)

<p align="center">
  <img src="misc/UCL-logo-black.jpg" alt="University Logo" width="200"><br/>
  <img src="misc/SGNL_logo_ColouronBlack.jpg" alt="Research Group Logo" width="200">
</p>

<h3 align="center">POD Density Inversion</h3>

<p align="center">
    This repo contains code accompanying the paper "Real-Time Thermospheric Density Retrieval from Low Earth Orbit Spacecraft Ephemerides During Geomagnetic Storms" by Charles Constant, Santosh Bhattarai, Indigo Brownhall, Anasuya Aruliah and Marek Zeibart (2024).
  <br />
  <a href="https://github.com/CharlesPlusC/PODDensity/issues">Report Bug</a>
  ·
  <a href="https://github.com/CharlesPlusC/PODDensity/pulls">Request Feature</a>
</p>

- The repo contains tools to pull SP3 orboits from GFZ potsdam FTP, merge the files into continuous interial (J2000/EME2000) ephemerides and perform density inversion on these.
- The underlying library used for many of the computations (force modelling/frame transformations/ density model calls) is the Orekit Python Wrapper, which is a Python wrapper for the Orekit Java library.
- The repo comes in 2 branches. The main branch contains the code, the estimated densities for CHAMP, TerraSAR-X and GRACE-A (in output/DensityInversion/PODDensityInversion/Data/StormAnalysis)and the code to reproduce the results in the paper. The `lite' branch is identical but does not contain the SWindices folder (~2Gb of data) which are used to plots containing space weather indices in the paper.