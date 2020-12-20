# Monte-Carlo-Gamma-Ray-Spectroscopy-Simulator

Work in progress.

My attempt at gamma-ray spectroscopy simulator using the Monte Carlo method. My plan for this project is for it to be able to simulate the detection of a radioactive point-source over a user-specified time interval.
For now the project is done in Python. If I walk into issues regarding computing time, I'll switch to a C++.

The main purpose of this project is for me to learn the Monte Carlo method and how it is applied in the field of medical physics. The reference for this project is Vassilev - Monte Carlo Methods For Radiation Therapy, 2017.

Variables that I intend for the user to change:
- Radioactive element or user inputted energies + their probabilities
- The activity of the source
- Basic detector geometry options
- Detector resolution
- Type of detector (probably just switch between HPGe and NaI)
