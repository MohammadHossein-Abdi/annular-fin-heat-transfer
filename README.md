# Annular Fin Heat Transfer Analysis

This repository contains numerical solutions for steady and transient heat
transfer in an annular fin with temperature-dependent thermal conductivity
and radiative heat exchange.

## Features
- Radial 1D heat conduction
- Variable thermal conductivity: k(T) = k0 (1 + αT)
- Stefan–Boltzmann radiation
- Steady-state and transient formulations
- Time-dependent base temperature (startup process)

## Files
- part_A_steady_state.py : Steady-state temperature distribution
- part_B_transient.py   : Transient temperature distribution
- validation_constant_k.py : Simplified validation model

## Requirements
```bash
pip install -r requirements.txt
