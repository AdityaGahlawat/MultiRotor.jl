# Codebase for Multirotor sim and viz 
- Generalized - can be used as a stochatsic system or a deterministic system 

## Setup 
Recommended to launch with multi thread support
```bash
julia -t auto
```
Add codebase to working environment
```julia
] add https://github.com/AdityaGahlawat/MultiRotor.git
```

## Ready-to-use Systems 
- ### GUAM 
    - Run `examples/GUAM/main.jl`
    - System description in `examples/GUAM/dynamics.md`
