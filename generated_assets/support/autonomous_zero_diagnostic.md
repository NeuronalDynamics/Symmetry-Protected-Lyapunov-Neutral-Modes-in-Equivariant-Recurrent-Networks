# Autonomous-Flow Zero-Exponent Diagnostic

Continuous-time autonomous flows can carry a zero Lyapunov exponent in the flow direction f(x).
This diagnostic separates that direction from the analytical group-tangent bundle E^G_x.

| model | q | rank(E^G) | rank([f,E^G]) | flow defined | angle flow-to-E^G (deg) | independent group directions | note |
|---|---:|---:|---:|---:|---:|---:|---|
| S1Attractor | 1 | 1 | 1 | False | undefined (f=0) | 1 | Attracting circle consists of equilibria, so f(x)=0 while the group tangent is nonzero. |
| T2Attractor | 2 | 2 | 2 | False | undefined (f=0) | 2 | Attracting torus consists of equilibria with two independent group tangents. |
| S1CoupledIrrepAttractor | 1 | 1 | 1 | False | undefined (f=0) | 1 | Non-radial equivariant RNN-style orbit with a slaved second harmonic and hidden invariant rates. |
| SONSphereAttractor_n3 | 2 | 2 | 2 | False | undefined (f=0) | 2 | SO(3)/SO(2) sphere example with nontrivial stabilizer. |
| SONSphereAttractor_n5 | 4 | 4 | 4 | False | undefined (f=0) | 4 | Higher-dimensional SO(n)/SO(n-1) sphere example. |
| UMSphereAttractor_m3 | 5 | 5 | 5 | False | undefined (f=0) | 5 | U(m)/U(m-1) complex-sphere example represented in real coordinates. |
| PhaseIntegrator_constant_velocity | 1 | 1 | 1 | True | 0 | 0 | Relative-equilibrium control: the autonomous flow direction coincides with the group tangent. |
| CollapseCounterexample | 0 | 1 | 2 | True | 90 | 1 | Exact equivariance control where the persistent nondegenerate orbit assumption fails. |

Interpretation: fixed-point continuous attractors have f(x)=0 on the orbit, so their group tangents are not inferred from an ordinary flow direction.
The constant-velocity phase-integrator row is a relative-equilibrium control where the flow is tangent to the S1 orbit, illustrating the caveat that one group direction may coincide with time translation.
