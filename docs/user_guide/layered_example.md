# Two-layer example: what the seismogram should show

Run `seisfem run examples/1d/layered.yaml`, then
`python examples/1d/analyze_layered.py`. Outputs include normalized config,
metadata, per-rank displacement/velocity/acceleration traces, DG0 material fields,
strided displacement snapshots, a velocity SVG and a JSON analytical comparison.
The analysis command can read either serial or MPI shards.

The source is at z=1000 m, interface z=2000 m. Below the interface,
rho1=2000 kg/m³, Vp1=2000 m/s; above it, rho2=2400 kg/m³, Vp2=3000 m/s.
The source peaks at 0.15 s and its strength is 8e6 Pa, giving unit incident
velocity amplitude in medium 1. Impedances are 4e6 and 7.2e6 kg/(m² s).
For displacement/velocity R=-2/7 and T=5/7; these are not traction coefficients.

| Receiver/phase | Travel time | Expected peak including source shift | Velocity amplitude |
|---|---:|---:|---:|
| z=1400 direct | 400/2000=0.2 s | 0.35 s | 1 m/s |
| z=1400 reflected | (1000+600)/2000=0.8 s | 0.95 s | -2/7 m/s |
| z=2600 transmitted | 1000/2000+600/3000=0.7 s | 0.85 s | 5/7 m/s |

These isolated arrival windows support signed amplitude comparisons. The full
analytical overlay assumes infinite layers, so late finite-domain returns are
not included. The endpoint impedance reduces, but does not identically remove,
discrete boundary reflections. Snapshots are every 250 steps, not every step.

For S propagation change `mode` to S and use a finer mesh/longer duration. The
automated S interface test uses 8000 cells, dt=0.0002 s and duration=2.2 s to resolve
the shorter wavelength and later arrivals; merely changing the mode in this
example is not an accuracy guarantee.
