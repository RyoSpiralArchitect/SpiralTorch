# Excluded Startup

No measurement workers ran. The preflight-report validator compared a JSON
number containing the exact f32 value of `1e-8` with a Python f64 literal.
This was a grid-identity bug in the analysis harness, not a numerical kernel
failure. The archived `analyze-before-f32-grid.py` preserves that failed check.
The corrected validator uses the declared f32 width on both sides. Numerical
tolerances, input conditions and runtime sources were not changed.
