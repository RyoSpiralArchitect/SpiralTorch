"""Check the installed native FFT against a small independent DFT, without NumPy."""

import cmath
import inspect
import json
import math

import spiraltorch as st


def main():
    assert getattr(st, "_rs", None) is not None, "FFT smoke requires the native wheel"
    assert inspect.isbuiltin(st.frac.fft_complex32)
    assert st.frac.fft_real([1, 2, 3, 4]) == [(10, 0), (-2, 2), (-2, 0), (-2, -2)]
    maximum_error = 0.0
    for n in [1, 2, 4, 8, 16, 32]:
        values = [
            complex(((j * 13) % 31 - 15) / 16, ((j * 7) % 19 - 9) / 8) for j in range(n)
        ]
        inputs = [(value.real, value.imag) for value in values]
        for inverse in [False, True]:
            sign = 1 if inverse else -1
            scale = 1 / n if inverse else 1
            expected = [
                scale
                * sum(
                    value * cmath.exp(sign * 2j * math.pi * j * k / n)
                    for j, value in enumerate(values)
                )
                for k in range(n)
            ]
            actual = st.frac.fft_complex32(inputs, inverse=inverse)
            error = max(
                abs(complex(*value) - target) for value, target in zip(actual, expected)
            )
            maximum_error = max(maximum_error, error)
            assert len(actual) == n and error < 3e-6 * n, (n, inverse, error)
        transformed = st.frac.fft_complex32(inputs)
        restored = st.frac.fft_complex32(transformed, inverse=True)
        assert (
            max(
                abs(complex(*value) - target) for value, target in zip(restored, values)
            )
            < 5e-6
        )
    print(
        json.dumps(
            {
                "native_fft": "passed",
                "version": st.__version__,
                "max_dft_error": maximum_error,
                "semantic_owner": "st-frac::fft",
            }
        )
    )


if __name__ == "__main__":
    main()
