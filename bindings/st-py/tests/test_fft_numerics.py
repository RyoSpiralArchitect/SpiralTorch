"""Exercise the shared native FFT against an independent complex DFT."""

import cmath
import math

import pytest
import spiraltorch as st


pytestmark = pytest.mark.skipif(
    getattr(st, "_rs", None) is None,
    reason="FFT regression requires the native extension",
)


def signal(n):
    return [
        complex(((j * 13) % 31 - 15) / 16, ((j * 7) % 19 - 9) / 8) for j in range(n)
    ]


def tuples(values):
    return [(value.real, value.imag) for value in values]


def dft(values, inverse=False):
    sign = 1 if inverse else -1
    scale = 1 / len(values) if inverse else 1
    return [
        scale
        * sum(
            value * cmath.exp(sign * 2j * math.pi * j * k / len(values))
            for j, value in enumerate(values)
        )
        for k in range(len(values))
    ]


def assert_close(result, expected, tolerance):
    assert len(result) == len(expected)
    for index, (actual, target) in enumerate(zip(result, expected)):
        assert abs(complex(*actual) - target) <= tolerance, (index, actual, target)


@pytest.mark.parametrize("scope", ["root", "frac"])
@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("inverse", [False, True])
def test_complex_fft_matches_independent_dft(scope, n, inverse):
    api = st if scope == "root" else st.frac
    values = signal(n)
    inputs = tuples(values)
    result = api.fft_complex32(inputs, inverse=inverse)
    assert_close(result, dft(values, inverse), 3e-6 * n)
    assert inputs == tuples(values)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("n", [1, 4, 8, 16, 64])
def test_real_fft_uses_the_same_transform(inverse, n):
    values = [value.real for value in signal(n)]
    expected = dft(values, inverse)
    assert_close(st.frac.fft_real(values, inverse=inverse), expected, 3e-6 * n)
    assert st.fft_real(values, inverse=inverse) == st.frac.fft_real(
        values, inverse=inverse
    )


def test_ramp_has_exact_forward_frequency_order():
    assert st.frac.fft_real([1, 2, 3, 4]) == [(10, 0), (-2, 2), (-2, 0), (-2, -2)]


def test_radix4_matches_weighted_four_point_forward_dft():
    values = signal(4)
    weights = [1j, -0.5 + 0.25j, 0.75 - 0.125j]
    expected = dft(
        [values[0], *(value * weight for value, weight in zip(values[1:], weights))]
    )
    assert_close(st.frac.fft_radix4(tuples(values), tuples(weights)), expected, 5e-7)
    assert_close(st.fft_radix4(tuples(values), tuples(weights)), expected, 5e-7)


@pytest.mark.parametrize("n", [4, 8, 16, 32, 128, 1024])
def test_complex_roundtrip_and_frequency_domain_filtering(n):
    values = signal(n)
    spectrum = st.frac.fft_complex32(tuples(values))
    assert_close(st.frac.fft_complex32(spectrum, inverse=True), values, 5e-6)
    kernel = [0.5, 0.25j, -0.125] + [0] * (n - 3)
    response = st.frac.fft_complex32(tuples([complex(x) for x in kernel]))
    filtered = [complex(*x) * complex(*y) for x, y in zip(spectrum, response)]
    actual = st.frac.fft_complex32(tuples(filtered), inverse=True)
    expected = [
        sum(values[(k - j) % n] * weight for j, weight in enumerate(kernel[:3]))
        for k in range(n)
    ]
    assert_close(actual, expected, 5e-6)


@pytest.mark.parametrize("n", [0, 3, 5, 6, 15])
def test_invalid_lengths_fail_without_mutating_inputs(n):
    values = tuples(signal(n))
    before = list(values)
    for inverse in [False, True]:
        with pytest.raises(ValueError, match="empty|power of two"):
            st.frac.fft_complex32(values, inverse=inverse)
        assert values == before
