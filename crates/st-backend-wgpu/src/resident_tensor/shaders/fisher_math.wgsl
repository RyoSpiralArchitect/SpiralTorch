// |x| <= ln(2)/2. Accumulate the deviation before adding one, so even
// subnormal source-logit perturbations keep their own Wide exponent.
fn fisher_expm1(x:Wide)->Wide {
    var term=parts(1.0);var deviation=wide_sum_zero();
    for(var n=1u;n<=17u;n++) {
        term=wide_div(wide_mul(term,x),parts(f32(n)));
        deviation=wide_sum_add(deviation,term);
    }
    return wide_sum_value(deviation);
}

// exp(x)-1-x for |x|<=0.5. Keep curvature terms separate from the linear
// term until cross-category cancellation has occurred.
fn fisher_exponential_remainder(x:Wide)->Wide {
    var term=x;var total=wide_sum_zero();
    for(var n=2u;n<=17u;n++) {
        term=wide_div(wide_mul(term,x),parts(f32(n)));
        total=wide_sum_add(total,term);
    }
    return wide_sum_value(total);
}
