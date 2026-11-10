//! Two-part log weights retain an ordinary suffix log-odds adjustment beside
//! a huge logit cost. Only round their difference when taking an exponential.
#[derive(Clone, Copy)]
pub(super) struct Term(f64, f64);

fn two_sum(a: f64, b: f64) -> Term {
    let s = a + b;
    let v = s - a;
    Term(s, (a - (s - v)) + (b - v))
}

impl Term {
    pub const ZERO: Self = Self(0., 0.);

    pub fn times(self, rhs: Self) -> Self {
        let hi = self.0 * rhs.0;
        let lo = self.0.mul_add(rhs.0, -hi) + (self.0 * rhs.1 + self.1 * rhs.0);
        two_sum(hi, lo + self.1 * rhs.1)
    }

    pub fn exp(self) -> Self {
        if self.0 < -750. {
            return Self::ZERO;
        }
        let base = if self.0.abs() < 0.5 {
            two_sum(1., self.0.exp_m1())
        } else {
            Self(self.0.exp(), 0.)
        };
        base.plus(base.times(Self(self.1.exp_m1(), 0.)))
    }

    pub fn logit(value: f32) -> Self {
        Self(f64::from(value), 0.)
    }

    pub fn rounded(self) -> f64 {
        self.0
    }

    pub fn plus(self, other: Self) -> Self {
        let Term(s, e) = two_sum(self.0, other.0);
        let Term(t, u) = two_sum(self.1, other.1);
        let Term(s, e) = two_sum(s, e + t);
        two_sum(s, e + u)
    }

    pub fn neg(self) -> Self {
        Self(-self.0, -self.1)
    }

    pub fn gap(self, other: Self) -> f64 {
        if self.1 == 0. && other.1 == 0. {
            self.0 - other.0
        } else {
            self.plus(other.neg()).0
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Group {
    peak: Option<Term>,
    tail: f64,
}

impl Group {
    pub fn add(&mut self, term: Term) {
        let Some(peak) = self.peak else {
            self.peak = Some(term);
            return;
        };
        let d = term.gap(peak);
        if d > 0. {
            self.tail = (1. + self.tail) * (-d).exp();
            self.peak = Some(term);
        } else {
            self.tail += d.exp();
        }
    }

    pub fn share(self, term: Term) -> f64 {
        term.gap(self.peak.expect("nonempty partition")).exp() / (1. + self.tail)
    }

    pub fn nll(self, term: Term) -> Term {
        self.peak
            .expect("nonempty partition")
            .plus(term.neg())
            .plus(Term(self.tail.ln_1p(), 0.))
    }

    fn log_ratio(self, other: Self) -> Term {
        self.peak
            .unwrap()
            .plus(other.peak.unwrap().neg())
            .plus(two_sum(self.tail.ln_1p(), -other.tail.ln_1p()))
    }

    pub fn mass_in(self, total: Self) -> f64 {
        self.peak.map_or(0., |_| self.log_ratio(total).0.exp())
    }
}

#[derive(Clone, Copy)]
pub(super) struct Outcome {
    pub yes: Term,
    pub no: Option<Term>,
}

impl Outcome {
    pub const STOP: Self = Self {
        yes: Term::ZERO,
        no: None,
    };

    pub fn new(yes: Group, no: Group) -> Self {
        if no.peak.is_none() {
            return Self::STOP;
        }
        let odds = no.log_ratio(yes);
        let tail = Term(-(-odds.0.abs()).exp().ln_1p(), 0.);
        if odds.0 >= 0. {
            Self {
                yes: odds.neg().plus(tail),
                no: Some(tail),
            }
        } else {
            Self {
                yes: tail,
                no: Some(odds.plus(tail)),
            }
        }
    }
}
