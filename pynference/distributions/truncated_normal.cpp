#include <cmath>
#include <torch/extension.h>


// 1 / sqrt(2*pi)
#define M_1_SQRT_2PI 0.398942280401432677939946059934

// log(sqrt(2*pi))
#define M_LN_SQRT_2PI 0.918938533204672741780329736406

const double log_t1 = std::log(0.15);
const double log_t2 = std::log(2.18);
const double t3 = 0.725;
const double t4 = 0.45;


// TODO: Use torch accessor with torch::Scalar as argument types?
// TODO: inline stuff


torch::Tensor log_dnorm(torch::Tensor x) {
    return -(M_LN_SQRT_2PI + 0.5 * x * x);
}


// Exponential rejection sampling (a, inf).
torch::Tensor ers_a_inf(torch::Tensor a) {
    torch::Tensor x, rho;

    do {
        x = -torch::log(torch::rand({1})) / a + a;
        rho = torch::exp(-0.5 * (x - a) * (x - a));
    } while ((torch::rand({1}) > rho).item<bool>());

    return x;
}

// Exponential rejection sampling (a, b).
torch::Tensor ers_a_b(torch::Tensor a, torch::Tensor b) {
    torch::Tensor x, rho;

    do {
        x = -torch::log(torch::rand({1})) / a + a;
        rho = torch::exp(-0.5 * (x - a) * (x - a));
    } while ((torch::rand({1}) > rho).__or__(x > b).item<bool>());

    return x;
}

// Normal rejection sampling (a, b).
torch::Tensor nrs_a_b(torch::Tensor a, torch::Tensor b) {
    auto x = randn_like(a);

    while ((x < a).__or__(x > b).item<bool>()) {
        x.normal_(0.0, 1.0);
    }

    return x;
}

// Normal rejection sampling (a, inf).
torch::Tensor nrs_a_inf(torch::Tensor a) {
    auto x = randn_like(a);

    while ((x < a).item<bool>()) {
        x.normal_(0.0, 1.0);
    }

    return x;
}

// Half-normal rejection sampling.
torch::Tensor hnrs_a_b(torch::Tensor a, torch::Tensor b) {
    auto x = a - 1.0;

    while ((x < a).__or__(x > b).item<bool>()) {
        x.normal_(0.0, 1.0).abs_();
    }

    return x;
}


// Uniform rejection sampling.
torch::Tensor urs_a_b(torch::Tensor a, torch::Tensor b) {
    const auto log_phi_a = log_dnorm(a);
    const auto log_ub = ((a < 0).__and__(b > 0)).item<bool>() ? torch::tensor({-M_LN_SQRT_2PI}) : log_phi_a;
    auto x = torch::empty_like(a);

    do {
        x.uniform_();
    } while ((torch::log(torch::rand({1})) + log_ub > log_dnorm(x)).item<bool>());

    return x;
}

// Previously, this was referred to as type 1 sampling.
torch::Tensor r_lefttruncnorm(torch::Tensor a, torch::Tensor mean, torch::Tensor sd) {
    auto alpha = (a - mean) / sd;

    if ((alpha < t4).item<bool>()) {
        return mean + sd * nrs_a_inf(alpha);
    } else {
        return mean + sd * ers_a_inf(alpha);
    }
}


torch::Tensor r_righttruncnorm(torch::Tensor b, torch::Tensor mean, torch::Tensor sd) {
    auto beta = (b - mean) / sd;

    // Exploit symmetry.
    return mean - sd * r_lefttruncnorm(-beta, torch::zeros_like(beta), torch::ones_like(beta));
}

torch::Tensor r_truncnorm(torch::Tensor a, torch::Tensor b, torch::Tensor mean, torch::Tensor sd) {
    auto alpha = (a - mean) / sd;
    auto beta = (b - mean) / sd;
    auto log_phi_a = log_dnorm(alpha);
    auto log_phi_b = log_dnorm(beta);

    if ((alpha <= 0).__and__(0 <= beta).item<bool>()) {  // 2
        if ((log_phi_a <= log_t1).__or__(log_phi_b <= log_t1).item<bool>()) {  // 2 (a)
            return mean + sd * nrs_a_b(alpha, beta);
        } else {  // 2 (b)
            return mean + sd * urs_a_b(alpha, beta);
        }
    } else if ((alpha > 0).item<bool>()) {  // 3
        if ((log_phi_a - log_phi_b <= log_t2).item<bool>()) {  // 3 (a)
            return mean + sd * urs_a_b(alpha, beta);
        } else {
            if ((alpha < t3).item<bool>()) {  // 3 (b)
                return mean + sd * hnrs_a_b(alpha, beta);
            } else {  // 3 (c)
                return mean + sd * ers_a_b(alpha, beta);
            }
        }
    } else {  // 3s
        if ((log_phi_b - log_phi_a <= log_t2).item<bool>()) {  // 3s (a)
            return mean - sd * urs_a_b(-beta, -alpha);
        } else {
            if ((beta > -t3).item<bool>()) {  // 3s (b)
                return mean - sd * hnrs_a_b(-beta, -alpha);
            } else {  // 3s (c)
                return mean - sd * ers_a_b(-beta, -alpha);
            }
        }
    }
}


torch::Tensor Phi(torch::Tensor x) {
    return 0.5 * (1.0 + torch::erf(x * M_SQRT1_2));
}


torch::Tensor PhiInv(torch::Tensor q) {
    return torch::erfinv(2.0 * q - 1.0) * M_SQRT2;
}


torch::Tensor sample_truncated_normal(
    torch::Tensor loc,
    torch::Tensor scale,
    torch::Tensor a,
    torch::Tensor b) {
    auto ret = torch::zeros_like(loc);
    const int n = loc.size(0);

    for (int i = 0; i < n; ++i) {
        auto cmean = loc[i];
        auto csd = scale[i];
        auto ca = a[i];
        auto cb = b[i];

        auto ca_finite = torch::isfinite(ca).item<bool>();
        auto cb_finite = torch::isfinite(cb).item<bool>();

        if (ca_finite && cb_finite) {
            // Truncation [a,b].
            //auto phi_alpha = Phi((ca - cmean) / csd);
            //auto phi_beta = Phi((cb - cmean) / csd);
            //auto u = torch::rand_like(cmean);
            //ret[i] = cmean + csd * PhiInv((phi_alpha + u * (phi_beta - phi_alpha)).clamp(0.0, 1.0));
            //ret[i] = r_truncnorm(ca, cb, cmean, csd).item();
            ret[i] = 1.0;
        } else if (cb_finite) {
            // Truncation (-inf,b].
            //auto phi_beta = Phi((cb - cmean) / csd);
            //auto u = torch::rand_like(cmean);
            //ret[i] = cmean + csd * PhiInv((u * phi_beta).clamp(0.0, 1.0));
            //ret[i] = r_righttruncnorm(cb, cmean, csd).item();
            ret[i] = 2.0;
        } else if (ca_finite) {
            // Truncation [a,inf).
            //auto phi_alpha = Phi((ca - cmean) / csd);
            //auto u = torch::rand_like(cmean);
            //ret[i] = cmean + csd * PhiInv((phi_alpha + u * (1.0 - phi_alpha)).clamp(0.0, 1.0));
            //ret[i] = r_lefttruncnorm(ca, cmean, csd).item();
            ret[i] = 3.0;
        } else {
            // Truncation (-inf,inf).
            //ret[i] = cmean + csd * torch::randn_like(cmean);
            ret[i] = 4.0;
        }
    }

    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_truncated_normal", &sample_truncated_normal, "Sample from a truncated normal distribution.");
}
