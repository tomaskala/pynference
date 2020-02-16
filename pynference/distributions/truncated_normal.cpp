#include <cmath>
#include <torch/extension.h>


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
    auto ret = torch::empty_like(loc);
    const int n = loc.size(0);

    for (int i = 0; i < n; ++i) {
        auto cmean = loc[i];
        auto csd = scale[i];
        auto ca = a[i];
        auto cb = b[i];

        auto ca_finite = torch::isfinite(ca).item().to<bool>();
        auto cb_finite = torch::isfinite(cb).item().to<bool>();

        if (ca_finite && cb_finite) {
            // Truncation [a,b].
            auto phi_alpha = Phi((ca - cmean) / csd);
            auto phi_beta = Phi((cb - cmean) / csd);
            auto u = torch::rand_like(cmean);
            ret[i] = cmean + csd * PhiInv((phi_alpha + u * (phi_beta - phi_alpha)).clamp(0.0, 1.0));
        } else if (cb_finite) {
            // Truncation (-inf,b].
            auto phi_beta = Phi((cb - cmean) / csd);
            auto u = torch::rand_like(cmean);
            ret[i] = cmean + csd * PhiInv((u * phi_beta).clamp(0.0, 1.0));
        } else if (ca_finite) {
            // Truncation [a,inf).
            auto phi_alpha = Phi((ca - cmean) / csd);
            auto u = torch::rand_like(cmean);
            ret[i] = cmean + csd * PhiInv((phi_alpha + u * (1.0 - phi_alpha)).clamp(0.0, 1.0));
        } else {
            // Truncation (-inf,inf).
            ret[i] = cmean + csd * torch::randn_like(cmean);
        }
    }

    return ret;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sample_truncated_normal", &sample_truncated_normal, "Sample from a truncated normal distribution.");
}
