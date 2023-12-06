#pragma once
#include <common/functional.h>
#include <common/macros.h>
#include <common/types.h>
#include <common/point/internal/pi_gaussian_naive.h>

namespace frisk {

struct ElnetPoint
        : ElnetPointInternal
{
private:
    using state_t = util::control_flow;
    using update_t = util::update_type;

public:
    ElnetPoint(double thr,
                       int maxit,
                       int nx,
                       int& nlp,
                       Eigen::Map<Eigen::VectorXi>& ia,
                       Eigen::VectorXd& y,
                       const Eigen::MatrixXd& X,
                       const Eigen::VectorXd& xv,
                       const Eigen::VectorXd& vp,
                       const Eigen::MatrixXd& cl,
                       const std::vector<bool>& ju);

    template <update_t upd, class PointPackType>
    GLMNETPP_STRONG_INLINE
    void update(int k, const PointPackType& pack)
    {
        double beta_diff = 0;

        auto state = update<upd>(k, pack, beta_diff);
        if (state == state_t::continue_) return;
        
        this->update_resid(k, beta_diff);
    }

    // from ElnetPointGaussianBase
    template <class PointConfigPack>
    void fit(const PointConfigPack& pack)
    {
        this->initialize(pack);

        if (this->is_warm_ever()) {
            partial_fit(pack);
        }

        while (1) {
            bool converged_kkt = this->initial_fit(
                [&]() { return crtp_fit<update_t::full, true>(pack); }
            );
            if (converged_kkt) return;
            partial_fit(pack);
        }
    }

protected:

    template <class PointPackType>
    GLMNETPP_STRONG_INLINE
    void partial_fit(const PointPackType& pack)
    {
        this->set_warm_ever();

        // fit on partial subset
        while (1) {
            bool converged = false, _ = false;
            std::tie(converged, _) = crtp_fit<update_t::partial, false>(pack);
            if (converged) break;
        }
    }

    template <update_t upd, class PointPackType, class DiffType>
    GLMNETPP_STRONG_INLINE
    state_t update(int k, const PointPackType& pack, DiffType&& diff)
    {
        state_t state = crtp_update<upd>(k, pack, diff);
        if (state == state_t::continue_) return state_t::continue_;
        this->update_rsq(k, diff);
        return state_t::noop_;
    }

protected:
    // FROM ElnetPointCRTPBase
    // Generate CRTP self()
    GLMNETPP_GENERATE_CRTP(ElnetPointInternal)

    template <update_t upd, bool do_kkt, class PointConfigPack>
    GLMNETPP_STRONG_INLINE
    std::pair<bool, bool> crtp_fit(const PointConfigPack& pack)
    {
        this->increment_passes();
        this->coord_desc_reset();
        util::if_else<upd == update_t::full>(
            [this, &pack]() {
                this->for_each_with_skip(
                    this->all_begin(),
                    this->all_end(),
                    [=, &pack](auto k) { update<update_t::full>(k, pack); },
                    [=](auto k) { return this->is_excluded(k); }
                );
            },
            [this, &pack]() {
                this->for_each_with_skip(
                    this->active_begin(),
                    this->active_end(),
                    [=, &pack](auto k) { update<update_t::partial>(k, pack); },
                    [](auto) { return false; } // no skip
                );
            });
        this->update_intercept();
        if (this->has_converged()) {
            return util::if_else<do_kkt>(
                [this, &pack]() -> std::pair<bool, bool> { return {true, this->check_kkt(pack)}; },
                []() -> std::pair<bool, bool> { return {true, true}; }
            );
        }
        if (this->has_reached_max_passes()) {
            throw util::maxit_reached_error();
        }
        return {false, false};
    }

    template <update_t upd, class PointPackType, class DiffType>
    GLMNETPP_STRONG_INLINE
    state_t crtp_update(index_t k, const PointPackType& pack, DiffType&& diff)
    {
        diff = this->beta(k);           // save old beta_k
        this->update_beta(k, pack);     // update new beta_k (assumes diff doesn't change)

        if (this->equal(diff, this->beta(k))) return state_t::continue_;

        // update active set stuff if full
        util::if_else<upd == update_t::full>(
            [=]() {
                if (!this->is_active(k)) {
                    this->update_active(k);
                }
            },
            []() {});

        diff = this->beta(k) - diff;    // new minus old beta_k
        this->update_dlx(k, diff);

        return state_t::noop_;
    }
};

} // namespace frisk
