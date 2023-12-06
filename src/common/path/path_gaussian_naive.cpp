#include <common/path/path_decl.h>
#include <common/path/path_gaussian_base.h>
#include <common/path/path_gaussian_naive.h>
#include <common/point/internal/pi_gaussian_naive.h>

namespace frisk {
void ElnetPath::fit(const ElnetPath::FitPack& pack) const
{
    using value_t = double;
    using int_t = int;

    int& jerr = pack.err_code();

    try {
        auto&& path_config_pack = initialize_path(pack);

        auto&& elnet_point = get_elnet_point(pack, path_config_pack);

        value_t lmda_curr = 0; // this makes the math work out in the point solver

        for (int_t m = 0; m < pack.path_size(); ++m) {

            PointConfigPackBase&& point_config_pack =
                initialize_point(m, lmda_curr, pack, path_config_pack, elnet_point);

            try {
                elnet_point.fit(point_config_pack);
            }
            catch (const util::maxit_reached_error& e) {
                jerr = e.err_code(m);
                return;
            }
            catch (const util::bnorm_maxit_reached_error& e) {
                jerr = e.err_code(m);
                return;
            }
            catch (const util::elnet_error& e) {
                jerr = e.err_code(m);
                break;
            }

            state_t state = process_point_fit(pack, path_config_pack, point_config_pack, elnet_point);

            if (state == state_t::continue_) continue;
            if (state == state_t::break_) break;
        }

        process_path_fit(pack, elnet_point);
    }
    catch (const util::elnet_error& e) {
        jerr = e.err_code(0);
    }
}

ElnetPoint ElnetPath::get_elnet_point(const FitPack& pack, const PathConfigPackBase&) const
{
    auto& sp = pack.sub_pack;
    auto& ssp = sp.sub_pack;
    return ElnetPoint(
        ssp.thr, ssp.maxit, ssp.nx, ssp.nlp, ssp.ia, pack.y, ssp.x,
        sp.xv, ssp.vp, ssp.cl, ssp.ju);
}
}