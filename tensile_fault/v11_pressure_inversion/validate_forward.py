"""Compare a direct MOOSE run driven by the inverted pressure (run_step_response.py --name fwd
--pressure-knots ...) with the superposition prediction used in the inversion."""
import numpy as np

import invert_pressure as ip

tg, Rs = ip.ramp_response("fwd")                      # per-psi scaling undone below
direct_s = ip.window_means(tg, Rs * ip.DP_RAMP, 0.0)
direct_s -= direct_s[:, [0]]
direct_r = ip.window_means(tg, Rs * ip.DP_RAMP, 0.0, deriv=True) * 1e6
Z = np.load(ip.RES / "inversion_results.npz")
sup_s, sup_r = Z["base/inverted/model_strain"], Z["base/inverted/model_rate"]
rd_s = np.linalg.norm(direct_s - sup_s) / np.linalg.norm(sup_s)
rd_r = np.linalg.norm(direct_r - sup_r) / np.linalg.norm(sup_r)
tgt = ip.TARGETS["inverted"]
print(f"direct vs superposition: strain {rd_s:.3%}, rate {rd_r:.3%}")
print(f"direct run vs target: strain RMSE {ip.rmse(direct_s, tgt['strain']):.5f} "
      f"(superposition {ip.rmse(sup_s, tgt['strain']):.5f})")
np.savez(ip.RES / "forward_validation.npz", direct_strain=direct_s, direct_rate=direct_r,
         super_strain=sup_s, super_rate=sup_r, target_strain=tgt["strain"], target_rate=tgt["rate"],
         rel_diff_strain=rd_s, rel_diff_rate=rd_r,
         rmse_direct=ip.rmse(direct_s, tgt["strain"]), rmse_super=ip.rmse(sup_s, tgt["strain"]))
