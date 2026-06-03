from __future__ import annotations

from typing import Dict, List, Optional

import torch

from model import HydroAIBasin as BaseHydroAIBasin


def _detach_aux_copy(aux: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    detached: Dict[str, torch.Tensor] = {}
    for key, value in aux.items():
        if torch.is_tensor(value):
            detached[key] = value.detach()
        else:
            detached[key] = value
    return detached


class HydroAIBasinModel(BaseHydroAIBasin):
    """Block-aware stateful basin forward for per-basin H5 storage."""

    def forward(
        self,
        x_dyn_flat: Optional[torch.Tensor],
        x_stat_flat: Optional[torch.Tensor],
        basin_meta: Optional[List[Dict]] = None,
        max_lag: int = 60,
        generator_chunk_size: Optional[int] = None,
        clear_cache: bool = False,
        use_checkpoint: bool = False,
        is_basin: bool = True,
        retain_runoff_grad_for_loss: bool = True,
        retain_route_grad_for_loss: bool = True,
    ):
        if not is_basin:
            return super().forward(
                x_dyn_flat=x_dyn_flat,
                x_stat_flat=x_stat_flat,
                basin_meta=basin_meta,
                max_lag=max_lag,
                generator_chunk_size=generator_chunk_size,
                clear_cache=clear_cache,
                use_checkpoint=use_checkpoint,
                is_basin=is_basin,
            )

        if basin_meta is None:
            raise ValueError("basin_meta is required when is_basin=True.")

        model_device = next(self.parameters()).device
        basin_runoff_raw: List[torch.Tensor] = []
        basin_runoff_mm: List[torch.Tensor] = []
        basin_runoff_m3s: List[torch.Tensor] = []
        basin_q_pred: List[torch.Tensor] = []
        basin_ids: List[str] = []
        basin_route_aux: List[Dict[str, torch.Tensor]] = []
        basin_runoff_m3s_for_loss: List[torch.Tensor] = []
        basin_route_aux_for_loss: List[Dict[str, torch.Tensor]] = []

        for meta in basin_meta:
            x_dyn_base = meta["x_dyn_base"]
            x_stat_base = meta["x_stat_base"]
            target_idx = meta["target_idx"].to(model_device, non_blocking=True).long()
            runoff_basin_full = self.generate_runoff_sequence(
                x_dyn=x_dyn_base,
                x_stat=x_stat_base,
                generator_chunk_size=generator_chunk_size,
                clear_cache=False,
                use_checkpoint=use_checkpoint,
            )
            runoff_basin_raw = runoff_basin_full.index_select(dim=1, index=target_idx)
            pixel_static = x_stat_base.to(model_device, non_blocking=True)

            dist_m = meta["dist_m"].to(model_device, non_blocking=True)
            area_m2 = meta["area_m2"].to(model_device, non_blocking=True)
            fraction = meta["fraction"].to(model_device, non_blocking=True)

            runoff_m3s = self.mmday_to_m3s(runoff_basin_raw, area_m2)
            runoff_m3s_scaled = self.apply_fraction_coefficient(runoff_m3s, fraction)
            runoff_mm_scaled = self.apply_fraction_coefficient(runoff_basin_raw, fraction)
            q_pred, route_aux = self.routing(
                runoff_m3s=runoff_m3s_scaled,
                dist_map_m=dist_m,
                pixel_static=pixel_static,
                max_lag=max_lag,
            )

            basin_ids.append(meta["sample_id"] if "sample_id" in meta else meta["basin_id"])
            basin_q_pred.append(q_pred)
            if self.training:
                if retain_runoff_grad_for_loss:
                    basin_runoff_raw.append(runoff_basin_raw)
                else:
                    basin_runoff_raw.append(runoff_basin_raw.detach())

                if retain_route_grad_for_loss:
                    basin_runoff_m3s_for_loss.append(runoff_m3s_scaled)
                    basin_route_aux_for_loss.append(route_aux)
            else:
                basin_runoff_raw.append(runoff_basin_raw)
                basin_runoff_m3s_for_loss.append(runoff_m3s_scaled)
                basin_route_aux_for_loss.append(route_aux)

            if self.training:
                basin_runoff_mm.append(runoff_mm_scaled.detach())
                basin_runoff_m3s.append(runoff_m3s_scaled.detach())
                basin_route_aux.append(_detach_aux_copy(route_aux))
            else:
                basin_runoff_mm.append(runoff_mm_scaled)
                basin_runoff_m3s.append(runoff_m3s_scaled)
                basin_route_aux.append(route_aux)

            del x_dyn_base, x_stat_base, target_idx, runoff_basin_full
            del runoff_m3s, runoff_mm_scaled
            del pixel_static, dist_m, area_m2, fraction
            if clear_cache and torch.cuda.is_available():
                torch.cuda.empty_cache()

        outputs = {
            "basin_ids": basin_ids,
            "runoff": basin_runoff_raw,
            "runoff_mm": basin_runoff_mm,
            "runoff_m3s": basin_runoff_m3s,
            "q_pred": basin_q_pred,
            "route_aux": basin_route_aux,
        }
        if self.training:
            outputs["runoff_m3s_for_loss"] = basin_runoff_m3s_for_loss
            outputs["route_aux_for_loss"] = basin_route_aux_for_loss
        return outputs
