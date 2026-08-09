"""
Turbine power curves.

First attempt built curves from a cubic law between cut-in and rated.
Testing that against the real measured V90/2000 curve showed it runs 30
percent low across the ramp, because the cubic law describes the energy
in the wind, not what the machine extracts. Real turbines run near peak
efficiency at low speed and shed efficiency as they approach rated.

So instead we take a real measured curve from a reference turbine, strip
it down to a power coefficient curve, and rescale that onto our turbine's
swept area and rating. Shape comes from measurement, size comes from the
spec sheet.
"""

import re

import numpy as np
import pandas as pd

AIR_DENSITY_STANDARD = 1.225
BETZ_LIMIT = 0.593

REFERENCE_TURBINES = {
    "modern_large": "V112/3000",
    "modern_medium": "V90/2000",
    "legacy_small": "V90/2000",
}

_REFERENCE_INDEX = None


def build_reference_index():
    """Catalogues every library turbine by specific power, so we can pick
    a reference that actually resembles the target instead of always
    using the same one."""
    global _REFERENCE_INDEX
    if _REFERENCE_INDEX is not None:
        return _REFERENCE_INDEX

    from windpowerlib import WindTurbine, get_turbine_types

    rows = []
    for turbine_type in get_turbine_types(print_out=False).turbine_type:
        try:
            wt = WindTurbine(turbine_type=turbine_type, hub_height=80)
            curve = wt.power_curve
            rated_kw = curve.value.max() / 1000
            diameter = float(
                re.sub(r"[^0-9.]", "", str(turbine_type).split("/")[0])
            )
        except Exception:
            continue
        if diameter <= 0:
            continue
        area = np.pi * (diameter / 2) ** 2
        rows.append({
            "turbine_type": turbine_type,
            "diameter_m": diameter,
            "rated_kw": rated_kw,
            "specific_power": rated_kw * 1000 / area,
        })

    _REFERENCE_INDEX = pd.DataFrame(rows)
    return _REFERENCE_INDEX


def pick_reference(specific_power, diameter_m=None, exclude=None,
                   w_specific=1.0, w_diameter=0.5):
    """Nearest library turbine, scored on specific power and rotor size
    together. Specific power alone picks bad matches for very large or
    very small rotors, because a 50 m machine and a 150 m machine behave
    differently even at the same W/m2."""
    index = build_reference_index()
    if exclude is not None:
        index = index[index.turbine_type != exclude]

    sp_ref = index.specific_power.std() or 1.0
    score = w_specific * ((index.specific_power - specific_power) / sp_ref) ** 2

    if diameter_m is not None:
        d_ref = index.diameter_m.std() or 1.0
        score = score + w_diameter * ((index.diameter_m - diameter_m) / d_ref) ** 2

    row = index.iloc[score.values.argmin()]
    return row.turbine_type, float(row.diameter_m), float(row.specific_power)


def load_reference_curve(turbine_type):
    from windpowerlib import WindTurbine

    wt = WindTurbine(turbine_type=turbine_type, hub_height=80)
    curve = wt.power_curve.copy()
    curve.columns = ["wind_speed_ms", "power_w"]
    curve["power_kw"] = curve.power_w / 1000
    return curve[["wind_speed_ms", "power_kw"]]


def curve_to_cp(curve, rotor_diameter_m):
    area = np.pi * (rotor_diameter_m / 2) ** 2
    available_kw = 0.5 * AIR_DENSITY_STANDARD * area * curve.wind_speed_ms ** 3 / 1000
    cp = np.where(available_kw > 0, curve.power_kw / available_kw, 0.0)
    out = curve.copy()
    out["cp"] = np.clip(cp, 0.0, BETZ_LIMIT)
    return out


class Turbine:
    def __init__(
        self,
        name,
        rated_power_kw,
        rotor_diameter_m,
        cut_in_ms,
        rated_ms,
        cut_out_ms,
        hub_height_m,
        iec_class=None,
        control="pitch",
        source=None,
        measured_curve=None,
        reference="auto",
        reference_rotor_m=90.0,
    ):
        self.name = name
        self.rated_power_kw = rated_power_kw
        self.rotor_diameter_m = rotor_diameter_m
        self.cut_in_ms = cut_in_ms
        self.rated_ms = rated_ms
        self.cut_out_ms = cut_out_ms
        self.hub_height_m = hub_height_m
        self.iec_class = iec_class
        self.control = control
        self.source = source
        self.measured_curve = measured_curve
        self.reference = reference
        self.reference_rotor_m = reference_rotor_m
        self._cache = None
        self.matched_reference = None
        self.matched_reference_specific = None

    @property
    def swept_area_m2(self):
        return np.pi * (self.rotor_diameter_m / 2) ** 2

    @property
    def specific_power_w_m2(self):
        return self.rated_power_kw * 1000 / self.swept_area_m2

    def power_curve(self, step=0.5, max_speed=30.0):
        if self.measured_curve is not None:
            return self.measured_curve.copy()

        if self._cache is not None:
            return self._cache.copy()

        if self.reference == "auto":
            ref_type, ref_rotor, ref_sp = pick_reference(
                self.specific_power_w_m2, self.rotor_diameter_m
            )
            self.matched_reference = ref_type
            self.matched_reference_specific = ref_sp
        else:
            ref_type = REFERENCE_TURBINES[self.reference]
            ref_rotor = self.reference_rotor_m
            self.matched_reference = ref_type
            self.matched_reference_specific = None

        ref = load_reference_curve(ref_type)
        ref_cp = curve_to_cp(ref, ref_rotor)

        ref_rated_kw = ref.power_kw.max()
        ref_rated_ms = float(
            ref.loc[ref.power_kw >= 0.99 * ref_rated_kw, "wind_speed_ms"].iloc[0]
        )
        ref_cut_in = float(ref.loc[ref.power_kw > 0, "wind_speed_ms"].iloc[0])

        speeds = np.arange(0, max_speed + step, step)

        scale = (self.rated_ms - self.cut_in_ms) / (ref_rated_ms - ref_cut_in)
        ref_equivalent = ref_cut_in + (speeds - self.cut_in_ms) / scale

        cp = np.interp(
            ref_equivalent,
            ref_cp.wind_speed_ms.values,
            ref_cp.cp.values,
            left=0.0,
            right=0.0,
        )

        available_kw = (
            0.5 * AIR_DENSITY_STANDARD * self.swept_area_m2 * speeds ** 3 / 1000
        )
        power = cp * available_kw

        power = np.where(speeds < self.cut_in_ms, 0.0, power)
        power = np.where(speeds >= self.rated_ms, self.rated_power_kw, power)
        power = np.where(speeds > self.cut_out_ms, 0.0, power)
        power = np.clip(power, 0.0, self.rated_power_kw)

        self._cache = pd.DataFrame({"wind_speed_ms": speeds, "power_kw": power})
        return self._cache.copy()

    def cp_curve(self, step=0.5, max_speed=30.0):
        curve = self.power_curve(step, max_speed)
        available = (
            0.5 * AIR_DENSITY_STANDARD * self.swept_area_m2
            * curve.wind_speed_ms ** 3 / 1000
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cp = np.where(available > 0, curve.power_kw / available, 0.0)
        curve["cp"] = np.clip(cp, 0, BETZ_LIMIT)
        return curve

    def __repr__(self):
        return (
            f"Turbine({self.name}, {self.rated_power_kw} kW, "
            f"D={self.rotor_diameter_m} m, hub={self.hub_height_m} m, "
            f"specific={self.specific_power_w_m2:.0f} W/m2)"
        )


def shear_exponent(speed_low, speed_high, height_low, height_high):
    speed_low = np.asarray(speed_low, dtype=float)
    speed_high = np.asarray(speed_high, dtype=float)
    valid = (speed_low > 0.1) & (speed_high > 0.1)
    alpha = np.full(speed_low.shape, np.nan)
    alpha[valid] = (
        np.log(speed_high[valid] / speed_low[valid])
        / np.log(height_high / height_low)
    )
    return alpha


def extrapolate_wind(speed, height_from, height_to, alpha):
    return np.asarray(speed) * (height_to / height_from) ** alpha


def air_density(temperature_c, pressure_kpa):
    r_specific = 287.058
    return (np.asarray(pressure_kpa) * 1000) / (
        r_specific * (np.asarray(temperature_c) + 273.15)
    )


def density_corrected_speed(speed, density, control="pitch"):
    """IEC 61400-12 correction, applied to speed. Stall machines react
    more strongly than pitch ones."""
    ratio = np.asarray(density) / AIR_DENSITY_STANDARD
    exponent = 1 / 3 if control == "pitch" else 1.0
    return np.asarray(speed) * ratio ** exponent


def apply_power_curve(speed, turbine, step=0.5):
    curve = turbine.power_curve(step=step)
    power = np.interp(
        np.asarray(speed),
        curve.wind_speed_ms.values,
        curve.power_kw.values,
        left=0.0,
        right=0.0,
    )
    return np.where(np.asarray(speed) > turbine.cut_out_ms, 0.0, power)
