"""
Builds turbine power curves from published operating points.

Manufacturers publish the curve as a picture, not a table, so we
reconstruct it from the four numbers they do publish: cut-in, rated
speed, rated power, cut-out. Between cut-in and rated the curve follows
a cubic law, which is the physics, not a guess. Above rated it flattens,
above cut-out it drops to zero.

If a real measured table ever becomes available for a turbine, drop it
in as 'measured_curve' and it wins over the reconstruction.
"""

import numpy as np
import pandas as pd

AIR_DENSITY_STANDARD = 1.225


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

    @property
    def swept_area_m2(self):
        return np.pi * (self.rotor_diameter_m / 2) ** 2

    @property
    def specific_power_w_m2(self):
        return self.rated_power_kw * 1000 / self.swept_area_m2

    def power_curve(self, step=0.5, max_speed=30.0):
        if self.measured_curve is not None:
            return self.measured_curve.copy()

        speeds = np.arange(0, max_speed + step, step)
        power = np.zeros_like(speeds)

        ramp = (speeds >= self.cut_in_ms) & (speeds < self.rated_ms)
        numerator = speeds[ramp] ** 3 - self.cut_in_ms ** 3
        denominator = self.rated_ms ** 3 - self.cut_in_ms ** 3
        power[ramp] = self.rated_power_kw * numerator / denominator

        flat = (speeds >= self.rated_ms) & (speeds <= self.cut_out_ms)
        power[flat] = self.rated_power_kw

        return pd.DataFrame({"wind_speed_ms": speeds, "power_kw": power})

    def cp_curve(self, step=0.5, max_speed=30.0):
        curve = self.power_curve(step, max_speed)
        available = (
            0.5 * AIR_DENSITY_STANDARD * self.swept_area_m2
            * curve.wind_speed_ms ** 3 / 1000
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            cp = np.where(available > 0, curve.power_kw / available, 0.0)
        curve["cp"] = np.clip(cp, 0, 0.593)
        return curve

    def __repr__(self):
        return (
            f"Turbine({self.name}, {self.rated_power_kw} kW, "
            f"D={self.rotor_diameter_m} m, hub={self.hub_height_m} m, "
            f"specific={self.specific_power_w_m2:.0f} W/m2)"
        )


def shear_exponent(speed_low, speed_high, height_low, height_high):
    """Fits the power law exponent from two measured heights."""
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
    """Power law extrapolation to hub height."""
    return np.asarray(speed) * (height_to / height_from) ** alpha


def air_density(temperature_c, pressure_kpa):
    """Ideal gas law. NASA gives PS in kPa and T2M in Celsius."""
    r_specific = 287.058
    return (np.asarray(pressure_kpa) * 1000) / (
        r_specific * (np.asarray(temperature_c) + 273.15)
    )


def density_corrected_speed(speed, density, control="pitch"):
    """
    IEC 61400-12 density correction, applied to wind speed not power.
    Pitch machines use the 1/3 exponent, stall machines 1/(3-2)=1 in the
    simplified form, so stall turbines are more sensitive.
    """
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
    power = np.where(np.asarray(speed) > turbine.cut_out_ms, 0.0, power)
    return power
