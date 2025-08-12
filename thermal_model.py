"""
Thermal Cooling Performance Model for Jacket-and-Denim Ensemble
================================================================

This module implements a holistic thermal analysis model for evaluating
the passive cooling performance of a leather jacket and denim jeans combination.
Based on the IEEE paper "Thermal Overload: A Holistic Analysis of the 
Jacket-and-Denim Heatsink Paradigm".

The model treats the human body as a parallel thermal network with:
- Upper body (jacket system): 3-zone parallel model
- Lower body (pants system): single-zone model

Author: Thermal Engineering Research Team
License: MIT
"""

import math
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class ThermalResults:
    """Data class to store thermal analysis results."""
    # Upper body results
    R_upper_equivalent: float
    heat_dissipation_upper: float
    
    # Lower body results  
    R_lower_total: float
    heat_dissipation_lower: float
    
    # Total system results
    total_heat_dissipation: float
    total_effective_resistance: float
    thermal_efficiency: float
    performance_status: str


class ThermalModel:
    """
    Thermal analysis model for jacket-and-denim ensemble.
    
    This class implements a full-body thermal resistance network model
    to evaluate the passive cooling performance of the iconic outfit.
    """
    
    def __init__(self):
        """Initialize the thermal model with default parameters."""
        # Default material properties (W/m·K)
        self.k_leather = 0.15
        self.k_cotton = 0.05
        self.k_denim = 0.06
        self.k_air = 0.026
        
        # Default material thicknesses (m)
        self.d_leather = 0.0015
        self.d_cotton = 0.001
        self.d_denim = 0.0015
        
        # Default convection coefficients (W/m²·K)
        self.h_open = 15.0     # Open front zone (chimney effect)
        self.h_loose = 6.0     # Loose fit zone
        self.h_tight = 3.5     # Tight fit zone
        self.h_lower = 5.0     # Lower body average
    
    def calculate_cooling_performance(
        self,
        # Environmental & physiological parameters
        T_skin: float = 33.5,           # Skin temperature (°C)
        T_ambient: float = 26.0,        # Ambient temperature (°C)
        A_total: float = 1.8,           # Total surface area (m²)
        
        # Body region distribution
        A_frac_upper: float = 0.6,      # Upper body area fraction
        
        # Material thermal conductivities (W/m·K)
        k_leather: float = None,
        k_cotton: float = None,
        k_denim: float = None,
        k_air: float = None,
        
        # Material thicknesses (m)
        d_leather: float = None,
        d_cotton: float = None,
        d_denim: float = None,
        d_air_gap_upper: float = 0.01,  # Upper body air gap
        d_air_gap_lower: float = 0.005, # Lower body air gap
        
        # Upper body geometry & convection parameters
        f_open: float = 0.35,           # Open front fraction
        f_loose: float = 0.50,          # Loose fit fraction
        h_open: float = None,
        h_loose: float = None,
        h_tight: float = None,
        
        # Lower body convection parameter
        h_lower: float = None
    ) -> ThermalResults:
        """
        Calculate the passive cooling performance of the full ensemble.
        
        The model divides the body into two parallel thermal networks:
        1. Upper body (Jacket System): 3-zone parallel model
        2. Lower body (Pants System): single-zone model
        
        Returns:
            ThermalResults: Complete thermal analysis results
        """
        
        # Use instance defaults if parameters not provided
        k_leather = k_leather or self.k_leather
        k_cotton = k_cotton or self.k_cotton
        k_denim = k_denim or self.k_denim
        k_air = k_air or self.k_air
        
        d_leather = d_leather or self.d_leather
        d_cotton = d_cotton or self.d_cotton
        d_denim = d_denim or self.d_denim
        
        h_open = h_open or self.h_open
        h_loose = h_loose or self.h_loose
        h_tight = h_tight or self.h_tight
        h_lower = h_lower or self.h_lower
        
        # Validate inputs
        if f_open + f_loose > 1.0:
            raise ValueError("Upper body area fractions (f_open + f_loose) cannot exceed 1.0")
        
        if T_skin <= T_ambient:
            raise ValueError("Skin temperature must be higher than ambient temperature")
        
        # --- Area Distribution ---
        A_upper = A_total * A_frac_upper
        A_lower = A_total * (1 - A_frac_upper)
        
        # --- Calculate Base Thermal Resistances (m²·K/W) ---
        R_cond_leather = d_leather / k_leather
        R_cond_cotton = d_cotton / k_cotton
        R_cond_denim = d_denim / k_denim
        R_cond_air_upper = d_air_gap_upper / k_air
        R_cond_air_lower = d_air_gap_lower / k_air
        
        R_conv_open = 1 / h_open
        R_conv_loose = 1 / h_loose
        R_conv_tight = 1 / h_tight
        R_conv_lower = 1 / h_lower
        
        # --- Upper Body Heat Transfer Analysis ---
        # Calculate series resistance for each zone
        R_zone_open = R_cond_cotton + R_conv_open
        R_zone_loose = R_cond_cotton + R_cond_air_upper + R_cond_leather + R_conv_loose
        R_zone_tight = R_cond_cotton + R_cond_leather + R_conv_tight
        
        # Calculate equivalent resistance for upper body parallel network
        f_tight = 1 - f_open - f_loose
        inv_R_upper = (f_open / R_zone_open) + (f_loose / R_zone_loose) + (f_tight / R_zone_tight)
        R_upper_equiv = 1 / inv_R_upper
        
        # Calculate upper body heat dissipation
        delta_T = T_skin - T_ambient
        Q_upper = (A_upper * delta_T) / R_upper_equiv
        
        # --- Lower Body Heat Transfer Analysis ---
        # Calculate series resistance for lower body
        R_lower_total = R_cond_denim + R_cond_air_lower + R_conv_lower
        
        # Calculate lower body heat dissipation
        Q_lower = (A_lower * delta_T) / R_lower_total
        
        # --- Total System Performance ---
        Q_total = Q_upper + Q_lower
        R_total_effective = (A_total * delta_T) / Q_total
        
        # Calculate thermal efficiency (compared to 100W TDP)
        TDP_reference = 100.0
        thermal_efficiency = (Q_total / TDP_reference) * 100
        
        # Determine performance status
        if Q_total >= TDP_reference:
            performance_status = "PASS - Adequate cooling performance"
        elif Q_total >= 0.9 * TDP_reference:
            performance_status = "MARGINAL - Near thermal limit"
        else:
            performance_status = f"FAIL - {TDP_reference - Q_total:.1f}W thermal deficit"
        
        return ThermalResults(
            R_upper_equivalent=R_upper_equiv,
            heat_dissipation_upper=Q_upper,
            R_lower_total=R_lower_total,
            heat_dissipation_lower=Q_lower,
            total_heat_dissipation=Q_total,
            total_effective_resistance=R_total_effective,
            thermal_efficiency=thermal_efficiency,
            performance_status=performance_status
        )
    
    def get_performance_summary(self, results: ThermalResults, TDP: float = 100.0) -> Dict[str, str]:
        """
        Generate a human-readable performance summary.
        
        Args:
            results: ThermalResults object from calculate_cooling_performance
            TDP: Target thermal design power in watts
            
        Returns:
            Dictionary containing formatted summary strings
        """
        summary = {
            "Upper Body Performance": f"{results.heat_dissipation_upper:.1f}W "
                                    f"(R_eq = {results.R_upper_equivalent:.3f} m²·K/W)",
            
            "Lower Body Performance": f"{results.heat_dissipation_lower:.1f}W "
                                    f"(R_total = {results.R_lower_total:.3f} m²·K/W)",
            
            "Total Heat Dissipation": f"{results.total_heat_dissipation:.1f}W",
            
            "Thermal Efficiency": f"{results.thermal_efficiency:.1f}% of {TDP}W target",
            
            "Performance Status": results.performance_status,
            
            "Engineering Assessment": self._get_engineering_assessment(results, TDP)
        }
        
        return summary
    
    def _get_engineering_assessment(self, results: ThermalResults, TDP: float) -> str:
        """Generate engineering assessment based on results."""
        Q_total = results.total_heat_dissipation
        
        if Q_total >= TDP:
            return ("System operates within thermal limits. "
                   "Passive cooling is sufficient for sustained operation.")
        elif Q_total >= 0.95 * TDP:
            return ("System operates at thermal edge. "
                   "Consider environmental optimization or brief duty cycles.")
        elif Q_total >= 0.85 * TDP:
            return ("Thermal deficit detected. "
                   "Active cooling or garment optimization recommended.")
        else:
            return ("Significant thermal bottleneck. "
                   "Major design modifications required for target TDP.")


def demo_calculation():
    """Demonstration of the thermal model with default parameters."""
    model = ThermalModel()
    results = model.calculate_cooling_performance()
    summary = model.get_performance_summary(results)
    
    print("=" * 60)
    print("THERMAL ANALYSIS: Jacket-and-Denim Heatsink Performance")
    print("=" * 60)
    
    for key, value in summary.items():
        print(f"{key:.<25}: {value}")
    
    print("\n" + "=" * 60)
    print("Analysis complete. See full results above.")
    
    return results, summary


if __name__ == "__main__":
    demo_calculation()
