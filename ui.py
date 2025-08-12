"""
Gradio Interactive Demo for Thermal Cooling Performance Analysis
================================================================

This interactive demo allows users to explore the thermal performance
of the iconic jacket-and-denim ensemble under various conditions.

Run this script to launch a web interface where users can:
- Adjust environmental conditions
- Modify material properties
- Experiment with fit parameters
- Visualize thermal performance results

Requirements:
    pip install gradio plotly thermal_model

Usage:
    python gradio_demo.py
"""

import gradio as gr
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from thermal_model import ThermalModel
import json


class ThermalDemoApp:
    """Gradio application for interactive thermal analysis."""
    
    def __init__(self):
        self.model = ThermalModel()
        self.setup_interface()
    
    def analyze_thermal_performance(
        self,
        # Environmental parameters
        T_ambient, T_skin, A_total,
        # Body region parameters
        A_frac_upper,
        # Upper body fit parameters
        f_open, f_loose,
        # Material properties
        k_leather, k_denim, k_cotton,
        # Air gap thicknesses
        d_air_gap_upper, d_air_gap_lower,
        # Convection coefficients
        h_open, h_loose, h_tight, h_lower
    ):
        """
        Perform thermal analysis with user-specified parameters.
        
        Returns formatted results and visualization.
        """
        try:
            # Run the thermal analysis
            results = self.model.calculate_cooling_performance(
                T_skin=T_skin,
                T_ambient=T_ambient,
                A_total=A_total,
                A_frac_upper=A_frac_upper,
                k_leather=k_leather,
                k_cotton=k_cotton,
                k_denim=k_denim,
                d_air_gap_upper=d_air_gap_upper,
                d_air_gap_lower=d_air_gap_lower,
                f_open=f_open,
                f_loose=f_loose,
                h_open=h_open,
                h_loose=h_loose,
                h_tight=h_tight,
                h_lower=h_lower
            )
            
            # Generate summary
            summary = self.model.get_performance_summary(results)
            
            # Create formatted output
            output_text = self._format_results(results, summary)
            
            # Create visualizations
            performance_chart = self._create_performance_chart(results)
            resistance_chart = self._create_resistance_breakdown(results)
            
            return output_text, performance_chart, resistance_chart
            
        except Exception as e:
            error_msg = f"❌ **Error in calculation:** {str(e)}\n\nPlease check your input parameters."
            empty_fig = go.Figure()
            empty_fig.add_annotation(text="Error in calculation", 
                                   x=0.5, y=0.5, showarrow=False)
            return error_msg, empty_fig, empty_fig
    
    def _format_results(self, results, summary):
        """Format results for display in the Gradio interface."""
        output = []
        
        # Header
        output.append("# 🔥 Thermal Analysis Results")
        output.append("")
        
        # Key metrics
        output.append("## 📊 Key Performance Metrics")
        output.append(f"- **Total Heat Dissipation:** {results.total_heat_dissipation:.1f} W")
        output.append(f"- **Thermal Efficiency:** {results.thermal_efficiency:.1f}% of 100W target")
        output.append(f"- **Performance Status:** {results.performance_status}")
        output.append("")
        
        # Detailed breakdown
        output.append("## 🔧 Detailed Analysis")
        output.append(f"- **Upper Body Contribution:** {results.heat_dissipation_upper:.1f} W")
        output.append(f"- **Lower Body Contribution:** {results.heat_dissipation_lower:.1f} W")
        output.append(f"- **Upper Body Resistance:** {results.R_upper_equivalent:.3f} m²·K/W")
        output.append(f"- **Lower Body Resistance:** {results.R_lower_total:.3f} m²·K/W")
        output.append("")
        
        # Engineering assessment
        output.append("## 🎯 Engineering Assessment")
        output.append(summary["Engineering Assessment"])
        
        return "\n".join(output)
    
    def _create_performance_chart(self, results):
        """Create a performance visualization chart."""
        # Data for the chart
        categories = ['Upper Body', 'Lower Body', 'Total System']
        heat_dissipation = [
            results.heat_dissipation_upper,
            results.heat_dissipation_lower,
            results.total_heat_dissipation
        ]
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars
        fig.add_trace(go.Bar(
            x=categories[:2],
            y=heat_dissipation[:2],
            name='Heat Dissipation',
            marker_color=['#FF6B6B', '#4ECDC4'],
            text=[f'{val:.1f}W' for val in heat_dissipation[:2]],
            textposition='auto'
        ))
        
        # Add target line
        fig.add_hline(y=100, line_dash="dash", line_color="red",
                      annotation_text="100W TDP Target")
        
        # Add total as separate trace
        fig.add_trace(go.Bar(
            x=[categories[2]],
            y=[heat_dissipation[2]],
            name='Total Performance',
            marker_color='#45B7D1',
            text=f'{heat_dissipation[2]:.1f}W',
            textposition='auto'
        ))
        
        fig.update_layout(
            title='Thermal Performance Breakdown',
            xaxis_title='System Component',
            yaxis_title='Heat Dissipation (W)',
            showlegend=False,
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def _create_resistance_breakdown(self, results):
        """Create thermal resistance breakdown visualization."""
        # Data for resistance comparison
        resistances = {
            'Upper Body': results.R_upper_equivalent,
            'Lower Body': results.R_lower_total
        }
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(resistances.keys()),
            values=list(resistances.values()),
            hole=0.3,
            marker_colors=['#FF6B6B', '#4ECDC4']
        )])
        
        fig.update_layout(
            title='Thermal Resistance Distribution',
            height=400,
            template='plotly_white'
        )
        
        return fig
    
    def parameter_sweep_analysis(self, param_name, param_range):
        """Perform parameter sweep analysis for sensitivity study."""
        base_params = {
            'T_ambient': 26.0,
            'T_skin': 33.5,
            'A_total': 1.8,
            'A_frac_upper': 0.6,
            'f_open': 0.35,
            'f_loose': 0.50
        }
        
        results_list = []
        param_values = np.linspace(param_range[0], param_range[1], 20)
        
        for value in param_values:
            params = base_params.copy()
            params[param_name] = value
            
            try:
                result = self.model.calculate_cooling_performance(**params)
                results_list.append({
                    param_name: value,
                    'Heat_Dissipation': result.total_heat_dissipation,
                    'Thermal_Efficiency': result.thermal_efficiency
                })
            except:
                continue
        
        return pd.DataFrame(results_list)
    
    def setup_interface(self):
        """Set up the Gradio interface."""
        
        # Custom CSS for styling
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .title {
            text-align: center;
            color: #2C3E50;
            margin-bottom: 20px;
        }
        .description {
            text-align: center;
            color: #7F8C8D;
            margin-bottom: 30px;
        }
        """
        
        with gr.Blocks(css=css, title="Jensen-TDP: Thermal Analysis Demo") as self.demo:
            
            # Header
            gr.Markdown("""
            # 🔥 Jensen-TDP: Thermal Cooling Performance Analyzer
            
            **Interactive demo for analyzing the passive cooling performance of an iconic jacket-and-denim ensemble**
            
            Based on the research paper: *"Thermal Overload: A Holistic Analysis of the Jacket-and-Denim Heatsink Paradigm"*
            
            ---
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("## 🌡️ Environmental Conditions")
                    
                    T_ambient = gr.Slider(
                        minimum=15, maximum=35, value=26.0, step=0.5,
                        label="Ambient Temperature (°C)",
                        info="Environmental temperature"
                    )
                    
                    T_skin = gr.Slider(
                        minimum=30, maximum=37, value=33.5, step=0.1,
                        label="Skin Temperature (°C)",
                        info="Body surface temperature"
                    )
                    
                    A_total = gr.Slider(
                        minimum=1.2, maximum=2.5, value=1.8, step=0.1,
                        label="Total Body Surface Area (m²)",
                        info="Total heat transfer area"
                    )
                    
                    gr.Markdown("## 👕 Upper Body Configuration")
                    
                    A_frac_upper = gr.Slider(
                        minimum=0.5, maximum=0.7, value=0.6, step=0.01,
                        label="Upper Body Area Fraction",
                        info="Fraction of total area (upper body)"
                    )
                    
                    f_open = gr.Slider(
                        minimum=0.1, maximum=0.6, value=0.35, step=0.05,
                        label="Open Front Zone Fraction",
                        info="Jacket front opening (chimney effect)"
                    )
                    
                    f_loose = gr.Slider(
                        minimum=0.2, maximum=0.7, value=0.50, step=0.05,
                        label="Loose Fit Zone Fraction",
                        info="Side areas with air gap"
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🧥 Material Properties")
                    
                    k_leather = gr.Slider(
                        minimum=0.08, maximum=0.25, value=0.15, step=0.01,
                        label="Leather Thermal Conductivity (W/m·K)",
                        info="Calfskin leather properties"
                    )
                    
                    k_denim = gr.Slider(
                        minimum=0.04, maximum=0.12, value=0.06, step=0.01,
                        label="Denim Thermal Conductivity (W/m·K)",
                        info="Cotton denim fabric"
                    )
                    
                    k_cotton = gr.Slider(
                        minimum=0.03, maximum=0.08, value=0.05, step=0.005,
                        label="Cotton T-shirt Conductivity (W/m·K)",
                        info="Base layer fabric"
                    )
                    
                    gr.Markdown("## 💨 Air Gap & Convection")
                    
                    d_air_gap_upper = gr.Slider(
                        minimum=0.002, maximum=0.020, value=0.01, step=0.002,
                        label="Upper Air Gap Thickness (m)",
                        info="Jacket interior air space"
                    )
                    
                    d_air_gap_lower = gr.Slider(
                        minimum=0.001, maximum=0.010, value=0.005, step=0.001,
                        label="Lower Air Gap Thickness (m)",
                        info="Pants interior air space"
                    )
                    
                    h_open = gr.Slider(
                        minimum=8, maximum=25, value=15.0, step=1.0,
                        label="Open Zone Convection (W/m²·K)",
                        info="Chimney effect strength"
                    )
                    
                    h_lower = gr.Slider(
                        minimum=2, maximum=10, value=5.0, step=0.5,
                        label="Lower Body Convection (W/m²·K)",
                        info="Pants external convection"
                    )
                    
                    h_loose = gr.Slider(
                        minimum=3, maximum=12, value=6.0, step=0.5,
                        label="Loose Zone Convection (W/m²·K)",
                        info="Jacket side areas"
                    )
                    
                    h_tight = gr.Slider(
                        minimum=2, maximum=8, value=3.5, step=0.5,
                        label="Tight Zone Convection (W/m²·K)",
                        info="Close-fitting areas"
                    )
            
            # Analysis button
            analyze_btn = gr.Button("🔬 Run Thermal Analysis", variant="primary", size="lg")
            
            # Results section
            with gr.Row():
                with gr.Column(scale=1):
                    results_output = gr.Markdown()
                
                with gr.Column(scale=1):
                    performance_plot = gr.Plot()
                    resistance_plot = gr.Plot()
            
            # Preset scenarios
            gr.Markdown("## 🎛️ Preset Scenarios")
            
            with gr.Row():
                scenario_btns = [
                    gr.Button("❄️ Cold Office (20°C)", size="sm"),
                    gr.Button("🌡️ Standard Room (26°C)", size="sm"),
                    gr.Button("🔥 Warm Environment (30°C)", size="sm"),
                    gr.Button("🏃 Active/Exercise Mode", size="sm")
                ]
            
            # Event handlers
            analyze_btn.click(
                fn=self.analyze_thermal_performance,
                inputs=[
                    T_ambient, T_skin, A_total, A_frac_upper,
                    f_open, f_loose,
                    k_leather, k_denim, k_cotton,
                    d_air_gap_upper, d_air_gap_lower,
                    h_open, h_loose, h_tight, h_lower
                ],
                outputs=[results_output, performance_plot, resistance_plot]
            )
            
            # Preset scenario handlers
            scenario_btns[0].click(  # Cold office
                lambda: self._apply_preset("cold"),
                outputs=[T_ambient, h_open, h_lower]
            )
            
            scenario_btns[1].click(  # Standard room  
                lambda: self._apply_preset("standard"),
                outputs=[T_ambient, h_open, h_lower]
            )
            
            scenario_btns[2].click(  # Warm environment
                lambda: self._apply_preset("warm"),
                outputs=[T_ambient, h_open, h_lower]
            )
            
            scenario_btns[3].click(  # Active mode
                lambda: self._apply_preset("active"),
                outputs=[T_skin, h_open, h_loose, h_lower]
            )
            
            # Footer
            gr.Markdown("""
            ---
            
            **About this model:** This analysis treats clothing as a thermal management system, 
            modeling heat transfer through multiple parallel and series resistance networks. 
            The model accounts for material conduction, air gaps, and natural convection effects.
            
            **Disclaimer:** This is a simplified engineering model for educational and 
            entertainment purposes. Real thermal behavior involves many additional factors.
            """)
    
    def _apply_preset(self, scenario):
        """Apply preset parameter configurations."""
        presets = {
            "cold": (20.0, 18.0, 4.0),      # T_ambient, h_open, h_lower
            "standard": (26.0, 15.0, 5.0),  # Default values
            "warm": (30.0, 12.0, 6.0),      # Reduced convection
            "active": (None, None, None)     # Special case handled below
        }
        
        if scenario == "active":
            return 35.0, 20.0, 8.0, 7.0  # T_skin, h_open, h_loose, h_lower
        else:
            return presets[scenario]
    
    def launch(self, **kwargs):
        """Launch the Gradio interface."""
        return self.demo.launch(**kwargs)


def create_parameter_sensitivity_analysis():
    """Create a parameter sensitivity analysis visualization."""
    model = ThermalModel()
    
    # Test different ambient temperatures
    temps = np.arange(15, 35, 1)
    cooling_performance = []
    
    for temp in temps:
        try:
            result = model.calculate_cooling_performance(T_ambient=temp)
            cooling_performance.append(result.total_heat_dissipation)
        except:
            cooling_performance.append(0)
    
    # Create sensitivity plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=temps,
        y=cooling_performance,
        mode='lines+markers',
        name='Cooling Performance',
        line=dict(color='#45B7D1', width=3),
        marker=dict(size=6)
    ))
    
    fig.add_hline(y=100, line_dash="dash", line_color="red",
                  annotation_text="100W TDP Target")
    
    fig.update_layout(
        title='Cooling Performance vs Ambient Temperature',
        xaxis_title='Ambient Temperature (°C)',
        yaxis_title='Heat Dissipation Capacity (W)',
        template='plotly_white',
        height=400
    )
    
    return fig


def main():
    """Main function to launch the demo application."""
    print("🚀 Launching Jensen-TDP Thermal Analysis Demo...")
    print("📊 Initializing thermal model and interface...")
    
    app = ThermalDemoApp()
    
    print("✅ Ready! Opening browser interface...")
    
    # Launch with public sharing option
    app.launch(
        # share=False,  # Set to True for public sharing
        # server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )


if __name__ == "__main__":
    main()
