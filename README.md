# 🔥 Jensen-TDP
Cooling Performance Estimation of A public-figure-inspired outfit

## 🎯 Project Goal

In short: **Why is Jensen cool?**

This project applies rigorous thermal engineering principles to analyze the passive heat dissipation performance of an iconic tech executive outfit. We model the human body as a 100W "android" and ask the critical question: *Does this famous ensemble risk thermal throttling under sustained operation?*

## 📊 The Answer

**Spoiler: It depends on the temperature!** 

- ❄️ **18°C Environment**: 116.7W dissipation → **PASS** ✅
- 🌡️ **26°C Environment**: 56.5W dissipation → **FAIL** ❌ (43.5W deficit)

The outfit performs surprisingly well in cooler conditions but suffers significant thermal bottlenecks at typical room temperature.

**🔗 [Full Analysis & Results](https://NewJerseyStyle.github.io/Jensen-TDP/)**

## 🎮 Interactive Demo

Try the live demo to explore different scenarios and see how environmental conditions dramatically affect thermal performance:

**🚀 [Launch Interactive Demo](https://huggingface.co/spaces/npc0/Jensen-TDP)**

## 💻 Running Locally

### Quick Start
```bash
# Clone the repository
git clone https://github.com/NewJerseyStyle/Jensen-TDP.git
cd Jensen-TDP

# Install dependencies
pip install gradio plotly numpy pandas

# Run the interactive UI
python ui.py
```

### Python REPL usage
```python
from thermal_model import ThermalModel

model = ThermalModel()
results = model.calculate_cooling_performance(T_ambient=26.0)
print(f"Heat dissipation: {results.total_heat_dissipation:.1f}W")
```

## 🔬 Technical Approach

The model implements a full-body thermal resistance network:
- **Upper Body**: 3-zone parallel model (open front, loose fit, tight fit)
- **Lower Body**: Single-zone model with denim and air gaps
- **Heat Transfer**: Conduction + convection analysis with real material properties
