# 🧬 BioSentinel 2.0 — AI Bioreactor Digital Twin & Fault Sentinel

> **Physics-grounded bioprocess digital twin, real-time ML anomaly classifier, and counterfactual prescriptive control engine for high-value microbial fermentations.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-157%20Passed-success.svg)](#test-suite)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem Statement & Overview

Industrial biomanufacturing batches (therapeutic proteins, enzymes, monoclonal antibodies, cultivated products) can cost **$50,000 to $500,000+ per run**. When physical actuator failures or biological contamination occur:
1. **Static SCADA threshold alarms trigger too late**—typically after irreversible cellular damage or culture loss.
2. **Operators face high cognitive load**, lacking actionable guidance on whether an anomaly is reversible through physical actuators (e.g., agitator speed, nutrient feed trim) or biological (irreversible contamination requiring emergency harvest/CIP abort).
3. **Black-box models fail** because they ignore the underlying biophysical and kinetic constraints governing cell respiration and substrate uptake.

**BioSentinel 2.0** bridges this gap by marrying **first-principles mechanistic ODE modeling** (dual-Monod growth and off-gas stoichiometry) with an industrial-grade **LightGBM regime classifier (184 rolling features, Macro F1 = 94.6%)**, a **Dynamic Intervention Window (DIW)** countdown, and a **Counterfactual Rollout Engine** that evaluates candidate interventions in real time.

---

## 🚀 Key Features

* **Mechanistic Digital Twin (`src/simulator.py`)**:
  * 4th/5th-order Runge-Kutta ODE integration (`scipy.integrate.solve_ivp`).
  * Continuous state tracking of Biomass ($X$), Substrate ($S$), Dissolved Oxygen ($C_L$), Contaminant ($X_c$), Oxygen Uptake Rate ($\text{OUR}$), Carbon Evolution Rate ($\text{CER}$), and Respiratory Quotient ($\text{RQ}$).
* **Telemetry Feature Engineering (`src/features.py`)**:
  * 184 engineered features computed over rolling multi-scale time windows (5, 15, 30, and 60 minutes) tracking sensor drift, volatility, kinetic ratios, and derivatives.
* **LightGBM Regime Classifier (`src/predict.py`)**:
  * 4-class gradient boosted model categorizing reactor state into **Healthy**, **$k_La$ Limitation**, **Substrate Overfeeding**, and **Contamination**.
  * Pre-onset label denoising during training ensures zero false alarms during pre-fault nominal operation.
* **Kinetic Stress $\lambda(t)$ & Dynamic Recoverability $R(t)$ (`src/stress.py`)**:
  * Biophysically grounded stress accumulation:
    $$R(t) = \exp\left(-\int_0^t \lambda(\tau)\,d\tau\right)$$
  * Live **Dynamic Intervention Window (DIW)**: Exact countdown in minutes until the culture crosses the irreversible **Point of No Return (PNR)** ($R \le 10\%$).
* **Counterfactual Rollout Engine (`src/counterfactual.py`)**:
  * Simulates candidate interventions forward in time ($+150\text{ RPM}$, $-30\%\text{ feed}$, combined, no-action).
  * Automatically handles biological logic: recognizes that foreign microbial contamination cannot be cured by physical actuators, prescribing **TRIGGER CIP / HARVEST ABORT**.
* **Real-Time Dual Cockpit Interface**:
  * **FastAPI + SSE Web App (`run.py` on `:8000`)**: Responsive dark-mode SCADA dashboard with interactive Plotly telemetry, dynamic fault color synchronization (Amber/Orange/Crimson), and Gemini 1.5 bioprocess reasoning.
  * **Streamlit Live Cockpit (`app.py` on `:8501`)**: Interactive playback slider, confusion matrix, and top-20 feature importance analyzer.

---

## 🔬 Operational Scenarios

| Scenario | Injected Fault | Biophysical Signature | AI & Physics Output | Prescribed Action |
| :--- | :--- | :--- | :--- | :--- |
| **✅ Healthy** | Nominal baseline | Balanced dual-Monod growth; $\text{RQ} \approx 1.0$ | $R(t) = 100\%$, $\text{DIW} > 99\text{ h}$<br/>Accent: **Neon Green** | Maintain baseline parameters |
| **⚠️ $k_La$ Limitation** | Sparger fouling ($t=8\text{h}$) | Oxygen mass transfer drops $70\%$; DO sags $< 2.0\text{ mg/L}$ | $R(t) \approx 80\%$, $\text{DIW} \approx 185\text{ min}$<br/>Accent: **Amber** | **$+150\text{ RPM}$ Agitation** restores oxygen transfer |
| **🔥 Substrate Overfeed** | Feed pump runaway ($t=6\text{h}$) | Glucose over-delivered ($3.0\times$); Crabtree overflow; $\text{RQ}$ surge | $R(t) \approx 85\%$, $\text{DIW} \approx 140\text{ min}$<br/>Accent: **Amber/Orange** | **$-30\%$ Feed Trim** halts overflow metabolism |
| **☣️ Contamination** | Foreign microbial breach ($t=5\text{h}$) | Competitor population diverges metabolism; sharp $\text{RQ}$ anomaly | $R(t) = 0.0\%$ (IRREVERSIBLE)<br/>$\text{DIW} = 0\text{ min}$ (BREACH)<br/>Accent: **Crimson** | **TRIGGER CIP / HARVEST ABORT** (physical actuators ineffective) |

---

## 📁 Repository Structure

```
BioSentinel/
├── api/
│   └── server.py              # FastAPI backend: SSE streaming, counterfactuals, Gemini explanation
├── data/
│   ├── formulas/              # Extended bioprocess model specifications & equations
│   ├── raw/                   # Reference datasets
│   └── synthetic/             # Generated batch simulation caches
├── frontend/
│   ├── css/style.css          # Industrial dark-theme SCADA stylesheet
│   ├── js/app.js              # Real-time SSE streaming, Plotly chart manager, banner & KPI updates
│   └── index.html             # Single-page cockpit interface
├── models/
│   ├── feature_cols.json      # 184 engineered feature names
│   ├── regime_classifier.pkl  # Trained LightGBM multi-class model
│   └── training_metadata.json # Classifier metrics (Macro F1 = 94.6%)
├── src/
│   ├── config.py              # Bioreactor kinetic constants & baseline parameters
│   ├── counterfactual.py      # Prescriptive intervention rollout engine
│   ├── features.py            # Multi-scale rolling telemetry feature extraction
│   ├── predict.py             # Model inference & confidence scoring
│   ├── regimes.py             # Regime definitions & ODE differential equations
│   ├── simulator.py           # Dual-Monod ODE numerical integrator (scipy solve_ivp)
│   ├── stress.py              # Kinetic stress accumulation, R(t), DIW, & PNR calculations
│   └── train_model.py         # LightGBM training pipeline with pre-onset label denoising
├── tests/
│   ├── test_counterfactual.py # Unit tests for rollout rankings & recovery
│   ├── test_features.py       # Unit tests for 184-feature extraction
│   ├── test_simulator.py      # Unit tests for ODE integration & mass balances
│   └── test_stress.py         # Unit tests for stress accumulation & DIW calculations
├── app.py                     # Streamlit live cockpit & ML performance explorer
├── generate_pitch_pdf.py      # Pitch document generator (ReportLab)
├── requirements.txt           # Python dependencies
├── run.py                     # Uvicorn FastAPI server launcher (:8000)
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone & Environment Setup
```bash
git clone https://github.com/Adit-K06/BioSentinel.git
cd BioSentinel

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (Optional for Gemini AI)
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```
*(Alternatively, you can enter your key directly in the cockpit UI's password field).*

---

## 🖥️ Running the Application

### Option A: Real-Time Web Application (Recommended)
Launches the FastAPI backend and dark-mode SCADA dashboard:
```bash
python run.py
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

* Select between **Healthy**, **kLa Limitation**, **Substrate Overfeed**, and **Contamination**.
* Use **⏸ Pause** and **▶ Play** to control playback seamlessly.
* Watch the timer stop cleanly at **24.00 h** with option to **↩ Replay**.
* Explore the **🤖 AI Insights** tab for counterfactual rankings and Gemini root-cause explanations.

### Option B: Streamlit Cockpit
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** for the model performance breakdown, confusion matrix, and feature importances.

---

## 🧪 Test Suite

The project includes a comprehensive unit test suite covering kinetic formulations, rolling feature generation, classifier outputs, and counterfactual rollouts:

```bash
pytest
```

**Results**:
```text
tests/test_counterfactual.py  ................................................  [ 32%]
tests/test_features.py        ..................................                [ 54%]
tests/test_simulator.py       ...........................................       [ 81%]
tests/test_stress.py          .............................                     [100%]

======================= 157 passed in 60.39s (100% pass rate) =======================
```

---

## 📐 Mathematical Formulation

### 1. Dual-Monod Growth Kinetics
$$\mu(S, C_L) = \mu_{\max} \cdot \left(\frac{S}{K_S + S}\right) \cdot \left(\frac{C_L}{K_O + C_L}\right)$$

### 2. Mass Balance Differential Equations
$$\frac{dX}{dt} = (\mu - D) X$$
$$\frac{dS}{dt} = D(S_{\text{feed}} - S) - \left(\frac{\mu}{Y_{X/S}} + m_S\right) X$$
$$\frac{dC_L}{dt} = k_La (C_L^* - C_L) - \text{OUR}$$

### 3. Off-Gas Transfer Rates & Respiratory Quotient
$$\text{OUR} = \left(\frac{\mu}{Y_{X/O}} + m_O\right) X \qquad \text{CER} = \left(\frac{\mu}{Y_{X/C}} + m_C\right) X \qquad \text{RQ} = \frac{\text{CER}}{\text{OUR}}$$

### 4. Kinetic Stress Accumulation & Dynamic Recoverability
$$R(t) = \exp\left(-\int_0^t \lambda(\tau)\,d\tau\right)$$
$$\text{DIW} = \frac{-\ln(R_{\text{pnr}}) - \int_0^t \lambda(\tau)\,d\tau}{\lambda(t)}$$

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
