# ⚡ GridSense — EV Charging Station Grid-Load Forecasting

_Predicting EV charging station load using Temporal Fusion Transformer (TFT) simulations to empower efficient grid management._

---

## 📌 Table of Contents
- <a href="#overview">Overview</a>
- <a href="#business-problem">Business Problem</a>
- <a href="#simulated-parameters">Simulated Parameters</a>
- <a href="#tools--technologies">Tools & Technologies</a>
- <a href="#project-structure">Project Structure</a>
- <a href="#api-endpoints">API Endpoints</a>
- <a href="#dashboard--design">Dashboard & Design</a>
- <a href="#how-to-run-this-project">How to Run This Project</a>
- <a href="#future-implementations">Future Implementations</a>

---
<h2><a class="anchor" id="overview"></a>Overview</h2>

GridSense is an advanced interactive Web platform designed to forecast electric vehicle (EV) charging station load on local grids. Through a premium, dark-themed dashboard, it provides powerful tools for administrators to simulate and visualize expected load profiles, allowing proactive energy management based on varying environmental and traffic factors.

---
<h2><a class="anchor" id="business-problem"></a>Business Problem</h2>

Managing the load of EV charging stations is a complex task due to varying user demand and environmental factors. This project aims to:
- Provide robust interactive tools to simulate live energy demands 
- Output reliable future forecasts for grid peak-hour management
- Avoid energy grid overloads by intelligently mapping weather, port, and traffic parameters to energy spikes
- Evaluate station status and visualize confidence metrics in real-time

---
<h2><a class="anchor" id="simulated-parameters"></a>Simulated Parameters</h2>

Currently, GridSense utilizes a simulated predictive engine mirroring a Temporal Fusion Transformer model, taking into account the following parameters:
- **Location**: Specific charging hub targeting
- **Weather**: Sunny, cloudy, or rainy variations affecting EV adoption for the day
- **Traffic**: Real-time congestion percentages dictating arrival rates
- **Ports**: Number of active charging connections

---
<h2><a class="anchor" id="tools--technologies"></a>Tools & Technologies</h2>

- **Backend / API**: Python & Flask
- **Frontend / UI**: HTML5, Vanilla JavaScript, Chart.js (Interactive forecasting graphs)
- **Styling**: Custom CSS properties with Dark Neon Theme (Electric Cyan, Volt Green, Deep Navy) 
- **Design System**: Glassmorphism cards, interactive particle backgrounds, responsive Layouts

---
<h2><a class="anchor" id="project-structure"></a>Project Structure</h2>

```
gridsense/
│
├── app.py                      # Flask backend + forecast API
├── requirements.txt            # Python dependencies
├── templates/
│   ├── base.html               # Shared layout (nav, footer)
│   ├── index.html              # Landing page
│   ├── about.html              # About / Team / Tech stack
│   ├── dashboard.html          # Live dashboard with Chart.js
│   └── predict.html            # User input & prediction form
└── static/
    ├── css/style.css           # Full design system (dark/light toggling)
    └── js/main.js              # Particles, theme, nav, scrolling utilities
```

---
<h2><a class="anchor" id="api-endpoints"></a>API Endpoints</h2>

GridSense serves its own internal API to actively manage forecasting requests without reloading the interface:
- `GET  /`            → Main Landing page
- `GET  /about`       → About / Team documentation
- `GET  /dashboard`   → Live interactive charting view
- `GET  /predict`     → User input and parameter simulation form
- `POST /api/forecast` → Generate JSON forecast given parameterized payload
- `GET  /api/dashboard-data` → Fetches the default or current load constraints

---
<h2><a class="anchor" id="dashboard--design"></a>Dashboard & Design</h2>
<img width="1887" height="896" alt="Screenshot 2026-04-05 215123" src="https://github.com/user-attachments/assets/eb720eb7-16b3-43fe-8d96-97c294b968b4" />
<img width="1913" height="901" alt="Screenshot 2026-04-05 215351" src="https://github.com/user-attachments/assets/591e70f2-2f08-4985-92c2-47290996ac9b" />


The application features a sleek dark neon aesthetic:
- **Color Palettes**: Electric cyan (`#00F5FF`), volt green (`#39FF14`), and deep navy (`#0A0E27`).
- **Typography**: Paired Orbitron and Syne fonts for a modern, futuristic gauge.
- **Interactions**: Smooth CSS transitions, glassmorphism over hover elements, and persistent light/dark mode states loaded via `localStorage`.
- Fully responsive architecture spanning from high-res desktops to mobile views.

---
<h2><a class="anchor" id="how-to-run-this-project"></a>How to Run This Project</h2>

1. Navigate to the project directory:
```bash
cd gridsense  # or the directory where it's cloned
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
3. Run the application:
```bash
python app.py
```
4. Access the App: Open your browser to `http://localhost:5000`.

---
<h2><a class="anchor" id="future-implementations"></a>Future Implementations</h2>

- Expand the Python backend to use a live PyTorch Temporal Fusion Transformer (TFT) instead of rule-based pseudo-generation.
- Integrate active weather API routing (e.g. OpenWeatherMap) to fetch real ambient conditions of the station automatically.
- Database expansion (PostgreSQL) for saving historical logs of loads over weeks to improve the model retraining lifecycle.
