# HyperlocalAQ: A High-Resolution Air Quality Modeling Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

`HyperlocalAQ` is a Python tool for creating high-resolution, neighborhood-level air quality maps. It works by fusing data from multiple sources: official ground stations, satellite observations, weather APIs, active fire sensors, real-time traffic data, and static pollution source maps.

This software was developed for a research paper submitted to the Journal of Open Source Software (JOSS). The paper and its associated files can be found in the `/paper` directory of this repository.

![Example Map](paper/map_screenshot.png)

## Statement of Need

Official air quality monitoring stations are often too sparse to capture the significant pollution variations that exist within a city. `HyperlocalAQ` addresses this critical data gap by providing an accessible, open-source framework to generate granular air quality estimates. It is designed for environmental health researchers, urban planners, and citizen scientists who need a more accurate picture of pollution exposure at the local level.

## Features

- **Multi-Source Data Fusion:** Integrates data from WAQI, Copernicus (CAMS), Open-Meteo, NASA FIRMS, TomTom, and OpenStreetMap.
- **Hybrid Modeling:** Uses a baseline from satellite and ground station data, which is then refined through a series of adjustments.
- **Dynamic Adjustments:** Accounts for real-time factors like local weather, downwind smoke from fires, and traffic congestion.
- **Static Source Impact:** Models the influence of nearby industrial zones, landfills, and major roads.
- **AQI Calculation:** Converts final pollutant concentrations (PM2.5, PM10, NO₂, etc.) into a standardized Air Quality Index (AQI).
- **Interactive Visualization:** Generates a detailed, interactive Folium map as the final output.

## Installation

To get started with `HyperlocalAQ`, you need Python 3.8+ and Git installed on your system.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/hyperlocal-aq-model.git
    cd hyperlocal-aq-model
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Before running the script, you must edit the `aqi_model.py` file and replace the placeholder API keys with your own valid keys.

Open `aqi_model.py` and update these lines:
```python
# --- API & File Configuration ---
WAQI_API_TOKEN = "YOUR_WAQI_TOKEN_HERE"
NASA_FIRMS_API_KEY = "YOUR_FIRMS_KEY_HERE"
TOMTOM_API_KEY = "YOUR_TOMTOM_KEY_HERE"
