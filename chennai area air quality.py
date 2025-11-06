# IDEATHON PROTOTYPE
import pandas as pd
import numpy as np
import requests
import folium
import io
import webbrowser
import os
from datetime import date, timedelta
from sklearn.metrics.pairwise import haversine_distances
import math

# --- API & File Configuration ---
WAQI_API_TOKEN = "da77cadeed5012f680511ebc6714745443487a0d"
NASA_FIRMS_API_KEY = "c72502e93dc1a16b9c2ceff0c95043fc"
TOMTOM_API_KEY = "G2IOAmewiav2OQWSPsbMUl2ECnLgtuA5"  # Your TomTom API Key
# The CSV file should be in the same directory as this script.
NEIGHBORHOOD_CSV_PATH = '/home/jeyadev/SHIRIHER/chennai-area.csv'
print("✅ Using hardcoded API Tokens for this test run.")

# --- Model & Pollutant Configuration ---
POLLUTANTS_TO_ANALYZE = ['pm25', 'pm10', 'o3', 'no2', 'so2', 'co']
POLLUTANT_API_MAP = {'pm25': 'pm2_5', 'pm10': 'pm10', 'o3': 'ozone', 'no2': 'nitrogen_dioxide',
                     'so2': 'sulphur_dioxide', 'co': 'carbon_monoxide'}
N_NEAREST_STATIONS = 3
CAMS_DATA_EQUIVALENT_DISTANCE_KM = 15.0
MAX_OSM_ADJUSTMENT_FACTOR = 2.0

# --- Meteorological Configuration ---
METEO_API_VARS = ['temperature_2m', 'relative_humidity_2m', 'wind_speed_10m', 'wind_direction_10m']
METEO_INTERNAL_NAMES = ['temperature', 'humidity', 'wind_speed', 'wind_direction']

# --- Fire Integration Configuration ---
FIRMS_AREA_BOUNDS = [77, 10, 83, 16]
FIRE_IMPACT_WEIGHTS = {'pm25': 1.0, 'co': 1.0, 'pm10': 0.8, 'no2': 0.6, 'o3': 0.5, 'so2': 0.2}
FIRE_IMPACT_SCALING_FACTOR = 0.001

# --- Overpass API & Local Source Configuration ---
OVERPASS_API_URL = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """[out:json][timeout:180];area["name"="Chennai"]["boundary"="administrative"]->.searchArea;(way["landuse"="industrial"](area.searchArea);way["industrial"~"factory|chemical|refinery|steel_works"](area.searchArea);way["man_made"="works"](area.searchArea);node["man_made"="kiln"](area.searchArea);way["power"="plant"](area.searchArea);way["landuse"~"landfill|quarry|brownfield"](area.searchArea);way["highway"~"motorway|trunk|primary"](area.searchArea);node["amenity"="fuel"](area.searchArea);way["construction"="site"](area.searchArea););out center;"""
OSM_INFLUENCE_RADIUS_KM = 2.5
OSM_IMPACT_FACTORS = {'industrial': {'pm25': 1.15, 'pm10': 1.12, 'so2': 1.20, 'no2': 1.10, 'co': 1.10},
                      'power_plant': {'so2': 1.25, 'no2': 1.15, 'pm25': 1.10},
                      'landfill_quarry_brownfield': {'pm25': 1.12, 'pm10': 1.15},
                      'major_road_or_fuel': {'no2': 1.18, 'pm25': 1.10, 'co': 1.15},
                      'construction': {'pm25': 1.20, 'pm10': 1.20}, 'kiln': {'co': 1.25, 'pm25': 1.18}}

# --- TomTom Traffic API Configuration ---
TOMTOM_TRAFFIC_RADIUS_METERS = 1500
TOMTOM_TRAFFIC_IMPACT_WEIGHTS = {'no2': 0.30, 'pm25': 0.20, 'co': 0.25, 'pm10': 0.10}

# --- AQI CALCULATION CONFIGURATION ---
AQI_CONFIG = {
    'breakpoints': {
        'pm25': [(0, 30, 0, 50), (31, 60, 51, 100), (61, 90, 101, 200), (91, 120, 201, 300), (121, 250, 301, 400),
                 (251, 500, 401, 500)],
        'pm10': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 250, 101, 200), (251, 350, 201, 300), (351, 430, 301, 400),
                 (431, 600, 401, 500)],
        'no2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 180, 101, 200), (181, 280, 201, 300), (281, 400, 301, 400),
                (401, 533, 401, 500)],
        'so2': [(0, 40, 0, 50), (41, 80, 51, 100), (81, 380, 101, 200), (381, 800, 201, 300), (801, 1600, 301, 400),
                (1601, 2130, 401, 500)],
        'co': [(0, 1.0, 0, 50), (1.1, 2.0, 51, 100), (2.1, 10.0, 101, 200), (10.1, 17.0, 201, 300),
               (17.1, 34.0, 301, 400), (34.1, 45.2, 401, 500)],
        'o3': [(0, 50, 0, 50), (51, 100, 51, 100), (101, 168, 101, 200), (169, 208, 201, 300), (209, 748, 301, 400),
               (749, 999, 401, 500)]
    },
    'categories': [
        (0, 50, 'Good', '#2ECC71'), (51, 100, 'Satisfactory', '#F1C40F'), (101, 200, 'Moderate', '#F39C12'),
        (201, 300, 'Poor', '#E74C3C'), (301, 400, 'Very Poor', '#8E44AD'), (401, 9999, 'Severe', '#78281F')
    ]
}


# ==============================================================================
# STEP 2: DEFINE HELPER FUNCTIONS
# ==============================================================================


def find_chennai_stations():
    print("\nSearching for all available stations in Chennai...")
    try:
        response = requests.get(f"https://api.waqi.info/search/?token={WAQI_API_TOKEN}&keyword=chennai", timeout=10)
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "ok" and data.get("data"): return [s['station']['url'] for s in data["data"]]
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to find stations: {e}")
    return []


def load_neighborhood_data(file_path):
    print(f"✅ Loading neighborhood data from '{file_path}'...")
    try:
        df = pd.read_csv(file_path)
        if not all(
            c in df.columns for c in ['Zone', 'Location', 'Category', 'Latitude', 'Longitude']): return pd.DataFrame()
        return df[['Zone', 'Location', 'Category', 'Latitude', 'Longitude']]
    except Exception as e:
        print(f"  ❌ CRITICAL ERROR loading CSV: {e}")
        return pd.DataFrame()


def fetch_waqi_data(station_urls, pollutants):
    print("\nFetching real-time data from WAQI ground stations...")
    if not station_urls: return pd.DataFrame()
    station_data = []
    for station_id in station_urls:
        try:
            response = requests.get(f"https://api.waqi.info/feed/{station_id}/?token={WAQI_API_TOKEN}", timeout=10)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "ok":
                iaqi = data["data"].get("iaqi", {})
                row = {
                    'station_name': data["data"]["city"]["name"],
                    'latitude': data["data"]["city"]["geo"][0],
                    'longitude': data["data"]["city"]["geo"][1],
                    'aqi': data["data"].get("aqi", np.nan)
                }
                for p in pollutants: row[p] = iaqi.get(p, {}).get('v', np.nan)
                station_data.append(row)
        except requests.exceptions.RequestException:
            print(f"  ❌ Failed to fetch data for {station_id}")
    return pd.DataFrame(station_data)


def fetch_cams_data(neighborhood_df, api_pollutant_map):
    print("\nFetching CAMS data for each neighborhood...")
    params = {"latitude": neighborhood_df['Latitude'].tolist(), "longitude": neighborhood_df['Longitude'].tolist(),
              "current": ",".join(api_pollutant_map.values()), "timezone": "auto"}
    try:
        response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list): return pd.DataFrame()
        cams_data, reverse_map = [], {v: k for k, v in api_pollutant_map.items()}
        for i, result in enumerate(data):
            point = {'Location': neighborhood_df.iloc[i]['Location']}
            for api_name, value in result.get('current', {}).items():
                internal_name = reverse_map.get(api_name)
                if internal_name:
                    if internal_name == 'co' and pd.notna(value):
                        point[internal_name] = value / 1000.0
                    else:
                        point[internal_name] = value
            cams_data.append(point)
        print("  ✅ CAMS data fetched and CO units corrected.")
        return pd.DataFrame(cams_data)
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to fetch CAMS data: {e}")
    return pd.DataFrame()


def fetch_meteo_data(neighborhood_df):
    print("\nFetching meteorological data...")
    params = {"latitude": neighborhood_df['Latitude'].tolist(), "longitude": neighborhood_df['Longitude'].tolist(),
              "current": ",".join(METEO_API_VARS), "timezone": "auto"}
    try:
        response = requests.get("https://api.open-meteo.com/v1/forecast", params=params)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list): return pd.DataFrame()
        meteo_data, api_map = [], dict(zip(METEO_API_VARS, METEO_INTERNAL_NAMES))
        for i, result in enumerate(data):
            point = {'Location': neighborhood_df.iloc[i]['Location']}
            for name, value in result.get('current', {}).items():
                if api_map.get(name): point[api_map[name]] = value
            meteo_data.append(point)
        return pd.DataFrame(meteo_data)
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to fetch meteorological data: {e}")
    return pd.DataFrame()


def fetch_firms_data(api_key, area_bounds, day_range=1):
    print("\nFetching fire data from NASA FIRMS...")
    start_date = (date.today() - timedelta(days=day_range)).strftime('%Y-%m-%d')
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{api_key}/VIIRS_SNPP_NRT/{','.join(map(str, area_bounds))}/{day_range}/{start_date}"
    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        if "No fire detected" in response.text or not response.text.strip(): return pd.DataFrame()
        return pd.read_csv(io.StringIO(response.text))[['latitude', 'longitude', 'frp']]
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to fetch NASA FIRMS data: {e}")
    return pd.DataFrame()


def fetch_osm_pollution_sources(url, query):
    print("\nFetching static pollution sources from OpenStreetMap (Overpass API)...")
    try:
        response = requests.post(url, data=query, timeout=180,
                                 headers={'Content-Type': 'application/x-www-form-urlencoded'})
        response.raise_for_status()
        data, sources = response.json(), []
        for el in data.get('elements', []):
            tags, s_type = el.get('tags', {}), 'unknown'
            if tags.get('landuse') == 'industrial':
                s_type = 'industrial'
            elif tags.get('landuse') in ['landfill', 'quarry', 'brownfield']:
                s_type = 'landfill_quarry_brownfield'
            elif tags.get('power') == 'plant':
                s_type = 'power_plant'
            elif tags.get('highway') in ['motorway', 'trunk', 'primary'] or tags.get('amenity') == 'fuel':
                s_type = 'major_road_or_fuel'
            elif tags.get('construction') == 'site':
                s_type = 'construction'
            elif tags.get('man_made') == 'kiln':
                s_type = 'kiln'
            center = el.get('center', {})
            lat, lon = center.get('lat', el.get('lat')), center.get('lon', el.get('lon'))
            if lat and lon and s_type != 'unknown':
                sources.append({'type': s_type, 'latitude': lat, 'longitude': lon, 'name': tags.get('name', 'N/A')})
        return pd.DataFrame(sources)
    except requests.exceptions.RequestException as e:
        print(f"  ❌ Failed to fetch Overpass API data: {e}")
    return pd.DataFrame()


def fetch_tomtom_traffic_data(neighborhood_df, api_key, radius_meters):
    print("\nFetching real-time traffic data from TomTom API...")
    traffic_data = []
    for _, row in neighborhood_df.iterrows():
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={row['Latitude']},{row['Longitude']}&radius={radius_meters}&key={api_key}"
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json().get('flowSegmentData', {})
            congestion = data.get('currentTravelTime', 0) / data.get('freeFlowTravelTime', 1) - 1 if data.get(
                'freeFlowTravelTime', 0) > 0 else 0
            traffic_data.append({'Location': row['Location'], 'traffic_congestion': max(0, congestion)})
        except requests.exceptions.RequestException:
            traffic_data.append({'Location': row['Location'], 'traffic_congestion': 0})
    return pd.DataFrame(traffic_data)


def find_n_nearest_stations(neighborhood_df, station_df, n):
    print(f"\nFinding the {n} nearest stations...")
    if station_df.empty:
        neighborhood_df['nearest_stations'] = [[] for _ in range(len(neighborhood_df))]
        return neighborhood_df
    neigh_coords, stat_coords = np.radians(neighborhood_df[['Latitude', 'Longitude']].values), np.radians(
        station_df[['latitude', 'longitude']].values)
    dist_matrix = haversine_distances(neigh_coords, stat_coords) * 6371.0088
    nearest_indices = np.argsort(dist_matrix, axis=1)[:, :n]
    neighborhood_df['nearest_stations'] = [
        [{'name': station_df.iloc[j]['station_name'], 'distance': dist_matrix[i, j]} for j in nearest_indices[i]] for i
        in range(len(neighborhood_df))]
    return neighborhood_df


def calculate_hybrid_idw_values(neighborhood_df, waqi_df, cams_df, pollutants):
    print("\nSTEP 1: Calculating initial HYBRID (IDW) pollutant estimates...")
    fused_data = []
    for _, row in neighborhood_df.iterrows():
        fused_row, cams_row = {'Location': row['Location']}, cams_df[cams_df['Location'] == row['Location']]
        for p in pollutants:
            w_sum, t_weight = 0, 0
            cams_val = cams_row.iloc[0][p] if not cams_row.empty and p in cams_row and pd.notna(
                cams_row.iloc[0][p]) else None
            if cams_val is not None:
                w_sum += cams_val * (1 / (CAMS_DATA_EQUIVALENT_DISTANCE_KM ** 2))
                t_weight += (1 / (CAMS_DATA_EQUIVALENT_DISTANCE_KM ** 2))
            for station in row['nearest_stations']:
                station_val = waqi_df[waqi_df['station_name'] == station['name']].iloc[0].get(p)
                if pd.notna(station_val):
                    weight = 1 / (station['distance'] ** 2) if station['distance'] > 0 else 1e9
                    w_sum += station_val * weight
                    t_weight += weight
            fused_row[f'{p}_initial'] = w_sum / t_weight if t_weight > 0 else np.nan
        fused_data.append(fused_row)
    return pd.DataFrame(fused_data)


def apply_meteorological_adjustments(df, pollutants):
    print("\nSTEP 2: Applying METEOROLOGICAL adjustments...")
    avg_t, avg_h, avg_w = df['temperature'].mean(), df['humidity'].mean(), df['wind_speed'].mean()
    for p in pollutants:
        df[f'{p}_meteo_adjusted'] = df.apply(lambda row: adjust_single_pollutant_weather(row, p, avg_t, avg_h, avg_w),
                                             axis=1)
    return df


def adjust_single_pollutant_weather(row, p, avg_t, avg_h, avg_w):
    val = row[f'{p}_initial']
    if pd.isna(val): return np.nan
    factor = 1.0
    if row['temperature'] > avg_t: factor *= 1.1 if p == 'o3' else 1.05
    if row['humidity'] > avg_h: factor *= 1.1 if p in ['pm25', 'pm10'] else (0.95 if p == 'o3' else 1.0)
    factor *= 1.2 if row['wind_speed'] < avg_w else 0.8
    if (45 <= row['wind_direction'] <= 135): factor *= 0.9
    return val * factor


def apply_fire_adjustments(df, firms_df, pollutants):
    print("\nSTEP 3: Applying FIRE-BASED adjustments...")
    for p in pollutants: df[f'{p}_fire_adjusted'] = df[f'{p}_meteo_adjusted']
    if firms_df.empty: return df
    df_rad, firms_rad = np.radians(df[['Latitude', 'Longitude']].values), np.radians(
        firms_df[['latitude', 'longitude']].values)
    dist_matrix_km = haversine_distances(df_rad, firms_rad) * 6371.0088
    for i, loc_row in df.iterrows():
        total_impact = 0
        for j, fire_row in firms_df.iterrows():
            if dist_matrix_km[i, j] > 0:
                lat1, lon1, lat2, lon2 = map(math.radians,
                                             [loc_row['Latitude'], loc_row['Longitude'], fire_row['latitude'],
                                              fire_row['longitude']])
                bearing = (math.degrees(math.atan2(math.sin(lon2 - lon1) * math.cos(lat2),
                                                   math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
                                                       lat2) * math.cos(lon2 - lon1))) + 360) % 360
                if abs(loc_row['wind_direction'] - bearing) <= 45:
                    total_impact += fire_row['frp'] / (dist_matrix_km[i, j] ** 2)
        for p in pollutants:
            adjustment = 1 + total_impact * FIRE_IMPACT_WEIGHTS.get(p, 0) * FIRE_IMPACT_SCALING_FACTOR
            df.loc[i, f'{p}_fire_adjusted'] *= adjustment
    return df


def apply_local_source_adjustments(df, osm_df, traffic_df, pollutants):
    print("\nSTEP 4: Applying CAPPED LOCAL SOURCE (OSM & Traffic) adjustments...")
    for p in pollutants: df[f'{p}_final'] = df[f'{p}_fire_adjusted']
    if osm_df.empty and (traffic_df.empty or 'traffic_congestion' not in traffic_df.columns): return df
    df = pd.merge(df, traffic_df, on="Location", how="left").fillna({'traffic_congestion': 0})
    if not osm_df.empty:
        df_rad, osm_rad = np.radians(df[['Latitude', 'Longitude']].values), np.radians(
            osm_df[['latitude', 'longitude']].values)
        dist_matrix_km = haversine_distances(df_rad, osm_rad) * 6371.0088
    for i, row in df.iterrows():
        for p in pollutants:
            base_val = row[f'{p}_fire_adjusted']
            if pd.isna(base_val): continue
            osm_impact_score = 0
            if not osm_df.empty:
                for j, source in osm_df.iterrows():
                    if dist_matrix_km[i, j] < OSM_INFLUENCE_RADIUS_KM:
                        osm_impact_score += (OSM_IMPACT_FACTORS.get(source['type'], {}).get(p, 1.0) - 1.0) * (
                                    1 - (dist_matrix_km[i, j] / OSM_INFLUENCE_RADIUS_KM))
            osm_factor = min(1 + osm_impact_score, MAX_OSM_ADJUSTMENT_FACTOR)
            traffic_factor = 1 + (row['traffic_congestion'] * TOMTOM_TRAFFIC_IMPACT_WEIGHTS.get(p, 0.0))
            df.loc[i, f'{p}_final'] = base_val * osm_factor * traffic_factor
    return df


def calculate_sub_index(pollutant, concentration):
    if pd.isna(concentration) or concentration < 0: return np.nan
    breakpoints = AQI_CONFIG['breakpoints'].get(pollutant, [])
    if not breakpoints: return np.nan
    for C_low, C_high, I_low, I_high in breakpoints:
        if C_low <= concentration <= C_high:
            return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
    C_low, C_high, I_low, I_high = breakpoints[-1]
    if concentration > C_high:
        return ((I_high - I_low) / (C_high - C_low)) * (concentration - C_low) + I_low
    return np.nan


def calculate_overall_aqi(row):
    sub_indices = [calculate_sub_index(p, row[f'{p}_final']) for p in POLLUTANTS_TO_ANALYZE]
    valid_indices = [i for i in sub_indices if pd.notna(i)]
    if not valid_indices: return pd.Series([np.nan, 'No Data', '#808080'])
    final_aqi = max(valid_indices)
    for aqi_low, aqi_high, category, color in AQI_CONFIG['categories']:
        if aqi_low <= final_aqi <= aqi_high:
            return pd.Series([round(final_aqi), category, color])
    return pd.Series([round(final_aqi), AQI_CONFIG['categories'][-1][2], AQI_CONFIG['categories'][-1][3]])


def create_map(final_df, waqi_df, firms_df, osm_df):
    """Creates the final map with all data layers and detailed pop-ups."""
    print("\nGenerating final interactive AQI map...")
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="CartoDB positron")

    # Neighborhood Layer with AQI colors
    for _, row in final_df.iterrows():
        popup_html = f"<h3>{row['Location']}</h3>"
        popup_html += f"<i>Zone: {row.get('Zone', 'N/A')} | Category: {row.get('Category', 'N/A')}</i><hr>"
        if pd.notna(row['AQI']):
            popup_html += f"<b>AQI: {row['AQI']} ({row['AQI_Category']})</b><hr>"
        popup_html += "<b><u>Final Pollutant Estimates:</u></b><br>" + "".join(
            [f"<b>{p.upper()}:</b> {row[f'{p}_final']:.2f}<br>" for p in POLLUTANTS_TO_ANALYZE if
             f'{p}_final' in row and pd.notna(row[f'{p}_final'])])
        popup_html += "<hr><b><u>Live Model Inputs:</u></b><br>"
        popup_html += f"- Weather: {row.get('temperature', 'N/A'):.1f}°C, {row.get('humidity', 'N/A'):.1f}%, {row.get('wind_speed', 'N/A'):.1f} km/h from {row.get('wind_direction', 'N/A')}°<br>"
        if 'traffic_congestion' in row and pd.notna(row['traffic_congestion']):
            popup_html += f"- Traffic Congestion Index: {row['traffic_congestion']:.2f}<br>"

        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=7,
            color=row['AQI_Color'],
            fill=True,
            fill_color=row['AQI_Color'],
            fill_opacity=0.8,
            popup=folium.Popup(popup_html, max_width=400)
        ).add_to(m)

    # Data Source Layers
    if not waqi_df.empty:
        stations_layer = folium.FeatureGroup(name="Official AQI Stations", show=True).add_to(m)
        for _, s in waqi_df.iterrows():
            station_aqi = s.get('aqi', 'N/A')
            popup_html = f"<h4>Official Monitoring Station</h4><b>Name:</b> {s['station_name']}<br><b>Live Reported AQI:</b> {station_aqi}"
            folium.Marker(
                location=[s['latitude'], s['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='darkblue', icon='broadcast-tower', prefix='fa')
            ).add_to(stations_layer)

    if not firms_df.empty:
        fires_layer = folium.FeatureGroup(name="Fire Hotspots (FIRMS)", show=False).add_to(m)
        for _, f in firms_df.iterrows():
            folium.Marker([f['latitude'], f['longitude']], popup=f"<b>Fire Hotspot</b><br>Intensity: {f['frp']}",
                          icon=folium.Icon(color='orange', icon='fire', prefix='fa')).add_to(fires_layer)

    if not osm_df.empty:
        osm_layer = folium.FeatureGroup(name="Pollution Sources (OSM)", show=False).add_to(m)
        for _, o in osm_df.iterrows():
            folium.CircleMarker([o['latitude'], o['longitude']], radius=4, color='purple', fill=True, fill_opacity=0.6,
                                popup=f"<b>{o['type'].replace('_', ' ').title()}</b><br>{o['name']}").add_to(osm_layer)

    # Add AQI Legend to the map
    legend_html = """<div style="position: fixed; top: 10px; right: 10px; width: 150px; border:2px solid grey; z-index:9999; font-size:14px; background-color:white; padding: 10px;"><b>AQI Legend</b><br>"""
    for _, _, category, color in AQI_CONFIG['categories']:
        legend_html += f'<i style="background:{color}; width: 20px; height: 15px; display: inline-block; margin-right: 5px; vertical-align: middle;"></i> {category}<br>'
    m.get_root().html.add_child(folium.Element(legend_html + "</div>"))

    folium.LayerControl().add_to(m)
    print("✅ Map generation complete.")
    return m


# ==============================================================================
# STEP 3: MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main function to run the entire AQI modeling and mapping pipeline."""
    neighborhood_df = load_neighborhood_data(NEIGHBORHOOD_CSV_PATH)
    if neighborhood_df.empty:
        print(
            "\n❌ Aborting: Neighborhood data could not be loaded. Ensure 'chennai-area.csv' is in the correct folder.")
        return

    # --- Data Fetching ---
    waqi_df = fetch_waqi_data(find_chennai_stations(), POLLUTANTS_TO_ANALYZE)
    cams_df = fetch_cams_data(neighborhood_df, POLLUTANT_API_MAP)
    meteo_df = fetch_meteo_data(neighborhood_df)
    firms_df = fetch_firms_data(NASA_FIRMS_API_KEY, FIRMS_AREA_BOUNDS)
    osm_df = fetch_osm_pollution_sources(OVERPASS_API_URL, OVERPASS_QUERY)
    traffic_df = fetch_tomtom_traffic_data(neighborhood_df, TOMTOM_API_KEY, TOMTOM_TRAFFIC_RADIUS_METERS)

    # --- Data Cleaning & Pre-processing ---
    if not waqi_df.empty:
        waqi_df.dropna(subset=POLLUTANTS_TO_ANALYZE, how='all', inplace=True)
        waqi_df = waqi_df[~waqi_df['station_name'].str.contains("IITM", case=False, na=False)]

    # --- Modeling Pipeline ---
    if not meteo_df.empty:
        neighborhood_df = find_n_nearest_stations(neighborhood_df, waqi_df, N_NEAREST_STATIONS)
        initial_fused_df = calculate_hybrid_idw_values(neighborhood_df, waqi_df, cams_df, POLLUTANTS_TO_ANALYZE)

        # Merge all dataframes for the main calculation
        merged_df = pd.merge(neighborhood_df, initial_fused_df, on="Location")
        df_with_meteo = pd.merge(merged_df, meteo_df, on="Location")

        # Apply sequential adjustments
        df_meteo_adjusted = apply_meteorological_adjustments(df_with_meteo, POLLUTANTS_TO_ANALYZE)
        df_fire_adjusted = apply_fire_adjustments(df_meteo_adjusted, firms_df, POLLUTANTS_TO_ANALYZE)
        final_df = apply_local_source_adjustments(df_fire_adjusted, osm_df, traffic_df, POLLUTANTS_TO_ANALYZE)

        # --- Final AQI Calculation ---
        print("\nFINAL STEP: Calculating AQI for all locations...")
        final_df[['AQI', 'AQI_Category', 'AQI_Color']] = final_df.apply(calculate_overall_aqi, axis=1)
        print("  ✅ AQI calculation complete.")

        # --- Map Generation and Display ---
        final_map = create_map(final_df, waqi_df, firms_df, osm_df)

        # Save map to an HTML file and open it in the browser
        output_filename = "chennai_aqi_map.html"
        final_map.save(output_filename)
        print(f"\n✅ Map successfully generated and saved as '{output_filename}'")

        # Get the full path to the file and open it
        try:
            full_path = 'file://' + os.path.realpath(output_filename)
            webbrowser.open(full_path, new=2)  # new=2 opens in a new tab, if possible
            print("  ✅ Automatically opening the map in your default web browser...")
        except Exception as e:
            print(
                f"  ❌ Could not automatically open the map. Please open '{output_filename}' manually in a browser. Error: {e}")

    else:
        print("\n❌ CRITICAL ERROR: Could not fetch meteorological data. Aborting.")


if __name__ == "__main__":
    main()