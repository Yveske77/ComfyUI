from flask import Flask, jsonify, render_template
import requests

app = Flask(__name__)

# Some common ICAO24 prefixes used by military aircraft
MILITARY_PREFIXES = [
    "ae",  # United States
    "43c", # United Kingdom
    "3f",  # Germany
    "3e",  # Germany
    "e8",  # Japan
    "880", # South Korea
    "894", # India
]

OPEN_SKY_URL = "https://opensky-network.org/api/states/all"

def is_military(icao24):
    icao24 = icao24.lower()
    return any(icao24.startswith(prefix) for prefix in MILITARY_PREFIXES)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/military-flights')
def military_flights():
    try:
        resp = requests.get(OPEN_SKY_URL, params={"extended": "1"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    states = data.get('states', [])
    flights = []
    for s in states:
        if not s:
            continue
        icao24 = s[0]
        if is_military(icao24):
            flight = {
                "icao24": icao24,
                "callsign": s[1].strip() if s[1] else "",
                "origin_country": s[2],
                "time_position": s[3],
                "longitude": s[5],
                "latitude": s[6],
                "baro_altitude": s[7],
                "on_ground": s[8],
                "velocity": s[9],
            }
            if flight["latitude"] is not None and flight["longitude"] is not None:
                flights.append(flight)
    return jsonify({"flights": flights})

if __name__ == '__main__':
    app.run(debug=True)
