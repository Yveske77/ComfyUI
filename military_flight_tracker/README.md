# Military Flight Tracker

This is a simple Flask-based web application that displays military flights on an interactive map. It queries the [OpenSky Network](https://opensky-network.org/) API and filters aircraft using a list of common military ICAO24 prefixes. The data is refreshed every 30 seconds and visualized using [Leaflet](https://leafletjs.com/).

## Setup

1. Create a Python virtual environment and install dependencies:

```bash
pip install Flask requests
```

2. Run the application:

```bash
python app.py
```

3. Open your browser at [http://localhost:5000](http://localhost:5000) to see the map.

The application fetches live flight data from OpenSky Network, so network access is required.
