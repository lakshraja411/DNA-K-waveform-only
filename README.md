# Minimal whole-waveform K-means Streamlit app

This stripped-down app contains only:
1. Upload dataset.npz, event_data.npz and event_fitting.npz
2. Whole-waveform K-means clustering
3. Silhouette-based k selection
4. Blockade-vs-dwell scatter
5. Midpoint-aligned waveform family plot

For speed on large recordings:
- K-means always fits all matched events.
- By default, silhouette is evaluated on a reproducible 3000-event sample.
- Set silhouette sample size to 0 for the full exact silhouette (slower).
- The k-scan defaults to 10 initialisations; the final selected k is refit with 100.

Deploy with app.py as the Streamlit entrypoint.
