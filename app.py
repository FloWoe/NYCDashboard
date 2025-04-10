import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import branca.colormap as cm
from streamlit_folium import st_folium
from streamlit_option_menu import option_menu
from shapely.geometry import Point
import matplotlib.pyplot as plt
import numpy as np
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

selected_zipcode = None
base_data = pd.read_csv("immobilien_mit_koordinaten.csv")
base_data_new = pd.read_csv("immobilien_mit_koordinaten_new.csv")

# Konvertiere ZIP CODE zu int für einfachere Vergleiche
base_data_new[' ZIP CODE'] = pd.to_numeric(base_data_new[' ZIP CODE'], errors='coerce')

# Konvertiere numerische Spalten richtig
numeric_columns = ['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS', 
                   'LAND SQUARE FEET', 'GROSS SQUARE FEET', 'YEAR BUILT', 
                   'SALE PRICE', 'PRICE_PER_SQFT']

for col in numeric_columns:
    if col in base_data_new.columns:
        # Ersetze Kommas und entferne führende/nachfolgende Leerzeichen
        if base_data_new[col].dtype == 'object':
            base_data_new[col] = base_data_new[col].astype(str).str.replace(',', '')
        # Konvertiere zu numerisch, ersetze Fehler mit NaN
        base_data_new[col] = pd.to_numeric(base_data_new[col], errors='coerce')

def create_folium_map(map_df):
    preis_data = map_df
    shapefile_path = "ShapeFiles/nyc_zcta520.shp"
    nyc_zip = gpd.read_file(shapefile_path)

    if 'ZIPCODE' not in nyc_zip.columns:
        if 'ZCTA5CE20' in nyc_zip.columns:
            nyc_zip['ZIPCODE'] = nyc_zip['ZCTA5CE20'].astype(int)
        else:
            raise ValueError("Keine geeignete ZIPCODE-Spalte gefunden")
    nyc_zip['ZIPCODE'] = nyc_zip['ZIPCODE'].astype(int)
    zip_codes = preis_data['ZIPCODE'].tolist()
    nyc_zip_filtered = nyc_zip[nyc_zip['ZIPCODE'].isin(zip_codes)]
    nyc_zip_prices = nyc_zip_filtered.merge(preis_data, on='ZIPCODE')
    if nyc_zip_prices.crs and nyc_zip_prices.crs != "EPSG:4326":
        nyc_zip_prices = nyc_zip_prices.to_crs(epsg=4326)
    center_lat = nyc_zip_prices.geometry.centroid.y.mean()
    center_lon = nyc_zip_prices.geometry.centroid.x.mean()
    m = folium.Map(location=[center_lat, center_lon],
                    zoom_start=11,
                    tiles='CartoDB positron')
    min_price = nyc_zip_prices['PRICE_PER_SQFT'].min()
    max_price = nyc_zip_prices['PRICE_PER_SQFT'].max()
    colormap = cm.LinearColormap(
        colors=['green', 'yellow', 'orange', 'red'],
        vmin=min_price,
        vmax=max_price
    )
    colormap.caption = 'Preis pro sqft (USD)'
    def style_function(feature):
        price = feature['properties']['PRICE_PER_SQFT']
        return {
            'fillColor': colormap(price),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.7
        }
    def highlight_function(feature):
        return {
            'fillColor': colormap(feature['properties']['PRICE_PER_SQFT']),
            'color': 'white',
            'weight': 3,
            'fillOpacity': 0.9
        }
    folium.GeoJson(
        nyc_zip_prices,
        name='Immobilienpreise',
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=['ZIPCODE', 'PRICE_PER_SQFT'],
            aliases=['PLZ:', 'Preis pro sqft ($):'],
            localize=False,
            sticky=True,
            labels=True
        ),
    ).add_to(m)
    colormap.add_to(m)
    return m, nyc_zip_prices

def get_zipcode_stats(zipcode):
    """Liefert detaillierte Statistiken für eine bestimmte PLZ"""
    # Filtern der Daten nach PLZ
    zip_data = base_data_new[base_data_new[' ZIP CODE'] == zipcode]
    
    if zip_data.empty:
        return None
    
    # Hilfsfunktion für sichere Mittelwertberechnung
    def safe_mean(series):
        # Entferne NaN-Werte und berechne den Mittelwert
        values = series.dropna()
        if len(values) > 0:
            return values.mean()
        return 0
    
    # Hilfsfunktion für sichere Min/Max-Berechnung
    def safe_min(series):
        values = series.dropna()
        if len(values) > 0:
            return values.min()
        return 0
    
    def safe_max(series):
        values = series.dropna()
        if len(values) > 0:
            return values.max()
        return 0
    
    def safe_median(series):
        values = series.dropna()
        if len(values) > 0:
            return values.median()
        return 0
    
    # Grundlegende Statistiken mit Fehlerbehandlung
    stats = {
        'count': len(zip_data),
        'avg_price': safe_mean(zip_data['SALE PRICE']),
        'median_price': safe_median(zip_data['SALE PRICE']),
        'min_price': safe_min(zip_data['SALE PRICE']),
        'max_price': safe_max(zip_data['SALE PRICE']),
        'avg_price_per_sqft': safe_mean(zip_data['PRICE_PER_SQFT']),
        'median_price_per_sqft': safe_median(zip_data['PRICE_PER_SQFT']),
        'avg_sqft': safe_mean(zip_data['GROSS SQUARE FEET']),
        'avg_land_sqft': safe_mean(zip_data['LAND SQUARE FEET']),
        'avg_units': safe_mean(zip_data['TOTAL UNITS']),
        'avg_res_units': safe_mean(zip_data['RESIDENTIAL UNITS']),
        'avg_com_units': safe_mean(zip_data['COMMERCIAL UNITS']),
        'neighborhoods': zip_data['NEIGHBORHOOD'].unique().tolist(),
        'building_classes': zip_data['BUILDING CLASS CATEGORY'].value_counts().to_dict()
    }
    
    # Verkaufstrend (letzten 12 Monate)
    if 'SALE DATE' in zip_data.columns:
        # Konvertiere das Datum
        try:
            zip_data['SALE DATE'] = pd.to_datetime(zip_data['SALE DATE'])
            # Sortiere nach Datum
            sales_by_month = zip_data.set_index('SALE DATE').sort_index()
            # Gruppiere nach Monat mit Durchschnittspreisen
            monthly_avg = sales_by_month.resample('M')['SALE PRICE'].mean().dropna().tail(12)
            stats['monthly_trend'] = monthly_avg.to_dict()
        except:
            stats['monthly_trend'] = {}
    
    return stats

def create_zipcode_viz(zipcode):
    """Erstellt Visualisierungen für eine bestimmte PLZ"""
    zip_data = base_data_new[base_data_new[' ZIP CODE'] == zipcode]
    
    if zip_data.empty:
        return None
    
    # Daten für Visualisierungen bereinigen
    # Entferne NaN-Werte für jede Visualisierung separat
    try:
        # Erstelle ein Figure-Objekt mit 2 Unterplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Gebäudeklassenverteilung
        if 'BUILDING CLASS CATEGORY' in zip_data.columns and not zip_data['BUILDING CLASS CATEGORY'].isna().all():
            building_counts = zip_data['BUILDING CLASS CATEGORY'].value_counts().head(5)
            axes[0, 0].bar(building_counts.index, building_counts.values)
            axes[0, 0].set_title('Top 5 Gebäudeklassen')
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            axes[0, 0].set_title('Top 5 Gebäudeklassen')
        
        # Plot 2: Verhältnis Preis zu Quadratmetern
        valid_data = zip_data.dropna(subset=['GROSS SQUARE FEET', 'SALE PRICE'])
        if not valid_data.empty:
            axes[0, 1].scatter(valid_data['GROSS SQUARE FEET'], valid_data['SALE PRICE'])
            axes[0, 1].set_title('Preis vs. Fläche')
            axes[0, 1].set_xlabel('Bruttofläche (sqft)')
            axes[0, 1].set_ylabel('Verkaufspreis ($)')
            axes[0, 1].ticklabel_format(style='plain', axis='y')
        else:
            axes[0, 1].text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            axes[0, 1].set_title('Preis vs. Fläche')
        
        # Plot 3: Preis/Quadratmeter-Verteilung
        valid_price_data = zip_data['PRICE_PER_SQFT'].dropna()
        if not valid_price_data.empty:
            axes[1, 0].hist(valid_price_data, bins=min(20, len(valid_price_data)), alpha=0.7)
            axes[1, 0].set_title('Verteilung der Preise pro sqft')
            axes[1, 0].set_xlabel('Preis pro sqft ($)')
            axes[1, 0].set_ylabel('Anzahl')
        else:
            axes[1, 0].text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            axes[1, 0].set_title('Verteilung der Preise pro sqft')
        
        # Plot 4: Baujahr vs. Preis
        valid_year_data = zip_data.dropna(subset=['YEAR BUILT', 'PRICE_PER_SQFT'])
        if not valid_year_data.empty:
            axes[1, 1].scatter(valid_year_data['YEAR BUILT'], valid_year_data['PRICE_PER_SQFT'])
            axes[1, 1].set_title('Baujahr vs. Preis pro sqft')
            axes[1, 1].set_xlabel('Baujahr')
            axes[1, 1].set_ylabel('Preis pro sqft ($)')
        else:
            axes[1, 1].text(0.5, 0.5, 'Keine Daten verfügbar', ha='center', va='center')
            axes[1, 1].set_title('Baujahr vs. Preis pro sqft')
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        # Bei Fehler eine einfache Fehlermeldung zurückgeben
        print(f"Fehler bei der Visualisierung: {str(e)}")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Fehler bei der Visualisierung: {str(e)}", ha='center', va='center')
        return fig

def interactive_section():
    map_df = pd.read_csv("average_price_per_sqft_sorted.csv")
    global selected_zipcode

    m, zip_price_data = create_folium_map(map_df)
    map_col, details_col = st.columns([3, 1])

    with map_col:
        map_data = st_folium(m, width="100%", height=1350)

    if map_data.get("last_clicked"):
        click_lat = map_data["last_clicked"]["lat"]
        click_lng = map_data["last_clicked"]["lng"]
        click_point = Point(click_lng, click_lat)
        zip_price_data['distance'] = zip_price_data.geometry.distance(click_point)
        nearest_zip = zip_price_data.loc[zip_price_data['distance'].idxmin()]
        selected_zipcode = nearest_zip['ZIPCODE']

    with details_col:
        st.subheader("PLZ-Details")

        if selected_zipcode:
            st.write(f"Ausgewählte PLZ: **{selected_zipcode}**")
            selected_data = zip_price_data[zip_price_data['ZIPCODE'] == selected_zipcode].iloc[0]
            price = selected_data['PRICE_PER_SQFT']
            st.write(f"Durchschnittlicher Preis pro sqft: **${price:.2f}**")
            avg_price = zip_price_data['PRICE_PER_SQFT'].mean()
            diff_percent = ((price - avg_price) / avg_price) * 100
            st.write(f"Verglichen mit dem Durchschnitt: **{diff_percent:.1f}%**")
            st.write("---")
            
            # Erweiterte Statistiken aus base_data_new
            stats = get_zipcode_stats(selected_zipcode)
            
            if stats:
                # Tabs für verschiedene Informationskategorien
                tab1, tab2 = st.tabs(["Übersicht", "Statistiken"])
                
                with tab1:
                    # Übersicht
                    st.subheader("Übersicht")
                    st.write(f"Anzahl der Immobilien: **{stats['count']}**")
                    st.write(f"Durchschnittspreis: **${stats['avg_price']:,.2f}**")
                    st.write(f"Median Preis: **${stats['median_price']:,.2f}**")
                    st.write(f"Preisspanne: **\\${stats['min_price']:,.2f}**" + " bis " + f"**\\${stats['max_price']:,.2f}**")
                    
                    # Nachbarschaften
                    st.subheader("Stadtteile in diesem zipcode")
                    if stats['neighborhoods']:
                        # Entferne Duplikate mit einer speziellen Normalisierungsfunktion
                        def normalize_string(s):
                            # Entferne zusätzliche Leerzeichen und konvertiere zu Kleinbuchstaben für den Vergleich
                            return ' '.join(s.strip().split()).lower()

                        # Erstelle ein Dictionary mit normalisierten Schlüsseln und ursprünglichen Werten
                        unique_neighborhoods = {}
                        for n in stats['neighborhoods']:
                            normalized = normalize_string(n)
                            if normalized not in unique_neighborhoods:
                                unique_neighborhoods[normalized] = n

                        # Sortiere und zeige die ursprünglichen Werte an
                        for n in sorted(unique_neighborhoods.values()):
                            st.write(f"- {n}")
                    else:
                        st.write("Keine Stadtteile gefunden.")
                
                with tab2:
                    # Detaillierte Statistiken
                    st.subheader("Immobilien-Statistiken")
                    st.write(f"Durchschnittliche Wohneinheiten: **{stats['avg_res_units']:.1f}**")
                    st.write(f"Durchschnittliche Gewerbeeinheiten: **{stats['avg_com_units']:.1f}**")
                    st.write(f"Durchschnittliche Gesamteinheiten: **{stats['avg_units']:.1f}**")
                    st.write(f"Durchschnittliche Grundstücksfläche: **{stats['avg_land_sqft']:.1f} sqft**")
                    st.write(f"Durchschnittliche Bruttofläche: **{stats['avg_sqft']:.1f} sqft**")
                    
                    # Gebäudeklassen
                    st.subheader("Häufigste Gebäudeklassen")
                    for class_name, count in sorted(stats['building_classes'].items(), 
                                                 key=lambda x: x[1], reverse=True)[:5]:
                        st.write(f"- {class_name}: {count}")
                
            
            else:
                st.write("Keine detaillierten Daten für diese PLZ verfügbar.")
                
            # Ursprüngliches Diagramm
            fig, ax = plt.subplots(figsize=(6, 6))
            metrics = ['Min', 'Ausgewählt', 'Durchschnitt', 'Max']
            values = [
                zip_price_data['PRICE_PER_SQFT'].min(),
                price,
                avg_price,
                zip_price_data['PRICE_PER_SQFT'].max()
            ]
            colors = ['green', 'blue', 'orange', 'red']
            ax.bar(metrics, values, color=colors)
            ax.set_ylabel('Preis pro m² ($)')
            ax.set_title('Preisvergleich')
            st.pyplot(fig)

        else:
            st.write("Klicken Sie auf die Karte, um eine PLZ auszuwählen.")

def inference_section():
    global base_data_new
    
    st.title("Price Prediction")
    
    # Eingabefelder für die Standortdaten
    st.subheader("Standortinformationen")
    borough = st.number_input("BOROUGH (1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island):", 
                             min_value=1, max_value=5, value=1)
    col1, col2 = st.columns(2)
    with col1:
        block = st.number_input("BLOCK:", min_value=1, step=1, value=100)
    with col2:
        lot = st.number_input("LOT:", min_value=1, step=1, value=25)
    
    # Eingabefelder für die Immobilieneigenschaften
    st.subheader("Immobilieneigenschaften")
    col3, col4 = st.columns(2)
    
    with col3:
        residential_units = st.number_input("RESIDENTIAL UNITS:", min_value=0, step=1, value=1)
        commercial_units = st.number_input("COMMERCIAL UNITS:", min_value=0, step=1, value=0)
        total_units = st.number_input("TOTAL UNITS:", min_value=1, step=1, value=1)
    
    with col4:
        land_sqft = st.number_input("LAND SQUARE FEET:", min_value=0, step=100, value=2000)
        gross_sqft = st.number_input("GROSS SQUARE FEET:", min_value=0, step=100, value=1500)
        year_built = st.number_input("YEAR BUILT:", min_value=1800, max_value=2025, step=1, value=2000)
    
    if st.button("Daten absenden"):
        try:
            # Modell laden falls nötig
            if 'model' not in globals() or model is None:
                import joblib
                model = joblib.load('nyc_borough_model.pkl')
            
            # Preis vorhersagen
            import pandas as pd
            import numpy as np
            
            # Dataframe mit einem Eintrag erstellen
            input_data = pd.DataFrame({
                'BOROUGH': [borough],
                'BLOCK': [block],
                'LOT': [lot],
                'RESIDENTIAL UNITS': [residential_units],
                'COMMERCIAL UNITS': [commercial_units],
                'TOTAL UNITS': [total_units],
                'LAND SQUARE FEET': [land_sqft],
                'GROSS SQUARE FEET': [gross_sqft],
                'YEAR BUILT': [year_built]
            })
            
            # Preis vorhersagen
            predicted_price_log = model.predict(input_data)[0]
            
            # Zurücktransformation des logarithmierten Preises
            predicted_price = np.expm1(predicted_price_log)
            
            # Preis mit großem Text anzeigen
            st.markdown(f"<h2 style='text-align: center; color: green;'>Vorhergesagter Preis: ${predicted_price:,.2f}</h2>", unsafe_allow_html=True)
            
            # Zusatzinformationen anzeigen
            borough_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
            borough_name = borough_names.get(borough, f"Borough {borough}")
            
            st.write(f"Standort: {borough_name}, Block {block}, Lot {lot}")
            st.write(f"Grundstücksfläche: {land_sqft} sq ft")
            st.write(f"Gebäudefläche: {gross_sqft} sq ft")
            st.write(f"Einheiten: {total_units} ({residential_units} Wohn, {commercial_units} Gewerbe)")
            st.write(f"Baujahr: {year_built}")
            
        except Exception as e:
            st.error(f"Fehler: {str(e)}")
            import traceback
            st.text(traceback.format_exc())

def show_zipcode_visualizations():
    global selected_zipcode
    zipcode = selected_zipcode
   
    if not zipcode:
        st.write("Bitte wählen Sie zuerst eine PLZ aus.")
        return
    
    # Headline des Abschnitts
    st.subheader(f"Visualisierungen für PLZ {zipcode}")
    
    # Visualisierungen erstellen
    fig = create_zipcode_viz(zipcode)
    if fig:
        st.pyplot(fig)
    else:
        st.write("Keine ausreichenden Daten für Visualisierungen vorhanden.")
    
    # Statistiken abrufen, um den Preistrend anzuzeigen
    stats = get_zipcode_stats(zipcode)
    
    # Preistrend wenn verfügbar
    if stats and 'monthly_trend' in stats and len(stats['monthly_trend']) > 1:
        st.subheader("Preistrend (letzte Monate)")
        trend_data = pd.Series(stats['monthly_trend'])
        trend_fig, trend_ax = plt.subplots(figsize=(8, 4))
        trend_ax.plot(trend_data.index, trend_data.values, marker='o')
        trend_ax.set_title(f'Preistrend für PLZ {zipcode}')
        trend_ax.set_ylabel('Durchschnittspreis ($)')
        trend_ax.ticklabel_format(style='plain', axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(trend_fig)
    
    # Optional: Zusätzliche Visualisierungen je nach Bedarf
    
    # Beispiel für eine weitere Visualisierung: Top Nachbarschaften in dieser PLZ
    # In der Funktion show_zipcode_visualizations() 
    if stats and 'neighborhoods' in stats and len(stats['neighborhoods']) > 0:
        zip_data = base_data_new[base_data_new[' ZIP CODE'] == zipcode]
        if not zip_data.empty and 'NEIGHBORHOOD' in zip_data.columns:
            try:
                neighborhood_counts = zip_data['NEIGHBORHOOD'].value_counts().head(10)
                if len(neighborhood_counts) > 1:  # Nur anzeigen, wenn es mehr als eine Nachbarschaft gibt
                    st.subheader("Top Nachbarschaften in dieser PLZ")
                    nh_fig, nh_ax = plt.subplots(figsize=(8, 4))
                    neighborhood_counts.plot(kind='bar', ax=nh_ax)
                    nh_ax.set_title('Anzahl der Immobilien nach Nachbarschaft')
                    nh_ax.set_ylabel('Anzahl')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(nh_fig)
            except Exception as e:
                st.error(f"Fehler bei der Nachbarschaftsdarstellung: {str(e)}")

    # Ersetze ihn mit diesem verbesserten Code:

    if stats and 'neighborhoods' in stats and len(stats['neighborhoods']) > 0:
        zip_data = base_data_new[base_data_new[' ZIP CODE'] == zipcode]
        if not zip_data.empty and 'NEIGHBORHOOD' in zip_data.columns:
            try:
                # Normalisiere die Nachbarschaftsnamen für konsistente Zählung
                def normalize_string(s):
                    return ' '.join(s.strip().split()).lower()

                # Erstelle eine kopie und normalisiere die Nachbarschaftsspalte
                zip_data_norm = zip_data.copy()
                zip_data_norm['NEIGHBORHOOD_NORM'] = zip_data['NEIGHBORHOOD'].apply(normalize_string)

                # Verwende die normalisierte Spalte für die Zählung
                neighborhood_counts_norm = zip_data_norm['NEIGHBORHOOD_NORM'].value_counts().head(10)

                # Mappe zurück zu einem der ursprünglichen Namen für die Anzeige
                name_mapping = {}
                for idx, row in zip_data_norm.iterrows():
                    name_mapping[row['NEIGHBORHOOD_NORM']] = row['NEIGHBORHOOD']

                # Erstelle das Series-Objekt mit ursprünglichen Namen aber deduplizierten Counts
                neighborhood_counts = pd.Series(neighborhood_counts_norm.values,
                                              index=[name_mapping[idx] for idx in neighborhood_counts_norm.index])

                if len(neighborhood_counts) > 1:  # Nur anzeigen, wenn es mehr als eine Nachbarschaft gibt
                    st.subheader("Top Nachbarschaften in dieser PLZ")
                    nh_fig, nh_ax = plt.subplots(figsize=(8, 4))
                    neighborhood_counts.plot(kind='bar', ax=nh_ax)
                    nh_ax.set_title('Anzahl der Immobilien nach Nachbarschaft')
                    nh_ax.set_ylabel('Anzahl')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(nh_fig)
            except Exception as e:
                st.error(f"Fehler bei der Nachbarschaftsdarstellung: {str(e)}")

    # Weitere Visualisierung: Preisverteilung nach Gebäudeklasse
    if stats and stats.get('building_classes'):
        zip_data = base_data_new[base_data_new[' ZIP CODE'] == zipcode]
        if not zip_data.empty:
            try:
                # Gruppieren nach Gebäudeklasse und Berechnung des durchschnittlichen Preises
                valid_data = zip_data.dropna(subset=['BUILDING CLASS CATEGORY', 'SALE PRICE'])
                
                if not valid_data.empty:
                    # Nur die häufigsten Gebäudeklassen (Top 5) berücksichtigen
                    top_classes = valid_data['BUILDING CLASS CATEGORY'].value_counts().head(5).index
                    filtered_data = valid_data[valid_data['BUILDING CLASS CATEGORY'].isin(top_classes)]
                    
                    if not filtered_data.empty:
                        class_price_avg = filtered_data.groupby('BUILDING CLASS CATEGORY')['SALE PRICE'].mean().sort_values(ascending=False)
                        
                        st.subheader("Durchschnittspreis nach Gebäudeklasse")
                        class_fig, class_ax = plt.subplots(figsize=(8, 4))
                        class_price_avg.plot(kind='bar', ax=class_ax)
                        class_ax.set_title('Durchschnittlicher Verkaufspreis nach Gebäudeklasse')
                        class_ax.set_ylabel('Durchschnittspreis ($)')
                        class_ax.ticklabel_format(style='plain', axis='y')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(class_fig)
            except Exception as e:
                st.error(f"Fehler bei der Preisverteilung nach Gebäudeklasse: {str(e)}")

def render_real_estate_analysis(container=None):
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    from datetime import datetime, timedelta

    global base_data_new
    df = base_data_new.copy()
    
    # Wenn kein Container übergeben wurde, verwende st direkt
    if container is None:
        container = st
    
    # Prüfe, ob alle erforderlichen Spalten vorhanden sind
    required_columns = ['SALE PRICE', 'SALE DATE', 'BOROUGH', 'NEIGHBORHOOD', 
                        'GROSS SQUARE FEET', 'PRICE_PER_SQFT']
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        container.error(f"Fehlende Spalten im DataFrame: {', '.join(missing_columns)}")
        return df

    # Stelle sicher, dass SALE DATE im Datetime-Format ist
    if not pd.api.types.is_datetime64_any_dtype(df['SALE DATE']):
        try:
            # Versuche verschiedene Formate und Optionen für die Konvertierung
            df['SALE DATE'] = pd.to_datetime(df['SALE DATE'], 
                                            format='mixed',  # Erlaubt gemischte Formate
                                            errors='coerce', # Konvertiert problematische Werte zu NaT
                                            infer_datetime_format=True)  # Versucht das Format zu erkennen

            # Entferne Zeilen mit ungültigen Datumswerten automatisch
            nat_count = df['SALE DATE'].isna().sum()
            if nat_count > 0:
                # Entferne Zeilen mit NaT-Werten stillschweigend
                df = df.dropna(subset=['SALE DATE'])
            
            # Überprüfe auf NaT-Werte (Not a Time) und informiere den Benutzer
            nat_count = df['SALE DATE'].isna().sum()
            if nat_count > 0:
                container.warning(f"{nat_count} Datumswerte konnten nicht konvertiert werden und wurden als NaT markiert.")
                
            # Entferne Zeilen mit NaT-Werten, wenn gewünscht
            # df = df.dropna(subset=['SALE DATE'])
        except Exception as e:
            container.error(f"Fehler beim Konvertieren von 'SALE DATE' zu Datetime: {e}")
            return df
    
    # ======= ZEITRAUMAUSWAHL =======
    container.header("Datenanalyse")
    container.subheader("Zeitraumauswahl")

    # Zeitraumoptionen
    time_options = {
        "Gesamter Zeitraum": None,
        "Dieses Jahr": "year",
        "Dieser Monat": "month",
        "Letzte 30 Tage": "30days",
        "Letzte 90 Tage": "90days",
        "Letzte 6 Monate": "6months",
        "Letzte 12 Monate": "12months",
        "Benutzerdefiniert": "custom"
    }

    # Zeitraumauswahl
    col1, col2 = container.columns([1, 2])
    with col1:
        selected_time_option = st.selectbox(
            "Zeitraum wählen:",
            options=list(time_options.keys()),
            index=0
        )

    # Definiere Startdatum und Enddatum basierend auf der Auswahl
    now = datetime(2017, 11, 13, 00, 00, 00, 00)
    start_date = None
    end_date = now

    if time_options[selected_time_option] == "year":
        start_date = datetime(now.year, 1, 1)
    elif time_options[selected_time_option] == "month":
        start_date = datetime(now.year, now.month, 1)
    elif time_options[selected_time_option] == "30days":
        start_date = now - timedelta(days=30)
    elif time_options[selected_time_option] == "90days":
        start_date = now - timedelta(days=90)
    elif time_options[selected_time_option] == "6months":
        start_date = now - timedelta(days=180)
    elif time_options[selected_time_option] == "12months":
        start_date = now - timedelta(days=365)
    elif time_options[selected_time_option] == "custom":
        with col2:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                start_date = st.date_input("Startdatum", min(df['SALE DATE']).date())
            with date_col2:
                end_date = st.date_input("Enddatum", now)
            
            start_date = datetime.combine(start_date, datetime.min.time())
            end_date = datetime.combine(end_date, datetime.max.time())

    # Filtere die Daten nach dem ausgewählten Zeitraum
    if start_date:
        filtered_df = df[(df['SALE DATE'] >= pd.Timestamp(start_date)) & 
                         (df['SALE DATE'] <= pd.Timestamp(end_date))]
    else:
        filtered_df = df

    container.markdown("---")

    # ======= KENNZAHLEN =======
    container.subheader("Wichtige Kennzahlen")

    # Berechne Kennzahlen
    total_sales_value = filtered_df['SALE PRICE'].sum()
    total_transactions = len(filtered_df)
    most_valuable_sale = filtered_df['SALE PRICE'].max()

    # Finde das Jahr mit den meisten Verkäufen
    if not filtered_df.empty:
        filtered_df['YEAR'] = filtered_df['SALE DATE'].dt.year
        sales_by_year = filtered_df.groupby('YEAR').size()
        most_sales_year = sales_by_year.idxmax() if not sales_by_year.empty else "N/A"
        most_sales_count = sales_by_year.max() if not sales_by_year.empty else 0
    else:
        most_sales_year = "N/A"
        most_sales_count = 0

    # Zeige Kennzahlen in Karten an
    col1, col2, col3, col4 = container.columns(4)

    with col1:
        st.metric(
            label="Gesamtumsatz",
            value=f"${total_sales_value:,.0f}"
        )

    with col2:
        st.metric(
            label="Anzahl Verkäufe",
            value=f"{total_transactions:,}"
        )

    with col3:
        st.metric(
            label="Wertvollster Verkauf",
            value=f"${most_valuable_sale:,.0f}"
        )

    with col4:
        st.metric(
            label=f"Meiste Verkäufe im Jahr {most_sales_year}",
            value=f"{most_sales_count:,}"
        )

    container.markdown("---")

    # ======= UMSATZ PRO ZEITRAUM =======
    container.subheader("Umsatz pro Zeitraum")

    # Zeitauflösung für die Umsatzdarstellung
    time_resolution_options = {
        "Monatlich": "M",
        "Vierteljährlich": "Q",
        "Jährlich": "Y",
        "Wöchentlich": "W"
    }

    time_resolution = container.selectbox(
        "Zeitauflösung wählen:",
        options=list(time_resolution_options.keys()),
        index=0,
        key="time_resolution_real_estate"  # Eindeutiger Key für multiple Instanzen
    )

    # Gruppiere die Daten nach Zeitauflösung
    if not filtered_df.empty:
        # Kopie erstellen, um Warnung über Kopie eines Slices zu vermeiden
        temp_df = filtered_df.copy()
        temp_df['PERIOD'] = temp_df['SALE DATE'].dt.to_period(time_resolution_options[time_resolution])
        
        # Berechne Umsatz pro Zeitraum
        sales_by_period = temp_df.groupby('PERIOD').agg({
            'SALE PRICE': 'sum',
            'SALE DATE': 'count'
        }).rename(columns={'SALE DATE': 'COUNT'})
        
        # Konvertiere Period zu String für die Anzeige
        sales_by_period = sales_by_period.reset_index()
        sales_by_period['PERIOD'] = sales_by_period['PERIOD'].astype(str)
        
        # Erstelle Visualisierungen
        fig = px.bar(
            sales_by_period,
            x='PERIOD',
            y='SALE PRICE',
            title=f"Umsatz pro {time_resolution}",
            labels={'PERIOD': 'Zeitraum', 'SALE PRICE': 'Umsatz ($)'},
            text_auto='.2s'
        )
        
        # Füge Linie für Anzahl der Verkäufe hinzu
        fig.add_trace(
            go.Scatter(
                x=sales_by_period['PERIOD'],
                y=sales_by_period['COUNT'],
                mode='lines+markers',
                name='Anzahl Verkäufe',
                yaxis='y2'
            )
        )
        
        # Konfiguriere die zweite Y-Achse
        fig.update_layout(
            yaxis2=dict(
                title='Anzahl Verkäufe',
                overlaying='y',
                side='right'
            ),
            height=500
        )
        
        container.plotly_chart(fig, use_container_width=True)
    else:
        container.info("Keine Daten für den ausgewählten Zeitraum verfügbar.")

    # ======= WEITERE ANALYSEN =======
    col1, col2 = container.columns(2)

    with col1:
        col1.subheader("Top 5 Neighborhoods nach Umsatz")
        if not filtered_df.empty:
            neighborhood_sales = filtered_df.groupby('NEIGHBORHOOD').agg({
                'SALE PRICE': 'sum',
                'SALE DATE': 'count'
            }).rename(columns={'SALE DATE': 'Anzahl'}).sort_values('SALE PRICE', ascending=False).head(5)
            
            fig = px.bar(
                neighborhood_sales.reset_index(),
                x='NEIGHBORHOOD',
                y='SALE PRICE',
                labels={'NEIGHBORHOOD': 'Neighborhood', 'SALE PRICE': 'Umsatz ($)'},
                text_auto='.2s'
            )
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col1.info("Keine Daten für den ausgewählten Zeitraum verfügbar.")

    with col2:
        col2.subheader("Durchschnittlicher Preis pro Quadratfuß nach Borough")
        if not filtered_df.empty:
            # Borough-Mapping erstellen (Zahlencode zu Namen)
            borough_mapping = {
                1: "Manhattan",
                2: "Bronx",
                3: "Brooklyn", 
                4: "Queens",
                5: "Staten Island",
                "1": "Manhattan",
                "2": "Bronx",
                "3": "Brooklyn", 
                "4": "Queens",
                "5": "Staten Island"
            }
            
            # Kopie erstellen, um Warnung zu vermeiden
            temp_df = filtered_df.copy()
            
            # Stelle sicher, dass PRICE_PER_SQFT numerisch ist und keine NaN-Werte hat
            temp_df['PRICE_PER_SQFT'] = pd.to_numeric(temp_df['PRICE_PER_SQFT'], errors='coerce')
            temp_df = temp_df.dropna(subset=['PRICE_PER_SQFT', 'BOROUGH'])
            
            # Borough-Codes in Namen umwandeln
            temp_df['BOROUGH_NAME'] = temp_df['BOROUGH'].map(lambda x: borough_mapping.get(x, f"Unbekannt ({x})"))
            
            # Daten nach dem Namen gruppieren
            borough_price_per_sqft = temp_df.groupby('BOROUGH_NAME').agg({
                'PRICE_PER_SQFT': 'mean'
            }).sort_values('PRICE_PER_SQFT', ascending=False)
            
            if not borough_price_per_sqft.empty:
                fig = px.bar(
                    borough_price_per_sqft.reset_index(),
                    x='BOROUGH_NAME',
                    y='PRICE_PER_SQFT',
                    title="Durchschnittlicher Preis pro Quadratfuß nach Borough",
                    labels={'BOROUGH_NAME': 'Borough', 'PRICE_PER_SQFT': 'Preis/sqft ($)'},
                    text_auto='.2f'
                )
                
                # Forciere die Y-Achse, bei 0 zu beginnen
                fig.update_layout(yaxis_range=[0, None])
                
                col2.plotly_chart(fig, use_container_width=True)
            else:
                col2.info("Keine gültigen Daten für die Gruppierung gefunden.")
        else:
            col2.info("Keine Daten für den ausgewählten Zeitraum verfügbar.")

    return filtered_df

def plot_building_age_distribution(df):
    """
    Erstellt ein Histogramm zur Altersverteilung der verkauften Gebäude.
    """
    try:
        # YEAR BUILT in numerischen Wert umwandeln
        df_clean = df.copy()
        # Prüfen und korrigieren der Spaltenbezeichnungen (ggf. mit führenden Leerzeichen)
        year_col = 'YEAR BUILT' if 'YEAR BUILT' in df_clean.columns else ' YEAR BUILT'
        
        # Konvertierung mit Fehlerbehandlung
        df_clean['YEAR BUILT'] = pd.to_numeric(df_clean[year_col], errors='coerce')
        
        # NaN-Werte filtern
        df_clean = df_clean.dropna(subset=['YEAR BUILT'])
        
        # Prüfen ob noch Daten vorhanden sind
        if len(df_clean) == 0:
            st.error("Keine gültigen Daten für die Visualisierung verfügbar")
            # Leeres Diagramm zurückgeben
            fig = go.Figure()
            fig.update_layout(
                title="Keine Daten verfügbar",
                xaxis_title="Gebäudealter (Jahre)",
                yaxis_title="Anzahl der Verkäufe"
            )
            return fig
        
        # Aktuelles Jahr für Altersberechnung
        current_year = datetime.now().year
        
        # Alter berechnen
        df_clean['BUILDING_AGE'] = current_year - df_clean['YEAR BUILT']
        
        # Entfernen von unrealistischen Altersangaben (z.B. negative Werte oder Gebäude älter als 400 Jahre)
        df_clean = df_clean[(df_clean['BUILDING_AGE'] >= 0) & (df_clean['BUILDING_AGE'] < 400)]
        
        # Histogramm erstellen
        fig = px.histogram(
            df_clean, 
            x='BUILDING_AGE',
            nbins=20,
            title='Altersverteilung der verkauften Gebäude',
            labels={'BUILDING_AGE': 'Gebäudealter (Jahre)'},
            color_discrete_sequence=['#3366CC']
        )
        
        fig.update_layout(
            xaxis_title='Gebäudealter (Jahre)',
            yaxis_title='Anzahl der Verkäufe',
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        # Fehlerbehandlung
        st.error(f"Fehler beim Erstellen des Diagramms: {str(e)}")
        # Leeres Diagramm zurückgeben
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler: {str(e)}",
            xaxis_title="Gebäudealter (Jahre)",
            yaxis_title="Anzahl der Verkäufe"
        )
        return fig

def plot_price_by_building_class(df):
    """
    Erstellt ein Boxplot zum Vergleich der Verkaufspreise nach Gebäudeklasse.
    """
    try:
        # SALE PRICE in numerischen Wert umwandeln
        df_clean = df.copy()
        # Prüfen und korrigieren der Spaltenbezeichnungen (ggf. mit führenden Leerzeichen)
        price_col = 'SALE PRICE' if 'SALE PRICE' in df_clean.columns else ' SALE PRICE'
        category_col = 'BUILDING CLASS CATEGORY' if 'BUILDING CLASS CATEGORY' in df_clean.columns else ' BUILDING CLASS CATEGORY'
        
        # Konvertierung mit Fehlerbehandlung
        df_clean['SALE PRICE'] = pd.to_numeric(df_clean[price_col], errors='coerce')
        
                # NaN-Werte filtern
        df_clean = df_clean.dropna(subset=['SALE PRICE', category_col])
        
        # Prüfen ob noch Daten vorhanden sind
        if len(df_clean) == 0:
            st.error("Keine gültigen Daten für die Visualisierung verfügbar")
            # Leeres Diagramm zurückgeben
            fig = go.Figure()
            fig.update_layout(
                title="Keine Daten verfügbar",
                xaxis_title="Gebäudeklasse",
                yaxis_title="Verkaufspreis ($)"
            )
            return fig
        
        # Nur die Top 10 Gebäudeklassen (nach Anzahl) auswählen
        top_categories = df_clean[category_col].value_counts().head(10).index
        df_top = df_clean[df_clean[category_col].isin(top_categories)]
        
        # Entfernen von Ausreißern (z.B. Preise = 0 oder unrealistisch hohe Preise)
        df_top = df_top[(df_top['SALE PRICE'] > 100) & (df_top['SALE PRICE'] < 1e9)]
        
        # Boxplot erstellen
        fig = px.box(
            df_top, 
            x=category_col, 
            y='SALE PRICE',
            title='Verkaufspreise nach Gebäudeklasse (Top 10)',
            labels={
                category_col: 'Gebäudeklasse',
                'SALE PRICE': 'Verkaufspreis ($)'
            },
            color=category_col
        )
        
        fig.update_layout(
            xaxis_title='Gebäudeklasse',
            yaxis_title='Verkaufspreis ($)',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    except Exception as e:
        # Fehlerbehandlung
        st.error(f"Fehler beim Erstellen des Diagramms: {str(e)}")
        # Leeres Diagramm zurückgeben
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler: {str(e)}",
            xaxis_title="Gebäudeklasse",
            yaxis_title="Verkaufspreis ($)"
        )
        return fig

def plot_landsize_vs_price(df):
    """
    Erstellt ein Streudiagramm zum Verhältnis zwischen Grundstücksgröße und Verkaufspreis.
    """
    try:
        # Daten vorbereiten und in numerische Werte umwandeln
        df_clean = df.copy()
        # Prüfen und korrigieren der Spaltenbezeichnungen (ggf. mit führenden Leerzeichen)
        land_col = 'LAND SQUARE FEET' if 'LAND SQUARE FEET' in df_clean.columns else ' LAND SQUARE FEET'
        price_col = 'SALE PRICE' if 'SALE PRICE' in df_clean.columns else ' SALE PRICE'
        borough_col = 'BOROUGH' if 'BOROUGH' in df_clean.columns else ' BOROUGH'
        
        # Konvertierung mit Fehlerbehandlung
        df_clean['LAND SQUARE FEET'] = pd.to_numeric(df_clean[land_col], errors='coerce')
        df_clean['SALE PRICE'] = pd.to_numeric(df_clean[price_col], errors='coerce')
        
        # Debug-Informationen
        st.write(f"Anzahl gültiger LAND SQUARE FEET Einträge: {df_clean['LAND SQUARE FEET'].notna().sum()}")
        st.write(f"Anzahl gültiger SALE PRICE Einträge: {df_clean['SALE PRICE'].notna().sum()}")
        st.write(f"Anzahl gültiger BOROUGH Einträge: {df_clean[borough_col].notna().sum()}")
        
        # NaN-Werte und 0-Werte filtern
        df_clean = df_clean[(df_clean['LAND SQUARE FEET'] > 0) & (df_clean['SALE PRICE'] > 0)]
        df_clean = df_clean.dropna(subset=['LAND SQUARE FEET', 'SALE PRICE', borough_col])
        
        # Prüfen ob noch Daten vorhanden sind
        if len(df_clean) == 0:
            st.error("Keine gültigen Daten für die Visualisierung verfügbar")
            # Leeres Diagramm zurückgeben
            fig = go.Figure()
            fig.update_layout(
                title="Keine Daten verfügbar",
                xaxis_title="Grundstücksgröße (Quadratfuß)",
                yaxis_title="Verkaufspreis ($)"
            )
            return fig
        
        # Entfernen von Ausreißern (z.B. extreme Werte)
        df_clean = df_clean[
            (df_clean['LAND SQUARE FEET'] < df_clean['LAND SQUARE FEET'].quantile(0.99)) & 
            (df_clean['SALE PRICE'] < df_clean['SALE PRICE'].quantile(0.99))
        ]
        
        # Streudiagramm erstellen mit Farbe nach Borough
        fig = px.scatter(
            df_clean, 
            x='LAND SQUARE FEET', 
            y='SALE PRICE',
            color=borough_col,
            title='Verhältnis zwischen Grundstücksgröße und Verkaufspreis',
            labels={
                'LAND SQUARE FEET': 'Grundstücksgröße (Quadratfuß)',
                'SALE PRICE': 'Verkaufspreis ($)',
                borough_col: 'Stadtbezirk'
            },
            opacity=0.7,
            log_x=True,  # Logarithmische Skala für bessere Darstellung
            log_y=True
        )
        
        # Trendlinie hinzufügen
        fig.update_layout(
            xaxis_title='Grundstücksgröße (Quadratfuß, logarithmisch)',
            yaxis_title='Verkaufspreis ($, logarithmisch)',
            template='plotly_white'
        )
        
        return fig
    except Exception as e:
        # Fehlerbehandlung
        st.error(f"Fehler beim Erstellen des Diagramms: {str(e)}")
        # Leeres Diagramm zurückgeben
        fig = go.Figure()
        fig.update_layout(
            title=f"Fehler: {str(e)}",
            xaxis_title="Grundstücksgröße (Quadratfuß)",
            yaxis_title="Verkaufspreis ($)"
        )
        return fig

# Beispiel für die Integration in ein Streamlit-Dashboard
def create_dashboard():
    global base_data_new
    df = base_data_new.copy()
    
    try:      
        # Anzeigen der Visualisierungen mit Fehlerbehandlung
       
        with st.container():
            st.header("Altersverteilung der verkauften Gebäude")
            st.plotly_chart(plot_building_age_distribution(df), use_container_width=True)
        
        with st.container():
            st.header("Verkaufspreise nach Gebäudeklasse") 
            st.plotly_chart(plot_price_by_building_class(df), use_container_width=True)
        
        with st.container():
            st.header("Verhältnis zwischen Grundstücksgröße und Verkaufspreis")
            st.plotly_chart(plot_landsize_vs_price(df), use_container_width=True)
            
    except Exception as e:
        st.error(f"Fehler bei der Erstellung des Dashboards: {str(e)}")
        st.error("Stacktrace:", exc_info=True)

st.set_page_config(layout="wide", page_title="NYC Real Estate Dashboard")

st.markdown("""
<style>
    .main .block-container {
        max-width: 100%;
        padding-top: 0rem;
        border: 1px solid black;
    }
</style>
""", unsafe_allow_html=True)

col_title, col_nav = st.columns([4, 2])
with col_title:
    st.title("NYC Real Estate Dashboard")

with st.sidebar:
    page = option_menu("Navigation", ["Interaktive Karte", "Zahlen", "Reports & Insights"],
                        icons=['map', '123', 'clipboard-pulse'], menu_icon="cast", default_index=0)

if page == "Interaktive Karte":
    interactive_section()
    other, prediction = st.columns([2, 2])
    with prediction:
        inference_section()
    with other:
        st.header("Zipcode Insight Visualization")
        show_zipcode_visualizations()

if page == "Zahlen":
    render_real_estate_analysis()

if page == "Reports & Insights":
    create_dashboard()
