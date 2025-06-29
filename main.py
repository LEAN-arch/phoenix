# main.py
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
import logging
import warnings
from pathlib import Path
import numpy as np

# --- EXPANSION: Import Plotly for advanced visualizations ---
import plotly.express as px
import plotly.graph_objects as go

# --- Geospatial libraries for isochrones ---
from shapely.geometry import Point

# Import from our refactored modules
from core import DataManager, PredictiveAnalyticsEngine, EnvFactors
from utils import load_config, ReportGenerator

# --- System Setup ---
st.set_page_config(page_title="RedShield AI: Phoenix v4.0", layout="wide", initial_sidebar_state="expanded")
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("logs/redshield_phoenix.log")])
logger = logging.getLogger(__name__)

# --- Main Dashboard Class ---
class Dashboard:
    """Handles the rendering of the Streamlit user interface for Phoenix v4.0."""
    def __init__(self, dm: DataManager, engine: PredictiveAnalyticsEngine):
        self.dm = dm
        self.engine = engine
        self.config = dm.config

        if 'historical_data' not in st.session_state:
            st.session_state['historical_data'] = []
            
        if 'env_factors' not in st.session_state:
            avg_pop_density = self.dm.zones_gdf['population'].mean() if not self.dm.zones_gdf.empty else 50000
            st.session_state['env_factors'] = EnvFactors(is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, population_density=avg_pop_density, air_quality_index=50.0, heatwave_alert=False, day_type='Weekday', time_of_day='Midday', public_event_type='None', hospital_divert_status=0.0, police_activity='Normal', school_in_session=True)

    def render(self):
        st.title("RedShield AI: Phoenix v4.0")
        st.markdown("##### Proactive Emergency Response & Resource Allocation Platform")
        self._render_sidebar()
        env_factors = st.session_state['env_factors']
        historical_data = st.session_state['historical_data']

        with st.spinner("Executing Advanced Analytics & Optimization Pipeline..."):
            current_incidents = self.dm.get_current_incidents(env_factors)
            kpi_df, sparkline_data = self.engine.generate_kpis_with_sparklines(historical_data, env_factors, current_incidents)
            forecast_df = self.engine.generate_forecast(kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df)
            
        st.session_state.update({
            'kpi_df': kpi_df, 'forecast_df': forecast_df, 'allocations': allocations,
            'sparkline_data': sparkline_data
        })

        tab1, tab2, tab3 = st.tabs(["🔥 Operational Command", "📊 KPI Deep Dive", "🧠 Methodology & Insights"])

        with tab1:
            self._render_operational_command_tab(kpi_df, allocations, current_incidents)
        with tab2:
            self._render_kpi_deep_dive_tab(kpi_df, forecast_df)
        with tab3:
            self._render_methodology_tab()

    def _plot_kpi_trend_chart(self, data, normal_range, color_map):
        fig = go.Figure()

        current_val = data[-1]
        if current_val > normal_range[1]: line_color = color_map['high']
        elif current_val < normal_range[0]: line_color = color_map['low']
        else: line_color = color_map['normal']

        fig.add_trace(go.Scatter(
            x=list(range(len(data))), y=data,
            mode='lines', name='Current Trend',
            line=dict(color=line_color, width=2.5),
            fill='tozeroy',
            fillcolor=f'rgba({int(line_color[1:3], 16)}, {int(line_color[3:5], 16)}, {int(line_color[5:7], 16)}, 0.1)'
        ))

        fig.update_layout(
            height=60, margin=dict(l=0, r=40, t=5, b=0),
            xaxis=dict(visible=False), 
            yaxis=dict(
                visible=True, showticklabels=True, side='right', nticks=3,
                tickfont=dict(size=9, color='grey'),
                range=[min(data)*0.95, max(data)*1.05]
            ),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
        )
        return fig

    def _plot_system_pressure_gauge(self, kpi_df, env_factors):
        traffic_norm = (env_factors.traffic_level - 0.5) / 2.5
        hospital_norm = env_factors.hospital_divert_status
        adequacy_norm = 1 - kpi_df['Resource Adequacy Index'].mean()
        
        pressure_score = (traffic_norm * 0.3) + (hospital_norm * 0.4) + (adequacy_norm * 0.3)
        pressure_score = min(pressure_score * 125, 100)

        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pressure_score,
            title = {'text': "System Pressure", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#222"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 40], 'color': '#388E3C'},
                    {'range': [40, 75], 'color': '#FBC02D'},
                    {'range': [75, 100], 'color': '#D32F2F'}],
            }))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        return fig

    def _render_operational_command_tab(self, kpi_df, allocations, incidents):
        sparkline_data = st.session_state.get('sparkline_data', {})
        
        st.subheader("System Vitals (Last 24 Hours)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            val = len(incidents)
            st.metric("🚨 Active Incidents", f"{val}")
            if 'active_incidents' in sparkline_data:
                st.plotly_chart(self._plot_kpi_trend_chart(
                    sparkline_data['active_incidents']['values'], sparkline_data['active_incidents']['range'],
                    {'normal': '#1E90FF', 'high': '#FF4500', 'low': '#1E90FF'}
                ), use_container_width=True)
        with c2:
            val = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
            st.metric("🚑 Available Units", f"{val}")
            if 'available_ambulances' in sparkline_data:
                st.plotly_chart(self._plot_kpi_trend_chart(
                    sparkline_data['available_ambulances']['values'], sparkline_data['available_ambulances']['range'],
                    {'normal': '#50C878', 'low': '#FF4500', 'high': '#50C878'}
                ), use_container_width=True)
        with c3:
            val = kpi_df['Integrated_Risk_Score'].max()
            st.metric("📈 Highest Zone Risk", f"{val:.3f}")
            if 'max_risk' in sparkline_data:
                st.plotly_chart(self._plot_kpi_trend_chart(
                    sparkline_data['max_risk']['values'], sparkline_data['max_risk']['range'],
                    {'normal': '#1E90FF', 'high': '#AF4035', 'low': '#1E90FF'}
                ), use_container_width=True)
        with c4:
            val = kpi_df['Resource Adequacy Index'].mean()
            st.metric("📊 System Adequacy", f"{val:.1%}")
            if 'adequacy' in sparkline_data:
                st.plotly_chart(self._plot_kpi_trend_chart(
                    sparkline_data['adequacy']['values'], sparkline_data['adequacy']['range'],
                    {'normal': '#50C878', 'low': '#FF4500', 'high': '#50C878'}
                ), use_container_width=True)
        st.divider()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Live Operations Map")
            self._render_dynamic_map(kpi_df, incidents, allocations)
        with col2:
            st.subheader("Decision Support")
            st.plotly_chart(self._plot_system_pressure_gauge(kpi_df, st.session_state['env_factors']), use_container_width=True)
            self._plot_resource_to_risk_adequacy_v2(kpi_df, allocations)

    def _plot_resource_to_risk_adequacy_v2(self, kpi_df, allocations):
        if kpi_df.empty: st.info("No data for risk adequacy plot."); return
        top_zones_df = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
        top_zones_df['allocated_units'] = top_zones_df['Zone'].map(allocations).fillna(0)
        risk_coverage_per_unit = self.config['model_params'].get('risk_coverage_per_unit', 0.25)
        top_zones_df['risk_covered'] = top_zones_df['allocated_units'] * risk_coverage_per_unit
        top_zones_df['adequacy_ratio'] = (top_zones_df['risk_covered'] / top_zones_df['Integrated_Risk_Score']).clip(0, 1.5)

        def get_color(ratio):
            if ratio < 0.7: return '#D32F2F';
            if ratio < 1.0: return '#FBC02D';
            return '#388E3C'
        top_zones_df['adequacy_color'] = top_zones_df['adequacy_ratio'].apply(get_color)

        fig = go.Figure()
        fig.add_trace(go.Bar(y=top_zones_df['Zone'], x=top_zones_df['Integrated_Risk_Score'], orientation='h', name='Total Risk (Demand)', marker_color='#e0e0e0', hovertemplate="<b>Zone:</b> %{y}<br><b>Total Risk:</b> %{x:.3f}<extra></extra>"))
        fig.add_trace(go.Bar(y=top_zones_df['Zone'], x=top_zones_df['risk_covered'], orientation='h', name='Covered Risk (Supply)', marker_color=top_zones_df['adequacy_color'], text=top_zones_df['allocated_units'].astype(int).astype(str) + " Unit(s)", textposition='inside', insidetextanchor='middle', textfont=dict(color='white', size=12), hovertemplate="<b>Zone:</b> %{y}<br><b>Risk Covered:</b> %{x:.3f}<br><b>Allocated:</b> %{text}<extra></extra>"))

        fig.update_layout(title='Resource vs. Demand for High-Risk Zones', xaxis_title='Integrated Risk Score', yaxis_title=None, height=350, yaxis={'categoryorder':'total ascending'}, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), barmode='overlay', plot_bgcolor='white', margin=dict(l=10, r=10, t=70, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**How to Read:** The light grey bar is the risk (demand). The colored bar is the coverage from allocated units (supply).")

    def _render_dynamic_map(self, kpi_df, incidents, allocations):
        if self.dm.zones_gdf.empty or kpi_df.empty: return
        try:
            map_gdf = self.dm.zones_gdf.join(kpi_df.set_index('Zone'))
            center = map_gdf.union_all().centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron", prefer_canvas=True)
            
            folium.Choropleth(geo_data=map_gdf.to_json(), data=map_gdf, columns=['Zone', 'Integrated_Risk_Score'], key_on='feature.properties.Zone', fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Integrated Risk Score', name="Risk Heatmap").add_to(m)

            incidents_fg = folium.FeatureGroup(name='Live Incidents', show=True)
            for inc in incidents:
                if 'location' in inc and 'lat' in inc['location'] and 'lon' in inc['location']:
                    icon_type = "car-crash" if "Accident" in inc['type'] else "first-aid"
                    folium.Marker(location=[inc['location']['lat'], inc['location']['lon']], tooltip=f"Type: {inc['type']}<br>Triage: {inc['triage']}", icon=folium.Icon(color='red', icon=icon_type, prefix='fa')).add_to(incidents_fg)
            incidents_fg.add_to(m)

            ambulance_fg = folium.FeatureGroup(name='Available Unit Reach (5-min)', show=False)
            for amb_id, amb_data in self.dm.ambulances.items():
                if amb_data['status'] == 'Disponible':
                    folium.Circle(location=[amb_data['location'].y, amb_data['location'].x], radius=1609 * 1.5,
                                  color='#1E90FF', fill=True, fill_opacity=0.1, tooltip=f"Unit {amb_id} Reach").add_to(ambulance_fg)
                    folium.Marker(location=[amb_data['location'].y, amb_data['location'].x], icon=folium.Icon(color='blue', icon='ambulance', prefix='fa'), tooltip=f"Unit {amb_id} (Available)").add_to(ambulance_fg)
            ambulance_fg.add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, use_container_width=True, height=600)
        except Exception as e:
            logger.error(f"Failed to render dynamic map: {e}", exc_info=True)
            st.error(f"Error rendering dynamic map: {e}")
            
    def _render_sidebar(self):
        st.sidebar.title("Strategic Controls")
        st.sidebar.markdown("Adjust real-time factors to simulate different scenarios.")
        env = st.session_state['env_factors']
        with st.sidebar.expander("General Environmental Factors", expanded=True):
            is_holiday = st.checkbox("Is Holiday", value=env.is_holiday, key="is_holiday_sb")
            weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(env.weather), key="weather_sb")
            aqi = st.slider("Air Quality Index (AQI)", 0.0, 500.0, env.air_quality_index, 5.0, key="aqi_sb")
            heatwave = st.checkbox("Heatwave Alert", value=env.heatwave_alert, key="heatwave_sb")
        with st.sidebar.expander("Contextual & Event-Based Factors", expanded=True):
            day_type = st.selectbox("Day Type", ['Weekday', 'Friday', 'Weekend'], index=['Weekday', 'Friday', 'Weekend'].index(env.day_type))
            time_of_day = st.selectbox("Time of Day", ['Morning Rush', 'Midday', 'Evening Rush', 'Night'], index=['Morning Rush', 'Midday', 'Evening Rush', 'Night'].index(env.time_of_day))
            public_event_type = st.selectbox("Public Event Type", ['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'], index=['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'].index(env.public_event_type))
            school_in_session = st.checkbox("School In Session", value=env.school_in_session)
        with st.sidebar.expander("System Strain & Response Factors", expanded=True):
            traffic = st.slider("General Traffic Level", 0.5, 3.0, env.traffic_level, 0.1, key="traffic_sb", help="A general multiplier for traffic congestion.")
            hospital_divert_status = st.slider("Hospital Divert Status (%)", 0, 100, int(env.hospital_divert_status * 100), 5, help="Percentage of hospitals on diversion, indicating system strain.") / 100.0
            police_activity = st.selectbox("Police Activity Level", ['Low', 'Normal', 'High'], index=['Low', 'Normal', 'High'].index(env.police_activity))
        major_event = public_event_type != 'None'
        new_env = EnvFactors(is_holiday=is_holiday, weather=weather, traffic_level=traffic, major_event=major_event, population_density=env.population_density, air_quality_index=aqi, heatwave_alert=heatwave, day_type=day_type, time_of_day=time_of_day, public_event_type=public_event_type, hospital_divert_status=hospital_divert_status, police_activity=police_activity, school_in_session=school_in_session)
        if new_env != st.session_state['env_factors']:
            st.session_state['env_factors'] = new_env
            st.rerun()
        st.sidebar.divider()
        st.sidebar.header("Data & Reporting")
        uploaded_file = st.sidebar.file_uploader("Upload Historical Incidents (JSON)", type=["json"])
        if uploaded_file:
            try:
                st.session_state['historical_data'] = json.load(uploaded_file)
                st.sidebar.success(f"Loaded {len(st.session_state['historical_data'])} historical records.")
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Error loading data: {e}")
        if st.sidebar.button("Generate & Download PDF Report", use_container_width=True):
            self._generate_report()

    def _generate_report(self):
        with st.spinner("Generating Report..."):
            pdf_buffer = ReportGenerator.generate_pdf_report(
                st.session_state.get('kpi_df', pd.DataFrame()),
                st.session_state.get('forecast_df', pd.DataFrame()),
                st.session_state.get('allocations', {}),
                st.session_state.get('env_factors')
            )
        if pdf_buffer.getvalue():
            st.sidebar.download_button(
                label="Download PDF Report", data=pdf_buffer,
                file_name=f"RedShield_Phoenix_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        else:
            st.sidebar.error("Report generation failed."); st.sidebar.info("Check logs for details.")
            
    def _render_kpi_deep_dive_tab(self, kpi_df, forecast_df):
        st.subheader("Comprehensive Risk Indicator Matrix")
        st.dataframe(kpi_df.set_index('Zone').style.format("{:.3f}").background_gradient(cmap='viridis', axis=0), use_container_width=True)
        st.divider()
        st.subheader("Advanced Analytical Visualizations")
        if not kpi_df.empty:
            tab1, tab2, tab3 = st.tabs(["📍 Zone Vulnerability Quadrant", "📊 Risk Contribution Drill-Down", "📈 Risk Forecast & Uncertainty"])
            with tab1:
                st.markdown("**Analysis:** This plot segments zones by their long-term structural vulnerability vs. their immediate dynamic risk.")
                self._plot_vulnerability_quadrant(kpi_df)
            with tab2:
                st.markdown("**Analysis:** Select a high-risk zone to break down its `Integrated_Risk_Score` into its constituent model components.")
                top_5_zones = kpi_df.nlargest(5, 'Integrated_Risk_Score')['Zone'].tolist()
                selected_zone = st.selectbox("Select a High-Risk Zone to Analyze:", options=top_5_zones)
                if selected_zone:
                    self._plot_risk_contribution_sunburst(kpi_df, selected_zone)
            with tab3:
                st.markdown("**Analysis:** This chart projects the selected zone's risk over 72 hours. The shaded area represents the model's **95% confidence interval** — a wider band indicates greater uncertainty.")
                all_zones = forecast_df['Zone'].unique().tolist()
                default_zones = kpi_df.nlargest(5, 'Integrated_Risk_Score')['Zone'].tolist()
                selected_zones_fc = st.multiselect("Select zones to visualize forecast:", options=all_zones, default=default_zones)
                if selected_zones_fc:
                    self._plot_forecast_with_uncertainty(forecast_df, selected_zones_fc)
        else:
            st.info("Insufficient data for advanced plots.")

    def _render_methodology_tab(self):
        st.header("System Architecture & Methodology")
        st.markdown("""
        ### I. High-Level Goal & Architectural Philosophy
        The fundamental goal of RedShield AI: Phoenix v4.0 is to engineer a paradigm shift in emergency response—from a traditional **reactive model** to a **proactive, predictive posture**. We anticipate where incidents will likely emerge and pre-position resources to minimize response times and maximize impact.
        To achieve this, the system is built on a philosophy of **Hierarchical Ensemble Modeling**. Instead of relying on a single algorithm, Phoenix v4.0 integrates a diverse portfolio of analytical techniques in a layered architecture. This creates a highly robust and resilient system where the weaknesses of any one model are offset by the strengths of others.
        #### The Three Analytical Layers:
        1.  **LAYER 1: Foundational Ensemble.** Consists of well-established statistical models (Hawkes Processes, Bayesian Networks) that create a robust baseline understanding of risk, producing the `Ensemble_Risk_Score`.
        2.  **LAYER 2: Advanced AI & Complexity Proxies.** Introduces computationally inexpensive but powerful proxies for cutting-edge models (ST-GPs, GNNs, Game Theory) to capture deeper, nuanced patterns.
        3.  **LAYER 3: Integrated Synthesis.** The outputs of the first two layers are combined in a final, weighted synthesis to produce the ultimate `Integrated_Risk_Score`, which drives all final recommendations.
        """)
        with st.expander("II. Detailed Methodology Breakdown", expanded=False):
            st.markdown("""
            #### **I. Stochastic & Statistical Modeling**
            *   **Non-Homogeneous Poisson Process (NHPP):** Models the time-varying baseline incident rate (`μ`), capturing predictable daily and weekly patterns.
            *   **Hawkes Process (Self-Exciting):** The core of our trauma/violence clustering models. It assumes some events can trigger "aftershocks."
            *   **Marked Point Processes:** Each incident is treated as a "marked" event with metadata (e.g., `type: 'Trauma'`), allowing models to respond selectively.
            
            #### **II. Spatiotemporal & Graph Models**
            *   **Spatiotemporal Gaussian Processes (ST-GPs):** Our `STGP_Risk` is a proxy. It interpolates risk intelligently across the map.
            *   **Graph Laplacians (`Spatial Spillover Risk`):** Models the city's road network as a graph to simulate how risk and chaos can diffuse.
            *   **Graph Neural Networks (GNNs):** Our `GNN_Structural_Risk` is a proxy. A GNN learns a zone's inherent vulnerability based on its position in the network.
            
            #### **III. Operations Research & Prescriptive Analytics**
            *   **Mixed-Integer Linear Programming (MILP):** Used for `Linear Optimal` allocation. It finds the best assignment of ambulances to maximize risk coverage under real-world constraints (e.g., total units available).
            *   **Non-Linear Programming (NLP):** Used for `Non-Linear Optimal` allocation. This more advanced model captures real-world effects like **diminishing returns** (the 1st ambulance in a zone is more valuable than the 5th) and **congestion penalties**. It provides the most realistic and robust recommendations.
            *   **Queueing Theory:** Conceptually used to model system strain, such as hospital ER wait times, which penalizes the overall system adequacy.
            """)
        with st.expander("III. Key Performance Indicator (KPI) Glossary", expanded=False):
            kpi_defs = {
                "Integrated Risk Score": "**The final, primary risk metric** used for all decisions, blending foundational and advanced AI models.",
                "Ensemble Risk Score": "Blended risk score from the foundational (Layer 1) statistical models.",
                "GNN Structural Risk": "A zone's intrinsic vulnerability due to its position in the road network.",
                "STGP Risk": "Risk from proximity to recent, severe incidents (spatiotemporal correlation).",
                "Game Theory Tension": "A measure of a zone's contribution to system-wide resource competition.",
                "Chaos Sensitivity Score": "Measures system volatility and fragility. High score = 'The system is unstable.'",
                "Anomaly Score": "Measures the 'strangeness' of the current incident pattern compared to history.",
                "Resource Adequacy Index": "System-wide ratio of available units to expected demand, penalized by hospital strain."
            }
            for kpi, definition in kpi_defs.items():
                st.markdown(f"**{kpi}**: {definition}")

    def _plot_risk_contribution_sunburst(self, kpi_df: pd.DataFrame, zone_name: str):
        zone_data = kpi_df[kpi_df['Zone'] == zone_name].iloc[0]
        adv_weights = self.config['model_params'].get('advanced_model_weights', {})
        data = {
            'ids': ['Integrated Risk', 'Base Ensemble', 'Advanced Models', 'STGP', 'HMM', 'GNN', 'Game Theory'],
            'labels': [f"Total: {zone_data['Integrated_Risk_Score']:.2f}", 'Base Ensemble', 'Advanced Models', 'STGP Risk', 'HMM State', 'GNN Structure', 'Game Tension'],
            'parents': ['', 'Integrated Risk', 'Integrated Risk', 'Advanced Models', 'Advanced Models', 'Advanced Models', 'Advanced Models'],
            'values': [
                zone_data['Integrated_Risk_Score'],
                adv_weights.get('base_ensemble', 0) * zone_data['Ensemble Risk Score'],
                (adv_weights.get('stgp', 0) * zone_data['STGP_Risk'] + adv_weights.get('hmm', 0) * zone_data['HMM_State_Risk'] +
                 adv_weights.get('gnn', 0) * zone_data['GNN_Structural_Risk'] + adv_weights.get('game_theory', 0) * zone_data['Game_Theory_Tension']),
                adv_weights.get('stgp', 0) * zone_data['STGP_Risk'], adv_weights.get('hmm', 0) * zone_data['HMM_State_Risk'],
                adv_weights.get('gnn', 0) * zone_data['GNN_Structural_Risk'], adv_weights.get('game_theory', 0) * zone_data['Game_Theory_Tension']
            ]
        }
        fig = go.Figure(go.Sunburst(ids=data['ids'], labels=data['labels'], parents=data['parents'], values=data['values'], branchvalues="total", hovertemplate='<b>%{label}</b><br>Contribution: %{value:.3f}<extra></extra>'))
        fig.update_layout(margin=dict(t=20, l=0, r=0, b=0), title_text=f"Risk Breakdown for Zone: {zone_name}", title_x=0.5, height=450)
        st.plotly_chart(fig, use_container_width=True)

    def _plot_vulnerability_quadrant(self, kpi_df: pd.DataFrame):
        x_mean = kpi_df['Ensemble Risk Score'].mean()
        y_mean = kpi_df['GNN_Structural_Risk'].mean()
        hover_text = [f"<b>Zone: {row['Zone']}</b><br><br>Dynamic Risk: {row['Ensemble Risk Score']:.3f}<br>Structural Risk: {row['GNN_Structural_Risk']:.3f}<br>Integrated Risk: {row['Integrated_Risk_Score']:.3f}<extra></extra>" for index, row in kpi_df.iterrows()]
        fig = px.scatter(kpi_df, x="Ensemble Risk Score", y="GNN_Structural_Risk", color="Integrated_Risk_Score", size="Expected Incident Volume", hover_name="Zone", color_continuous_scale="reds", size_max=18)
        fig.update_traces(hovertemplate=hover_text)
        fig.add_vline(x=x_mean, line_width=1, line_dash="dash", line_color="grey")
        fig.add_hline(y=y_mean, line_width=1, line_dash="dash", line_color="grey")
        fig.add_shape(type="rect", x0=x_mean, y0=y_mean, x1=kpi_df['Ensemble Risk Score'].max()*1.1, y1=kpi_df['GNN_Structural_Risk'].max()*1.1, line=dict(width=0), fillcolor="rgba(255, 0, 0, 0.1)", layer="below")
        fig.add_shape(type="rect", x0=0, y0=y_mean, x1=x_mean, y1=kpi_df['GNN_Structural_Risk'].max()*1.1, line=dict(width=0), fillcolor="rgba(0, 0, 255, 0.1)", layer="below")
        fig.add_shape(type="rect", x0=x_mean, y0=0, x1=kpi_df['Ensemble Risk Score'].max()*1.1, y1=y_mean, line=dict(width=0), fillcolor="rgba(255, 165, 0, 0.1)", layer="below")
        fig.add_annotation(x=x_mean*1.5 if x_mean > 0 else 0.8, y=y_mean*1.8 if y_mean > 0 else 0.8, text="<b>Crisis Zones</b>", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=x_mean/2, y=y_mean*1.8 if y_mean > 0 else 0.8, text="<b>Latent Threats</b>", showarrow=False, font=dict(color="navy"))
        fig.add_annotation(x=x_mean*1.5 if x_mean > 0 else 0.8, y=y_mean/2, text="<b>Acute Hotspots</b>", showarrow=False, font=dict(color="darkorange"))
        fig.update_layout(xaxis_title="Dynamic Risk (Real-time Threat)", yaxis_title="Structural Vulnerability (Intrinsic Threat)", coloraxis_colorbar_title_text='Integrated<br>Risk')
        st.plotly_chart(fig, use_container_width=True)

    def _plot_forecast_with_uncertainty(self, forecast_df, selected_zones):
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, zone in enumerate(selected_zones):
            zone_df = forecast_df[forecast_df['Zone'] == zone]
            if zone_df.empty: continue
            color = colors[i % len(colors)]
            fig.add_trace(go.Scatter(x=np.concatenate([zone_df['Horizon (Hours)'], zone_df['Horizon (Hours)'][::-1]]), y=np.concatenate([zone_df['Upper_Bound'], zone_df['Lower_Bound'][::-1]]), fill='toself', fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)', line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=zone_df['Horizon (Hours)'], y=zone_df['Combined Risk'], name=zone, line=dict(color=color, width=2), mode='lines+markers'))
        fig.update_layout(title='72-Hour Risk Forecast with 95% Confidence Interval', xaxis_title='Horizon (Hours)', yaxis_title='Projected Integrated Risk Score', legend_title_text='Zone', hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

def main():
    """Main function to run the application."""
    try:
        config = load_config()
        data_manager = DataManager(config)
        engine = PredictiveAnalyticsEngine(data_manager, config)
        dashboard = Dashboard(data_manager, engine)
        dashboard.render()
    except Exception as e:
        logger.error(f"A fatal error occurred in the application: {e}", exc_info=True)
        st.error(f"A fatal application error occurred: {e}. Please check logs and configuration file.")

if __name__ == "__main__":
    main()
