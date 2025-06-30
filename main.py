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

    def _render_system_status_bar(self, kpi_df, incidents, sparkline_data):
        st.subheader("System Health Status")

        incidents_val = len(incidents)
        incidents_data = sparkline_data.get('active_incidents', {'values': [incidents_val], 'range': [incidents_val-1, incidents_val+1]})
        
        ambulances_val = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        ambulances_data = sparkline_data.get('available_ambulances', {'values': [ambulances_val], 'range': [ambulances_val-1, ambulances_val+1]})
        
        max_risk_val = kpi_df['Integrated_Risk_Score'].max()
        max_risk_data = sparkline_data.get('max_risk', {'values': [max_risk_val], 'range': [0, 1]})
        
        adequacy_val = kpi_df['Resource Adequacy Index'].mean()
        adequacy_data = sparkline_data.get('adequacy', {'values': [adequacy_val], 'range': [0, 1]})

        def get_status(val, normal_range, high_is_bad=True):
            if (high_is_bad and val > normal_range[1]) or (not high_is_bad and val < normal_range[0]):
                return "Critical", "#D32F2F"
            if (high_is_bad and val > normal_range[0]) or (not high_is_bad and val < normal_range[1]):
                return "Elevated", "#FBC02D"
            return "Normal", "#388E3C"

        def get_trend_arrow(data):
            if len(data) < 2: return ""
            if data[-1] > data[-2]: return "▲"
            if data[-1] < data[-2]: return "▼"
            return "▬"

        inc_status, inc_color = get_status(incidents_val, incidents_data['range'])
        amb_status, amb_color = get_status(ambulances_val, ambulances_data['range'], high_is_bad=False)
        risk_status, risk_color = get_status(max_risk_val, max_risk_data['range'])
        adeq_status, adeq_color = get_status(adequacy_val, adequacy_data['range'], high_is_bad=False)

        st.markdown(f"""
        <div style="width: 100%; display: flex; border: 1px solid #444; border-radius: 5px; overflow: hidden; font-family: sans-serif;">
            <div style="flex: 1; background-color: {inc_color}; padding: 10px; text-align: center; color: white; border-right: 1px solid #fff4;">
                <div style="font-size: 1.5rem; font-weight: bold;">{incidents_val} {get_trend_arrow(incidents_data['values'])}</div>
                <div style="font-size: 0.8rem;">Active Incidents</div>
            </div>
            <div style="flex: 1; background-color: {amb_color}; padding: 10px; text-align: center; color: white; border-right: 1px solid #fff4;">
                <div style="font-size: 1.5rem; font-weight: bold;">{ambulances_val} {get_trend_arrow(ambulances_data['values'])}</div>
                <div style="font-size: 0.8rem;">Available Units</div>
            </div>
            <div style="flex: 1; background-color: {risk_color}; padding: 10px; text-align: center; color: white; border-right: 1px solid #fff4;">
                <div style="font-size: 1.5rem; font-weight: bold;">{max_risk_val:.3f} {get_trend_arrow(max_risk_data['values'])}</div>
                <div style="font-size: 0.8rem;">Max Zone Risk</div>
            </div>
            <div style="flex: 1; background-color: {adeq_color}; padding: 10px; text-align: center; color: white;">
                <div style="font-size: 1.5rem; font-weight: bold;">{adequacy_val:.1%} {get_trend_arrow(adequacy_data['values'])}</div>
                <div style="font-size: 0.8rem;">System Adequacy</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    def _render_operational_command_tab(self, kpi_df, allocations, incidents):
        sparkline_data = st.session_state.get('sparkline_data', {})
        self._render_system_status_bar(kpi_df, incidents, sparkline_data)
        st.divider()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Live Operations Map")
            self._render_dynamic_map(kpi_df, incidents, allocations)
        with col2:
            st.subheader("Decision Support")
            st.plotly_chart(self._plot_system_pressure_gauge(kpi_df, st.session_state['env_factors']), use_container_width=True)
            self._plot_resource_to_risk_adequacy_v2(kpi_df, allocations)

    def _plot_system_pressure_gauge(self, kpi_df, env_factors):
        traffic_norm = (env_factors.traffic_level - 0.5) / 2.5
        hospital_norm = env_factors.hospital_divert_status
        adequacy_norm = 1 - kpi_df['Resource Adequacy Index'].mean()
        pressure_score = (traffic_norm * 0.3) + (hospital_norm * 0.4) + (adequacy_norm * 0.3)
        pressure_score = min(pressure_score * 125, 100)
        fig = go.Figure(go.Indicator(
            mode = "gauge+number", value = pressure_score,
            title = {'text': "System Pressure", 'font': {'size': 20}},
            gauge = { 'axis': {'range': [None, 100]}, 'bar': {'color': "#222"},
                      'steps': [{'range': [0, 40], 'color': '#388E3C'}, {'range': [40, 75], 'color': '#FBC02D'}, {'range': [75, 100], 'color': '#D32F2F'}]}))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        return fig
    
    def _plot_resource_to_risk_adequacy_v2(self, kpi_df, allocations):
        if kpi_df.empty: st.info("No data for risk adequacy plot."); return
        top_zones_df = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
        top_zones_df['allocated_units'] = top_zones_df['Zone'].map(allocations).fillna(0)
        risk_coverage_per_unit = self.config['model_params'].get('risk_coverage_per_unit', 0.25)
        top_zones_df['risk_covered'] = top_zones_df['allocated_units'] * risk_coverage_per_unit
        top_zones_df['adequacy_ratio'] = (top_zones_df['risk_covered'] / top_zones_df['Integrated_Risk_Score']).clip(0, 1.5)

        def get_color(ratio):
            if ratio < 0.7: return '#D32F2F'
            if ratio < 1.0: return '#FBC02D'
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
            # --- START: KEYERROR FIX ---
            # Reset the index so 'Zone' becomes a regular column that Folium can find.
            # Then, rename the original 'name' column (from the GeoDataFrame) to 'Zone' to match kpi_df.
            map_gdf.reset_index(inplace=True)
            if 'name' not in map_gdf.columns and 'Zone' in map_gdf.columns:
                pass # Already correct
            elif 'name' in map_gdf.columns and 'Zone' not in map_gdf.columns:
                 map_gdf.rename(columns={'name':'Zone'}, inplace=True)
            # --- END: KEYERROR FIX ---

            center = map_gdf.union_all().centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron", prefer_canvas=True)
            
            folium.Choropleth(
                geo_data=map_gdf.to_json(), data=map_gdf,
                columns=['Zone', 'Integrated_Risk_Score'], key_on='feature.properties.Zone',
                fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
                legend_name='Integrated Risk Score', name="Risk Heatmap"
            ).add_to(m)

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

# In main.py, replace the existing _render_methodology_tab method

    def _render_methodology_tab(self):
        st.header("System Architecture & Methodology")
        st.markdown("This section provides a deep dive into the analytical engine powering the Phoenix v4.0 platform. It is designed for data scientists, analysts, and command staff who wish to understand the 'why' behind the system's predictions and prescriptions.")

        with st.expander("I. Architectural Philosophy: From Prediction to Prescription", expanded=True):
            st.markdown("""
            The fundamental goal of RedShield AI: Phoenix v4.0 is to engineer a paradigm shift in emergency response—from a traditional **reactive model** (dispatching units after an incident occurs) to a **proactive, prescriptive posture** (anticipating where incidents are likely to emerge and prescriptively positioning resources to minimize response times and maximize impact).

            To achieve this, the system is built on a philosophy of **Hierarchical Ensemble Modeling**. Instead of relying on a single "black box" algorithm, Phoenix v4.0 integrates a diverse portfolio of analytical techniques in a layered, "glass box" architecture. This creates a highly robust and resilient system where the weaknesses of any one model are offset by the strengths of others.

            The architecture is composed of four primary layers:
            1.  **LAYER 1: Foundational Models.** Consists of well-established statistical models (Hawkes Processes, Bayesian Networks, Graph Laplacians) that create a robust baseline understanding of risk. This produces the `Ensemble_Risk_Score`.
            2.  **LAYER 2: Advanced AI & Complexity Proxies.** Introduces computationally inexpensive but powerful proxies for cutting-edge models (ST-GPs, HMMs, GNNs) to capture deeper, more nuanced patterns that complement the foundational layer.
            3.  **LAYER 3: Integrated Synthesis (Prediction).** The outputs of the first two layers are combined in a final, weighted synthesis to produce the ultimate `Integrated_Risk_Score`. This score represents the system's best **prediction** of risk.
            4.  **LAYER 4: Prescriptive Optimization (Prescription).** The `Integrated_Risk_Score` and `Expected Incident Volume` are fed into an Operations Research (OR) engine. This layer moves beyond prediction to **prescription**, determining the *optimal* real-world action to take (e.g., ambulance allocation) to best mitigate the predicted risk.
            """)

        with st.expander("II. The Prediction Engine: A Multi-Model Deep Dive", expanded=False):
            st.info("#### Core Principle: Different questions require different tools. No single model can capture all facets of urban risk.", icon="💡")
            
            st.markdown("---")
            st.markdown("#### **A. Stochastic & Statistical Models (The 'When')**")
            st.markdown("""
            *   **Non-Homogeneous Poisson Process (NHPP):** This model forms the temporal backbone of our predictions. It understands that incident rates are not constant.
                -   **Question it Answers:** *"What is the baseline probability of an incident at 3 AM on a Tuesday versus 6 PM on a Friday?"*
                -   **Relevance:** Captures the predictable, cyclical nature of urban life, ensuring our baseline risk is sensitive to the time of day and day of the week.

            *   **Hawkes Process (Self-Exciting Point Process):** This is the cornerstone of our violence and cascading accident models. It operates on the principle that certain events can trigger "aftershocks."
                -   **Question it Answers:** *"Given a shooting just occurred, what is the immediate, elevated risk of another shooting in the same area?"* or *"After a major highway collision, what is the increased likelihood of secondary accidents due to traffic build-up?"*
                -   **Relevance:** Critical for modeling retaliatory gang violence and chain-reaction traffic incidents. It directly powers the `Trauma Clustering Score`.
                -   **Mathematical Formulation:** The conditional intensity `λ(t)` of an event at time `t` is defined as:
                    $$
                    \\lambda(t) = \\mu(t) + \\sum_{t_i < t} \\alpha \\cdot g(t - t_i)
                    $$
                    where `μ(t)` is the background rate from the NHPP, the sum is over past event times `tᵢ`, `α` is the branching ratio (strength of aftershock), and `g(t - tᵢ)` is the triggering kernel modeling the decaying influence of past events.
            
            *   **Bayesian Networks:** These models represent our understanding of causal relationships. They combine static base rates with real-time environmental factors.
                -   **Question it Answers:** *"How does a public holiday, combined with rainy weather and a major concert, collectively influence the probability of an incident?"*
                -   **Relevance:** Allows the system to reason with expert knowledge and adapt to contextual factors like `Weather`, `Is Holiday`, and `Major Event`. It is a core driver of the baseline `Incident Probability`.
                -   **Mathematical Formulation:** Based on the chain rule of probability, where the joint probability is the product of conditional probabilities: $P(X_1, ..., X_n) = \\prod_{i=1}^{n} P(X_i | \\text{Parents}(X_i))$. Our network models `P(IncidentRate | Weather, Holiday, ...)` to find the most likely baseline rate.
            """)
            
            st.markdown("---")
            st.markdown("#### **B. Spatiotemporal & Graph Models (The 'Where' and 'How It Spreads')**")
            st.markdown("""
            *   **Spatiotemporal Gaussian Processes (ST-GPs):** Our `STGP_Risk` KPI is a proxy for this advanced technique. It models risk as a continuous fluid over the map.
                -   **Question it Answers:** *"An incident occurred 500 meters from this zone's border. How much 'risk pressure' does that exert on this zone?"*
                -   **Relevance:** Interpolates risk intelligently across the map, ensuring that proximity to danger is always accounted for, even across arbitrary zone boundaries.

            *   **Graph Neural Networks (GNNs):** The city's road network and zone adjacencies are treated as a complex graph. A GNN learns a deep, structural understanding of each zone's role within this network.
                -   **Question it Answers:** *"Is this zone inherently vulnerable simply due to its position as a major crossroads, regardless of recent events?"*
                -   **Relevance:** Identifies long-term, structural vulnerabilities that may not be apparent from recent incident data alone. It powers the `GNN_Structural_Risk`, representing a zone's intrinsic risk.
            
            *   **Graph Laplacian Diffusion:** This technique models how effects (like traffic, panic, or police cordons) "spill over" from one zone to its neighbors through the road network.
                -   **Question it Answers:** *"A major fire has shut down three blocks in Zone A. How does this increase the traffic-related accident risk in the adjacent Zone B?"*
                -   **Relevance:** Essential for modeling the secondary effects of major incidents. It directly calculates the `Spatial Spillover Risk`.
            """)

            st.markdown("---")
            st.markdown("#### **C. Complexity & Information Theory (The 'System State')**")
            st.markdown("""
            *   **Lyapunov Exponents (Chaos Sensitivity Score):** A concept from Chaos Theory that measures a system's sensitivity to small changes. A high score means the system is in a fragile, unpredictable state.
                -   **Question it Answers:** *"Is the city operating normally, or is it in a 'brittle' state where one small incident could cascade into a major crisis?"*
                -   **Relevance:** This is a critical "instability alarm" for command staff. It doesn't predict a specific incident, but warns that the entire system is volatile.

            *   **Kullback-Leibler (KL) Divergence (Anomaly Score):** An information theory metric that measures how much the current pattern of incidents deviates from the historical norm.
                -   **Question it Answers:** *"Are we seeing the right number of incidents, but in all the wrong places today? Or are we seeing a bizarre new type of incident we've never seen before?"*
                -   **Relevance:** Detects "pattern anomalies" that simple volume-based metrics would miss. A high score is a clear signal that "today is not a normal day."
                -   **Mathematical Formulation:** 
                    $$
                    D_{KL}(P || Q) = \\sum_{z \\in \\text{Zones}} P(z) \\log{\\frac{P(z)}{Q(z)}}
                    $$
            """)

        with st.expander("III. The Prescription Engine: Optimal Resource Allocation", expanded=False):
            st.info("#### Core Principle: Moving from 'what will happen' to 'what is the best thing to do'.", icon="🎯")
            st.markdown("""
            The prescriptive engine uses the risk scores from the prediction layer as inputs into sophisticated Operations Research models. This ensures that resource allocation is not just intuitive, but mathematically optimal based on our objectives.

            *   **Mixed-Integer Linear Programming (MILP):** This is the workhorse for `Linear Optimal` allocation. It finds the provably best way to assign a whole number of ambulances to zones.
                -   **Objective:** Maximize the total amount of risk "covered" across the entire city.
                -   **Relevance:** Excellent for finding the most efficient solution under a single, clear objective. It is fast and guarantees a mathematically optimal result for a linear problem.
                -   **Mathematical Formulation (Simplified):**
                    $$
                    \\begin{aligned}
                    & \\text{maximize} && \\sum_{i \\in \\text{Zones}} R_i \\cdot c_i \\\\
                    & \\text{subject to} && \\sum_{i \\in \\text{Zones}} c_i \\leq N, \\quad c_i \\in \\mathbb{Z}^+
                    \\end{aligned}
                    $$
                    where `Rᵢ` is the risk score for zone `i`, `cᵢ` is the integer number of ambulances assigned, and `N` is the total available.
            *   **Non-Linear Programming (NLP):** This is our most advanced model for `Non-Linear Optimal` allocation. It captures complex, real-world dynamics that linear models miss.
                -   **Objective:** Minimize a "system dissatisfaction" function, which includes two key non-linear effects:
                    1.  **Diminishing Returns (Logarithmic Utility):** The first ambulance sent to a zone provides a huge benefit; the fifth provides much less. The model understands this and avoids over-saturating a single high-risk zone if another zone has zero coverage.
                    2.  **Congestion Penalties (Quadratic Penalty):** As the number of expected incidents in a zone vastly outpaces the allocated units, the "harm" (e.g., response time delay) grows exponentially, not linearly.
                -   **Relevance:** This provides the most realistic and robust recommendations. It makes intelligent trade-offs that a human or a simpler model might miss, leading to a more resilient overall system posture.
                -   **Mathematical Formulation (Simplified):**
                    $$
                    \\begin{aligned}
                    & \\text{minimize} && \\sum_{i \\in \\text{Zones}} \\left( w_1(R_i - R_i \\log(1+c_i)) + w_2 \\left( \\frac{E_i}{1+c_i} \\right)^2 \\right) \\\\
                    & \\text{subject to} && \\sum_{i \\in \\text{Zones}} c_i = N, \\quad c_i \\geq 0
                    \\end{aligned}
                    $$
                    where `Eᵢ` is expected incidents, the `log` term models **diminishing returns**, and the quadratic term models **congestion penalties**.
            *   **Queueing Theory:** This mathematical theory is conceptually used to model system strain, particularly at hospitals.
                -   **Relevance:** By understanding arrival rates (from our predictions) and service rates, we can better estimate wait times and the impact of hospital diversions, which feeds into the `Resource Adequacy Index`.
           
            """)
        
        with st.expander("IV. Incident-Specific Model Weighting", expanded=False):
            st.markdown("""
            The system is not one-size-fits-all. The final `Integrated_Risk_Score` is a weighted sum of many model outputs, and these weights are dynamically influenced by the nature of the risk being assessed.

            #### **Trauma - Violence**
            *   **Primary Predictive Models:** **Hawkes Processes** are paramount, as they explicitly model the retaliatory, self-exciting nature of violence. **GNN Structural Risk** is also critical for identifying long-term territorial hotspots.
            *   **Primary Prescriptive Model:** **NLP** is often preferred to not only cover risk but also to avoid over-saturating one area, which can be crucial in fluid tactical situations.

            #### **Trauma - Accidents**
            *   **Primary Predictive Models:** **Bayesian Networks** (incorporating weather and traffic) and **Graph Laplacians** (modeling spillover from congestion) are the key drivers.
            *   **Primary Prescriptive Model:** **NLP** is highly effective here as its built-in congestion penalty directly models the real-world consequence of traffic jams, leading to smarter staging decisions.

            #### **Medical Emergencies**
            *   **Primary Predictive Models:** **Bayesian Networks** are crucial for incorporating environmental factors like heatwaves and air quality. Spatiotemporal models analyzing population density and demographics (e.g., age) are also key.
            *   **Primary Prescriptive Model:** The choice of model is heavily influenced by **Hospital Divert Status**. When hospitals are under strain, the prescriptive models must weigh not just the risk of an incident, but the added travel time and risk of delay upon arrival, a factor that NLP can incorporate more naturally.
            """)

        with st.expander("V. Key Performance Indicator (KPI) Glossary", expanded=False):
            kpi_defs = {
                "Integrated Risk Score": "**The final, primary risk metric** used for all decisions, blending foundational and advanced AI models.",
                "Ensemble Risk Score": "Blended risk score from the foundational (Layer 1) models.",
                "GNN Structural Risk": "A zone's intrinsic vulnerability due to its position in the road network.",
                "STGP Risk": "Risk from proximity to recent, severe incidents (spatiotemporal correlation).",
                "Game Theory Tension": "A measure of a zone's contribution to system-wide resource competition.",
                "Chaos Sensitivity Score": "Measures system volatility and fragility. High score = 'The system is unstable.'",
                "Anomaly Score": "Measures the 'strangeness' of the current incident pattern compared to history.",
                "Resource Adequacy Index": "System-wide ratio of available units to expected demand, penalized by hospital strain."
            }
            for kpi, definition in kpi_defs.items():
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.markdown(f"**{kpi}**")
                with c2:
                    st.markdown(definition)
                st.markdown("---", unsafe_allow_html=True)

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
