# main.py
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime
import logging
import warnings
from pathlib import Path
# --- EXPANSION: Import Plotly for advanced visualizations ---
import plotly.express as px
import plotly.graph_objects as go

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
            st.session_state['env_factors'] = EnvFactors(
                is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, 
                population_density=avg_pop_density, air_quality_index=50.0, heatwave_alert=False,
                day_type='Weekday', time_of_day='Midday', public_event_type='None',
                hospital_divert_status=0.0, police_activity='Normal', school_in_session=True
            )

    def render(self):
        """Main rendering loop for the Streamlit application."""
        st.title("RedShield AI: Phoenix v4.0")
        st.markdown("##### Proactive Emergency Response & Resource Allocation Platform")

        self._render_sidebar()

        env_factors = st.session_state['env_factors']
        historical_data = st.session_state['historical_data']

        with st.spinner("Executing Advanced Analytics Pipeline..."):
            current_incidents = self.dm.get_current_incidents(env_factors)
            kpi_df = self.engine.generate_kpis(historical_data, env_factors, current_incidents)
            forecast_df = self.engine.generate_forecast(kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df)
        
        st.session_state['kpi_df'] = kpi_df
        st.session_state['forecast_df'] = forecast_df
        st.session_state['allocations'] = allocations

        # --- UI OVERHAUL: Use tabs for a clean, organized interface ---
        tab1, tab2, tab3 = st.tabs(["🔥 Operational Dashboard", "📊 KPI Deep Dive", "🧠 Methodology & Insights"])

        with tab1:
            self._render_main_dashboard_tab(kpi_df, allocations, current_incidents)
        with tab2:
            self._render_kpi_deep_dive_tab(kpi_df, forecast_df)
        with tab3:
            self._render_methodology_tab()

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
        new_env = EnvFactors(
            is_holiday=is_holiday, weather=weather, traffic_level=traffic, major_event=major_event,
            population_density=env.population_density, air_quality_index=aqi, heatwave_alert=heatwave,
            day_type=day_type, time_of_day=time_of_day, public_event_type=public_event_type,
            hospital_divert_status=hospital_divert_status, police_activity=police_activity, school_in_session=school_in_session
        )

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

    def _render_main_dashboard_tab(self, kpi_df, allocations, incidents):
        st.subheader("Current Situational Overview")
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active Incidents", len(incidents), help="Number of incidents currently being processed.")
        c2.metric("Available Ambulances", sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible'), delta_color="off")
        c3.metric("Highest Zone Risk", f"{kpi_df['Integrated_Risk_Score'].max():.3f}", help="The peak Integrated Risk Score across all zones.")
        c4.metric("System Adequacy", f"{kpi_df['Resource Adequacy Index'].mean():.1%}", help="Ratio of available units to system-wide expected incidents, penalized by hospital strain.")

        st.divider()

        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Live Risk Heatmap")
            self._render_map(kpi_df)
        with col2:
            st.subheader("Recommended Ambulance Allocation")
            alloc_df = pd.DataFrame(list(allocations.items()), columns=['Zone', 'Recommended Units']).set_index('Zone')
            st.dataframe(alloc_df, use_container_width=True)
            
            st.subheader("Top 5 Highest-Risk Zones")
            top_zones = kpi_df[['Zone', 'Integrated_Risk_Score']].sort_values(by='Integrated_Risk_Score', ascending=False).head(5)
            st.dataframe(top_zones.set_index('Zone').style.format("{:.3f}"), use_container_width=True)

    def _render_kpi_deep_dive_tab(self, kpi_df, forecast_df):
        st.subheader("Comprehensive Risk Indicator Matrix")
        st.markdown("This table shows the complete set of calculated KPIs for each zone, including base, ensemble, and advanced model outputs.")
        st.dataframe(kpi_df.set_index('Zone').style.format("{:.3f}").background_gradient(cmap='viridis', axis=0))
        
        st.divider()
        
        # --- EXPANSION: NEW SECTION FOR ADVANCED VISUALIZATIONS ---
        st.subheader("Advanced Analytical Plots")
        
        if not kpi_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                self._plot_risk_contribution_sunburst(kpi_df)
            with c2:
                self._plot_advanced_model_correlation(kpi_df)
            
            st.divider()
            self._plot_vulnerability_quadrant(kpi_df)
        else:
            st.info("Insufficient data to generate advanced analytical plots.")
        # --- END OF EXPANSION ---
        
        st.divider()
        st.subheader("Risk Forecast by Zone")
        st.markdown("Projected `Integrated_Risk_Score` over the next 72 hours.")
        if not forecast_df.empty:
            forecast_pivot = forecast_df.pivot(index='Zone', columns='Horizon (Hours)', values='Combined Risk')
            st.dataframe(forecast_pivot.style.format("{:.3f}").background_gradient(cmap='YlOrRd', axis=1), use_container_width=True)

    def _render_map(self, kpi_df):
        if self.dm.zones_gdf.empty or kpi_df.empty: return
        try:
            map_gdf = self.dm.zones_gdf.join(kpi_df.set_index('Zone'))
            map_gdf.reset_index(inplace=True)
            center = map_gdf.unary_union.centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=12, tiles="cartodbpositron", prefer_canvas=True)
            
            risk_col = 'Integrated_Risk_Score'
            choropleth = folium.Choropleth(
                geo_data=map_gdf.to_json(), data=map_gdf,
                columns=['name', risk_col], key_on='feature.properties.name',
                fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2, legend_name='Integrated Risk Score'
            ).add_to(m)
            
            tooltip_html = "<b>Zone:</b> {name}<br><b>Integrated Risk:</b> {risk:.3f}"
            map_gdf['tooltip'] = map_gdf.apply(lambda row: tooltip_html.format(name=row['name'], risk=row[risk_col]), axis=1)
            folium.GeoJson(map_gdf, style_function=lambda x: {'color': 'black', 'weight': 1, 'fillOpacity': 0},
                           tooltip=folium.features.GeoJsonTooltip(fields=['tooltip'], labels=False)).add_to(m)
            
            st_folium(m, use_container_width=True, height=550)
        except Exception as e:
            logger.error(f"Failed to render map: {e}", exc_info=True)
            st.error(f"Error rendering map: {e}")

    def _render_methodology_tab(self):
        st.header("System Architecture & Methodology")
        st.markdown("""
        ### I. High-Level Goal & Architectural Philosophy

        The fundamental goal of RedShield AI: Phoenix v4.0 is to engineer a paradigm shift in emergency response—from a traditional **reactive model** (dispatching units after an incident occurs) to a **proactive, predictive posture** (anticipating where incidents are likely to emerge and pre-positioning resources to minimize response times and maximize impact).

        To achieve this, the system is built on a philosophy of **Hierarchical Ensemble Modeling**. Instead of relying on a single algorithm, Phoenix v4.0 integrates a diverse portfolio of analytical techniques in a layered architecture. This creates a highly robust and resilient system where the weaknesses of any one model are offset by the strengths of others, generating a final prediction that represents a true "wisdom of the crowds."

        The system architecture is composed of three primary analytical layers:

        1.  **Layer 1: Foundational Ensemble.** This layer consists of well-established statistical and first-principle models (Hawkes Processes, SIR, Bayesian Networks, Graph Laplacians) that create a robust baseline understanding of risk. This produces the `Ensemble_Risk_Score`.
        2.  **Layer 2: Advanced AI & Complexity Proxies.** This layer introduces computationally inexpensive but analytically powerful proxies for cutting-edge AI and complexity science models (ST-GPs, HMMs, GNNs, Game Theory). These models capture deeper, more nuanced patterns that complement the foundational layer.
        3.  **Layer 3: Integrated Synthesis.** The outputs of the first two layers are combined in a final, weighted synthesis to produce the ultimate `Integrated_Risk_Score`, which drives the system's final recommendations.

        ---

        ### II. Detailed Methodology Breakdown

        This section details the theoretical underpinnings of the models and techniques that power the Phoenix v4.0 engine.

        #### **I. Stochastic & Statistical Modeling**

        *   **Non-Homogeneous Poisson Process (NHPP):** Used to model the baseline incident rate (`μ`). Unlike a standard Poisson process with a constant rate, the NHPP allows this rate to vary over time, capturing predictable patterns like higher call volumes during evening rush hour versus midday.
        *   **Hawkes Process (Self-Exciting):** This is the core of our trauma and violence clustering models. It assumes that some events can trigger "aftershocks."
            > *Mathematical Intuition:* `λ(t) = μ(t) + Σ α * exp[-β(t-tᵢ)]`
            > The incident rate `λ(t)` at time `t` is the sum of the time-varying baseline `μ(t)` and the decaying influence `exp[...]` of every past incident `tᵢ`.
            > *Significance:* Answers the question: *"Given a shooting just occurred, what is the immediate, elevated risk of another shooting in the same area?"* It's crucial for modeling gang-related violence and cascading traffic accidents.
        *   **Marked Point Processes:** The system implicitly uses this concept by treating each incident not just as a point in time and space, but as a "marked" event with metadata (e.g., `type: 'Trauma-Accident'`, `triage: 'Red'`). This allows different models to respond selectively to different types of marks.

        #### **II. Spatiotemporal & Graph Models**

        *   **Spatiotemporal Gaussian Processes (ST-GPs):** Our `STGP_Risk` KPI is a proxy for this technique. A full ST-GP would model incident intensity as a continuous function over space and time, providing robust predictions with confidence bounds even in areas with no data.
            > *Significance:* Answers the question: *"What is the likely risk in this park, which is 1km away from a major incident, even if no calls have come from the park itself?"* It interpolates risk intelligently across the map.
        *   **Dynamic Spatial Graphs / Graph Laplacians:** The `Spatial Spillover Risk` KPI is a direct implementation of this. The city's road network is treated as a graph, and the Graph Laplacian matrix is used to model the diffusion of risk from one zone to its neighbors, simulating how traffic or chaos can spread.
        *   **Graph Neural Networks (GNNs):** Our `GNN_Structural_Risk` KPI is a proxy for a GNN. A full GNN would learn a dense vector representation (embedding) for each zone based on its structural properties and incident history.
            > *Significance:* Identifies zones that are inherently vulnerable due to their position in the network (e.g., a central hub with many connections), regardless of recent incident history. It represents a deep, structural understanding of the urban environment.

        #### **III. Deep Learning Architectures**

        *   **Temporal Convolutional Networks (TCNs):** The system is equipped with a TCNN module. TCNs use convolutions over time, allowing them to capture long-range temporal patterns with high stability and efficiency, making them ideal for high-resolution time-series forecasting.
        *   **Other Architectures (Conceptual):** While not all implemented as full models, the system's architecture is designed to incorporate proxies or future versions of:
            *   **Transformers (e.g., TimeGPT):** For very long-range forecasting (e.g., weekly or monthly trends).
            *   **ConvLSTM / 3D CNNs:** To predict the evolution of a 2D risk heatmap over time.
            *   **Variational Autoencoders (VAEs):** To learn the "latent language" of incident patterns, ideal for detecting highly unusual, never-before-seen anomaly types.

        #### **IV. Hybrid & Adaptive Systems**

        *   **Bayesian Deep Learning:** The combination of a statistical Bayesian Network with a Deep Learning TCNN is a form of Bayesian Deep Learning, blending probabilistic reasoning with high-dimensional feature extraction.
        *   **Concept Drift Adaptation:** The system is designed to handle concept drift. By allowing a user to upload new historical data, the models can be implicitly retrained and re-cached, adapting to fundamental shifts in urban dynamics (e.g., post-pandemic traffic patterns).

        #### **V. Chaos Theory & Complexity Science**

        *   **Lyapunov Exponents:** The `Chaos Sensitivity Score` is a direct proxy for this. It measures the system's sensitivity to initial conditions.
            > *Significance:* It is an "instability alarm." A high score doesn't predict a specific incident, but it warns command staff that the entire system is in a fragile, unpredictable state where a small event could cascade into a major crisis.
        *   **Agent-Based Modeling (ABM):** The `Game_Theory_Tension` KPI is a macro-level outcome of what an ABM would simulate. An ABM would model individual agents (ambulances, cars, people) and their interactions, from which competitive bottlenecks for resources emerge.

        #### **VI. Information Theory**

        *   **Shannon Entropy (`Risk Entropy`):** Quantifies the level of disorder or uncertainty in the spatial distribution of risk. A low entropy value is desirable, indicating that risk is concentrated in predictable "hotspots." High entropy means risk is spread thinly and evenly, making resource allocation more difficult.
        *   **Kullback-Leibler (KL) Divergence (`Anomaly Score`):** Measures how much the current incident distribution `P(x)` diverges from the historical baseline `Q(x)`.
            > *Significance:* It detects "pattern anomalies." It answers: *"Are we seeing the right types of incidents, but just in the wrong places today?"* A high score indicates that the city is behaving unusually.
        *   **Mutual Information (`Information Value Index`):** This KPI is a proxy for mutual information. It quantifies how much "information" the current risk map provides for making a decision. A high value (high standard deviation between zone risks) means there are clear high-risk and low-risk zones, making the decision of where to send resources easy and impactful. A low value means all zones have similar risk, and the prediction is less valuable for differentiating.

        #### **VII. Simulation, Optimization, and Strategic Layers**

        *   **Game Theory Models:** The `Game_Theory_Tension` KPI is a proxy for this. It models the city as a non-cooperative game where each zone "competes" for a finite pool of EMS resources. A high tension score for a zone means it is a major driver of resource competition.
        *   **Operational Research Models:** The final allocation algorithm is a direct application of operational research principles. It is a **proportional risk allocation** strategy that optimally distributes a fixed number of assets (`N` ambulances) across a set of targets (zones) based on their weighted risk scores, solving a classic resource optimization problem.
        
        ---

        ### **III. Key Performance Indicator (KPI) Glossary**

        *   **Incident Probability:** The baseline probability (0-1) of an incident occurring in a zone, primarily driven by the Bayesian Network and environmental factors.
        *   **Expected Incident Volume:** A Poisson-based estimate of the *number* of incidents to expect in a zone over a short time horizon.
        *   **Risk Entropy:** Measures the *uncertainty* of the spatial risk distribution. High entropy = unpredictable risk. Low entropy = concentrated hotspots.
        *   **Anomaly Score:** Measures the "strangeness" of the current incident pattern compared to history. High score = "Today is not a normal day."
        *   **Spatial Spillover Risk:** Risk "leaking" from neighboring zones, based on the Graph Laplacian.
        *   **Resource Adequacy Index:** Ratio of available ambulances to system-wide expected demand, penalized by hospital strain.
        *   **Chaos Sensitivity Score:** Measures system volatility and fragility. High score = "The system is unstable."
        *   **Bayesian Confidence Score:** How certain the Bayesian Network is about its predictions.
        *   **Information Value Index:** How "actionable" the current risk map is. High value = clear hotspots.
        *   **STGP Risk:** (Advanced) Risk from proximity to recent, severe incidents.
        *   **HMM State Risk:** (Advanced) Risk from being in a latent "Agitated" or "Critical" state.
        *   **GNN Structural Risk:** (Advanced) A zone's intrinsic vulnerability due to its position in the road network.
        *   **Game Theory Tension:** (Advanced) A measure of a zone's contribution to resource competition.
        *   **Ensemble Risk Score:** The blended risk score from the foundational (Layer 1) models.
        *   **Integrated Risk Score:** The ultimate, final risk metric, combining the Ensemble score with all Advanced AI & Complexity KPIs. **This is the primary score used for forecasting and allocation.**
        """)

    # --- EXPANSION: NEW PLOTTING METHODS ADDED TO THE CLASS ---
    def _plot_risk_contribution_sunburst(self, kpi_df: pd.DataFrame):
        st.markdown("**Risk Contribution Analysis**")
        st.help("This sunburst chart breaks down the final `Integrated_Risk_Score` for the highest-risk zone, showing the contribution of each analytical layer.")
        
        highest_risk_zone = kpi_df.loc[kpi_df['Integrated_Risk_Score'].idxmax()]
        zone_name = highest_risk_zone['Zone']

        adv_weights = self.config['model_params'].get('advanced_model_weights', {})
        
        data = {
            'ids': [
                'Integrated Risk', 'Base Ensemble', 'Advanced Models', 
                'STGP', 'HMM', 'GNN', 'Game Theory'
            ],
            'labels': [
                f"Total: {highest_risk_zone['Integrated_Risk_Score']:.2f}", 
                'Base Ensemble', 'Advanced Models', 
                'STGP Risk', 'HMM State', 'GNN Structure', 'Game Tension'
            ],
            'parents': [
                '', 'Integrated Risk', 'Integrated Risk', 
                'Advanced Models', 'Advanced Models', 'Advanced Models', 'Advanced Models'
            ],
            'values': [
                highest_risk_zone['Integrated_Risk_Score'],
                adv_weights.get('base_ensemble', 0) * highest_risk_zone['Ensemble Risk Score'],
                (adv_weights.get('stgp', 0) * highest_risk_zone['STGP_Risk'] +
                 adv_weights.get('hmm', 0) * highest_risk_zone['HMM_State_Risk'] +
                 adv_weights.get('gnn', 0) * highest_risk_zone['GNN_Structural_Risk'] +
                 adv_weights.get('game_theory', 0) * highest_risk_zone['Game_Theory_Tension']),
                adv_weights.get('stgp', 0) * highest_risk_zone['STGP_Risk'],
                adv_weights.get('hmm', 0) * highest_risk_zone['HMM_State_Risk'],
                adv_weights.get('gnn', 0) * highest_risk_zone['GNN_Structural_Risk'],
                adv_weights.get('game_theory', 0) * highest_risk_zone['Game_Theory_Tension']
            ]
        }

        fig = go.Figure(go.Sunburst(
            ids=data['ids'], labels=data['labels'], parents=data['parents'], 
            values=data['values'], branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Contribution: %{value:.3f}<extra></extra>',
        ))
        fig.update_layout(
            margin = dict(t=20, l=0, r=0, b=0),
            title_text=f"Risk Breakdown for Zone: {zone_name}",
            title_x=0.5,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_advanced_model_correlation(self, kpi_df: pd.DataFrame):
        st.markdown("**Advanced Model Correlation**")
        st.help("This heatmap shows the correlation between the advanced KPI scores across all zones. High correlation suggests models are identifying similar underlying risk patterns.")
        
        advanced_cols = ['STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk', 'Game_Theory_Tension']
        corr_df = kpi_df[advanced_cols].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_df.values,
            x=corr_df.columns,
            y=corr_df.columns,
            colorscale='Blues',
            text=corr_df.values,
            texttemplate="%{text:.2f}",
            hoverongaps = False))
        fig.update_layout(
            margin = dict(t=20, l=0, r=0, b=0),
            height=400,
            title_text="Correlation Matrix of Advanced KPIs",
            title_x=0.5
        )
        st.plotly_chart(fig, use_container_width=True)

    def _plot_vulnerability_quadrant(self, kpi_df: pd.DataFrame):
        st.markdown("**Zone Vulnerability Quadrant Analysis**")
        st.help("This plot segments zones based on their long-term structural vulnerability versus their immediate real-time risk, identifying latent threats and acute hotspots.")

        fig = px.scatter(
            kpi_df,
            x="Ensemble Risk Score",
            y="GNN_Structural_Risk",
            color="Integrated_Risk_Score",
            size="Expected Incident Volume",
            hover_name="Zone",
            color_continuous_scale="reds",
            size_max=18
        )
        
        x_mean = kpi_df['Ensemble Risk Score'].mean()
        y_mean = kpi_df['GNN_Structural_Risk'].mean()
        
        fig.add_vline(x=x_mean, line_width=1, line_dash="dash", line_color="grey")
        fig.add_hline(y=y_mean, line_width=1, line_dash="dash", line_color="grey")
        
        fig.add_annotation(x=x_mean/2, y=y_mean*1.8 if y_mean > 0 else 0.8, text="<b>Latent Threats</b><br>(High Vulnerability, Low Risk)", showarrow=False, font=dict(color="navy"))
        fig.add_annotation(x=x_mean*1.5 if x_mean > 0 else 0.8, y=y_mean*1.8 if y_mean > 0 else 0.8, text="<b>Crisis Zones</b><br>(High Vulnerability, High Risk)", showarrow=False, font=dict(color="red"))
        fig.add_annotation(x=x_mean*1.5 if x_mean > 0 else 0.8, y=y_mean/2, text="<b>Acute Hotspots</b><br>(Low Vulnerability, High Risk)", showarrow=False, font=dict(color="darkorange"))
        fig.add_annotation(x=x_mean/2, y=y_mean/2, text="<b>Quiet Zones</b>", showarrow=False, font=dict(color="green"))
        
        fig.update_layout(
            xaxis_title="Dynamic Risk (Real-time Threat)",
            yaxis_title="Structural Vulnerability (Intrinsic Threat)",
            coloraxis_colorbar_title_text='Integrated<br>Risk'
        )
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
