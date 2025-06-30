# main.py
"""
RedShield AI: Phoenix v4.0 - Proactive Emergency Response Platform

This is the main entry point for the Streamlit application. It orchestrates the
user interface, data management, and predictive analytics engine to deliver
a real-time, interactive dashboard for emergency response command staff.

The application is structured into three main tabs:
1. Operational Command: A live map and decision support gauges for immediate action.
2. KPI Deep Dive: Advanced analytics and visualizations for deeper insights.
3. Methodology & Insights: A detailed explanation of the underlying models.
"""

import logging
import warnings
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

# --- CORRECTED: System Setup MUST be the first Streamlit command ---
st.set_page_config(
    page_title="RedShield AI: Phoenix v4.0",
    layout="wide",
    initial_sidebar_state="expanded"
)

import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# --- Import from refactored modules (AFTER st.set_page_config) ---
from core import DataManager, EnvFactors, PredictiveAnalyticsEngine
from utils import ReportGenerator, load_config

# --- Optimized: Centralized Application Constants ---
CONSTANTS = {
    'RISK_COVERAGE_PER_UNIT': 0.25,
    'PRESSURE_WEIGHTS': {'traffic': 0.3, 'hospital': 0.4, 'adequacy': 0.3},
    'TRAFFIC_MIN': 0.5,
    'TRAFFIC_MAX': 3.0,
    'FALLBACK_POP_DENSITY': 50000,
    'FLOAT_TOLERANCE': 1e-6
}

# --- Post-Config Setup ---
st.cache_data.clear()
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- Logging Configuration ---
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/redshield_phoenix.log")
    ]
)
logger = logging.getLogger(__name__)


# --- Optimized: Custom Data Class for Efficient Reruns ---
class EnvFactorsWithTolerance(EnvFactors):
    """
    Extends EnvFactors to include a custom equality check with a float tolerance.
    This prevents unnecessary Streamlit reruns from minor floating-point noise.
    """
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EnvFactors):
            return NotImplemented

        for attr in ['traffic_level', 'air_quality_index', 'hospital_divert_status']:
            if abs(getattr(self, attr, 0) - getattr(other, attr, 0)) > CONSTANTS['FLOAT_TOLERANCE']:
                return False
        for attr in ['is_holiday', 'weather', 'major_event', 'heatwave_alert',
                     'day_type', 'time_of_day', 'public_event_type',
                     'police_activity', 'school_in_session']:
            if getattr(self, attr, None) != getattr(other, attr, None):
                return False
        return True


# --- Main Dashboard Class ---
class Dashboard:
    """Handles the rendering of the Streamlit user interface for Phoenix v4.0."""

    def __init__(self, dm: DataManager, engine: PredictiveAnalyticsEngine):
        self.dm = dm
        self.engine = engine
        self.config = dm.config
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initializes all required session state keys if they don't exist."""
        if 'avg_pop_density' not in st.session_state:
            if not self.dm.zones_gdf.empty and 'population' in self.dm.zones_gdf.columns:
                st.session_state['avg_pop_density'] = self.dm.zones_gdf['population'].mean()
            else:
                st.session_state['avg_pop_density'] = CONSTANTS['FALLBACK_POP_DENSITY']
                logger.warning("zones_gdf empty or missing 'population'; using fallback density.")

        default_states = {
            'historical_data': [], 'kpi_df': pd.DataFrame(), 'forecast_df': pd.DataFrame(),
            'allocations': {}, 'sparkline_data': {}, 'current_incidents': [],
            'env_factors': EnvFactorsWithTolerance(
                is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False,
                population_density=st.session_state['avg_pop_density'],
                air_quality_index=50.0, heatwave_alert=False, day_type='Weekday',
                time_of_day='Midday', public_event_type='None',
                hospital_divert_status=0.0, police_activity='Normal', school_in_session=True
            )
        }
        for key, value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def render(self):
        """Main rendering method for the entire dashboard."""
        st.title("RedShield AI: Phoenix v4.0")
        st.markdown("##### Proactive Emergency Response & Resource Allocation Platform")
        self._render_sidebar()

        with st.spinner("Executing Advanced Analytics & Optimization Pipeline..."):
            self._run_analytics_pipeline()

        tab1, tab2, tab3 = st.tabs(["🔥 Operational Command", "📊 KPI Deep Dive", "🧠 Methodology & Insights"])
        with tab1: self._render_operational_command_tab()
        with tab2: self._render_kpi_deep_dive_tab()
        with tab3: self._render_methodology_tab()

    def _run_analytics_pipeline(self):
        """Executes the analytics pipeline and updates session state."""
        try:
            env_factors = st.session_state.env_factors
            historical_data = st.session_state.historical_data
            current_incidents = self.dm.get_current_incidents(env_factors)
            kpi_df, sparkline_data = self.engine.generate_kpis_with_sparklines(
                historical_data, env_factors, current_incidents
            )
            forecast_df = self.engine.generate_forecast(kpi_df)
            allocations = self.engine.generate_allocation_recommendations(kpi_df)
            st.session_state.update({
                'kpi_df': kpi_df, 'forecast_df': forecast_df, 'allocations': allocations,
                'sparkline_data': sparkline_data, 'current_incidents': current_incidents
            })
        except Exception as e:
            logger.error(f"Analytics pipeline failed: {e}", exc_info=True)
            st.error(f"Error in analytics pipeline: {e}. Check logs for details.")
            st.session_state.update({
                'kpi_df': pd.DataFrame(), 'forecast_df': pd.DataFrame(),
                'allocations': {}, 'sparkline_data': {}
            })

    # --- TAB 1: OPERATIONAL COMMAND ---

    def _render_operational_command_tab(self):
        """Renders the main operational command view."""
        self._render_system_status_bar()
        st.divider()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Live Operations Map")
            map_object = self._render_dynamic_map(
                kpi_df=st.session_state.kpi_df,
                incidents=st.session_state.current_incidents,
                _zones_gdf=self.dm.zones_gdf,
                _ambulances=self.dm.ambulances,
            )
            if map_object:
                st_folium(map_object, use_container_width=True, height=600)
        with col2:
            st.subheader("Decision Support")
            self._plot_system_pressure_gauge()
            self._plot_resource_to_risk_adequacy()

    @staticmethod
    def _create_status_metric(label: str, value: str, trend: str, color: str) -> str:
        return f"""<div style="flex:1;background-color:{color};padding:10px;text-align:center;color:white;border-right:1px solid #fff4;"><div style="font-size:1.5rem;font-weight:bold;">{value} {trend}</div><div style="font-size:0.8rem;">{label}</div></div>"""

    def _render_system_status_bar(self):
        st.subheader("System Health Status")
        kpi_df, spark_data = st.session_state.kpi_df, st.session_state.sparkline_data
        if kpi_df.empty or not spark_data:
            st.info("System status unavailable: Waiting for data..."); return
        try:
            inc_val, amb_val = len(st.session_state.current_incidents), sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
            risk_val, adeq_val = kpi_df['Integrated_Risk_Score'].max(), kpi_df['Resource Adequacy Index'].mean()
            inc_data, amb_data, risk_data, adeq_data = (
                spark_data.get('active_incidents', {'values': [inc_val], 'range': [inc_val-1, inc_val+1]}),
                spark_data.get('available_ambulances', {'values': [amb_val], 'range': [amb_val-1, amb_val+1]}),
                spark_data.get('max_risk', {'values': [risk_val], 'range': [0, 1]}),
                spark_data.get('adequacy', {'values': [adeq_val], 'range': [0, 1]}))
            def get_color(v, r, high_is_bad=True): return "#D32F2F" if (high_is_bad and v > r[1]) or (not high_is_bad and v < r[0]) else "#FBC02D" if (high_is_bad and v > r[0]) or (not high_is_bad and v < r[1]) else "#388E3C"
            def get_trend(d): return "▲" if len(d)>1 and d[-1]>d[-2] else "▼" if len(d)>1 and d[-1]<d[-2] else "▬"
            metrics = [self._create_status_metric("Active Incidents",f"{inc_val}",get_trend(inc_data['values']),get_color(inc_val,inc_data['range'])), self._create_status_metric("Available Units",f"{amb_val}",get_trend(amb_data['values']),get_color(amb_val,amb_data['range'],False)), self._create_status_metric("Max Zone Risk",f"{risk_val:.3f}",get_trend(risk_data['values']),get_color(risk_val,risk_data['range'])), self._create_status_metric("System Adequacy",f"{adeq_val:.1%}",get_trend(adeq_data['values']),get_color(adeq_val,adeq_data['range'],False))]
            metrics[-1] = metrics[-1].replace('border-right: 1px solid #fff4;', '')
            st.markdown(f'<div style="display:flex;border:1px solid #444;border-radius:5px;overflow:hidden;font-family:sans-serif;">{"".join(metrics)}</div>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error rendering status bar: {e}", exc_info=True)
            st.warning(f"Could not render system status bar: {e}")

    @st.cache_data(show_spinner="Rendering Operations Map...")
    def _render_dynamic_map(_self, kpi_df: pd.DataFrame, incidents: List[Dict], _zones_gdf: gpd.GeoDataFrame, _ambulances: Dict) -> Optional[folium.Map]:
        if _zones_gdf.empty or kpi_df.empty or 'Zone' not in kpi_df.columns:
            st.warning("Map data not yet available."); return None
        try:
            map_gdf = _zones_gdf.join(kpi_df.set_index('Zone'), on='name')
            # --- CORRECTED: Reset the index so 'name' becomes a column for Folium ---
            map_gdf.reset_index(inplace=True)
            
            center = _zones_gdf.union_all().centroid
            m = folium.Map(location=[center.y, center.x], zoom_start=11, tiles="cartodbpositron", prefer_canvas=True)
            folium.Choropleth(
                geo_data=map_gdf, data=map_gdf, columns=['name', 'Integrated_Risk_Score'],
                key_on='feature.properties.name', fill_color='YlOrRd', fill_opacity=0.7, line_opacity=0.2,
                legend_name='Integrated Risk Score', name="Risk Heatmap"
            ).add_to(m)
            incidents_fg = MarkerCluster(name='Live Incidents', show=True).add_to(m)
            for inc in incidents:
                if (loc := inc.get('location')) and 'lat' in loc and 'lon' in loc:
                    icon = "car-crash" if "Accident" in inc.get('type', '') else "first-aid"
                    folium.Marker([loc['lat'], loc['lon']], tooltip=f"Type: {inc.get('type','N/A')}<br>Triage: {inc.get('triage','N/A')}",
                                  icon=folium.Icon(color='red', icon=icon, prefix='fa')).add_to(incidents_fg)
            ambulance_fg = folium.FeatureGroup(name='Available Unit Reach (5-min)', show=False).add_to(m)
            for amb_id, amb_data in _ambulances.items():
                if amb_data.get('status') == 'Disponible':
                    loc = amb_data.get('location')
                    folium.Circle([loc.y, loc.x], radius=2400, color='#1E90FF', fill=True, fill_opacity=0.1, tooltip=f"Unit {amb_id} Reach").add_to(ambulance_fg)
                    folium.Marker([loc.y, loc.x], icon=folium.Icon(color='blue', icon='ambulance', prefix='fa'), tooltip=f"Unit {amb_id} (Available)").add_to(ambulance_fg)
            folium.LayerControl().add_to(m)
            return m
        except Exception as e:
            logger.error(f"Failed to render map: {e}", exc_info=True)
            st.error(f"Error rendering map: {e}")
            return None

    def _plot_system_pressure_gauge(self):
        try:
            kpi_df, env = st.session_state.kpi_df, st.session_state.env_factors
            if kpi_df.empty: return
            t_norm = np.clip((env.traffic_level-CONSTANTS['TRAFFIC_MIN'])/(CONSTANTS['TRAFFIC_MAX']-CONSTANTS['TRAFFIC_MIN']), 0, 1)
            h_norm, a_norm = env.hospital_divert_status, 1 - kpi_df['Resource Adequacy Index'].mean()
            score = (t_norm*CONSTANTS['PRESSURE_WEIGHTS']['traffic'] + h_norm*CONSTANTS['PRESSURE_WEIGHTS']['hospital'] + a_norm*CONSTANTS['PRESSURE_WEIGHTS']['adequacy']) * 125
            fig = go.Figure(go.Indicator(mode="gauge+number", value=min(score, 100), title={'text':"System Pressure"},
                gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#222"}, 'steps':[{'range':[0,40],'color':'#388E3C'},{'range':[40,75],'color':'#FBC02D'},{'range':[75,100],'color':'#D32F2F'}]}))
            fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in pressure gauge: {e}", exc_info=True); st.warning("Could not display System Pressure gauge.")

    def _plot_resource_to_risk_adequacy(self):
        try:
            kpi_df, allocations = st.session_state.kpi_df, st.session_state.allocations
            if kpi_df.empty or 'Integrated_Risk_Score' not in kpi_df.columns: st.info("No data for risk adequacy plot."); return
            df = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
            df['allocated'] = df['Zone'].map(allocations).fillna(0)
            risk_cov = self.config.get('model_params',{}).get('risk_coverage_per_unit', CONSTANTS['RISK_COVERAGE_PER_UNIT'])
            df['risk_covered'] = df['allocated'] * risk_cov
            df['ratio'] = np.clip((df['risk_covered']+1e-9) / (df['Integrated_Risk_Score']+1e-9), 0, 1.5)
            df['color'] = df['ratio'].apply(lambda r: '#D32F2F' if r < 0.7 else '#FBC02D' if r < 1.0 else '#388E3C')
            fig = go.Figure()
            fig.add_trace(go.Bar(y=df['Zone'], x=df['Integrated_Risk_Score'], orientation='h', name='Total Risk', marker_color='#e0e0e0', hovertemplate="<b>Zone:</b> %{y}<br><b>Total Risk:</b> %{x:.3f}<extra></extra>"))
            fig.add_trace(go.Bar(y=df['Zone'], x=df['risk_covered'], orientation='h', name='Covered Risk', marker_color=df['color'], text=df['allocated'].astype(int).astype(str)+" Unit(s)", textposition='inside', textfont=dict(color='white',size=12), hovertemplate="<b>Zone:</b> %{y}<br><b>Risk Covered:</b> %{x:.3f}<br><b>Allocated:</b> %{text}<extra></extra>"))
            fig.update_layout(title='Resource vs. Demand for High-Risk Zones', xaxis_title='Integrated Risk Score', yaxis_title=None, height=350, yaxis={'categoryorder':'total ascending'}, legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1), barmode='overlay', plot_bgcolor='white', margin=dict(l=10,r=10,t=70,b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**How to Read:** Grey bar is risk (demand). Colored bar is coverage (supply).")
        except Exception as e:
            logger.error(f"Error in adequacy plot: {e}", exc_info=True); st.warning("Could not display Resource Adequacy plot.")

    # --- TAB 2: KPI DEEP DIVE ---
    def _render_kpi_deep_dive_tab(self):
        st.subheader("Comprehensive Risk Indicator Matrix")
        if not (kpi_df := st.session_state.kpi_df).empty:
            st.dataframe(kpi_df.set_index('Zone').style.format("{:.3f}").background_gradient(cmap='viridis'), use_container_width=True)
        else: st.info("KPI data not yet available.")
        st.divider()
        st.subheader("Advanced Analytical Visualizations")
        if not kpi_df.empty:
            tab1, tab2, tab3 = st.tabs(["📍 Zone Vulnerability", "📊 Risk Drill-Down", "📈 Risk Forecast"])
            with tab1: self._plot_vulnerability_quadrant(kpi_df)
            with tab2: self._plot_risk_contribution_sunburst(kpi_df)
            with tab3: self._plot_forecast_with_uncertainty()
        else: st.info("Advanced visualizations unavailable: Waiting for data...")

    def _plot_vulnerability_quadrant(self, kpi_df: pd.DataFrame):
        st.markdown("**Analysis:** Segments zones by long-term structural vulnerability vs. immediate dynamic risk.")
        try:
            req = ['Ensemble Risk Score', 'GNN_Structural_Risk', 'Integrated_Risk_Score', 'Expected Incident Volume']
            if not all(c in kpi_df.columns for c in req): st.error("Data missing for Vulnerability plot."); return
            x_m, y_m = kpi_df['Ensemble Risk Score'].mean(), kpi_df['GNN_Structural_Risk'].mean()
            fig = px.scatter(kpi_df, x="Ensemble Risk Score", y="GNN_Structural_Risk", color="Integrated_Risk_Score", size="Expected Incident Volume", hover_name="Zone", color_continuous_scale="reds", size_max=18, hover_data={'Zone':False, 'Ensemble Risk Score':':.3f', 'GNN_Structural_Risk':':.3f'})
            fig.update_traces(hovertemplate="<b>Zone: %{hovertext}</b><br><br>Dynamic Risk: %{x:.3f}<br>Structural Risk: %{y:.3f}<extra></extra>")
            fig.add_vline(x=x_m, line_width=1, line_dash="dash", line_color="grey"); fig.add_hline(y=y_m, line_width=1, line_dash="dash", line_color="grey")
            
            # --- CORRECTED: Combined font arguments into a single dictionary per call ---
            base_anno = {'showarrow': False}
            fig.add_annotation(x=x_m*1.5, y=y_m*1.8, text="<b>Crisis Zones</b>", font={'color':"red", 'size':12}, **base_anno)
            fig.add_annotation(x=x_m/2, y=y_m*1.8, text="<b>Latent Threats</b>", font={'color':"navy", 'size':12}, **base_anno)
            fig.add_annotation(x=x_m*1.5, y=y_m/2, text="<b>Acute Hotspots</b>", font={'color':"darkorange", 'size':12}, **base_anno)
            
            fig.update_layout(xaxis_title="Dynamic Risk", yaxis_title="Structural Vulnerability", coloraxis_colorbar_title_text='Integrated<br>Risk')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in vulnerability quadrant: {e}", exc_info=True)
            st.warning("Could not display Vulnerability Quadrant plot.")

    def _plot_risk_contribution_sunburst(self, kpi_df: pd.DataFrame):
        st.markdown("**Analysis:** Break down a zone's `Integrated_Risk_Score` into its model components.")
        try:
            zones = kpi_df.nlargest(5, 'Integrated_Risk_Score')['Zone'].tolist()
            if not zones: st.info("No high-risk zones to analyze."); return
            zone = st.selectbox("Select a High-Risk Zone to Analyze:", options=zones)
            if not zone: return
            z_data, weights = kpi_df.loc[kpi_df['Zone']==zone].iloc[0], self.config.get('model_params',{}).get('advanced_model_weights',{})
            data = {'ids':['IR','BE','AM','STGP','HMM','GNN','GT'], 'labels':[f"Total: {z_data.get('Integrated_Risk_Score',0):.2f}",'Base','Adv.','STGP','HMM','GNN','Game Theory'], 'parents':['','IR','IR','AM','AM','AM','AM'],
                    'values':[z_data.get('Integrated_Risk_Score',0), weights.get('base_ensemble',0)*z_data.get('Ensemble Risk Score',0), (weights.get('stgp',0)*z_data.get('STGP_Risk',0)+weights.get('hmm',0)*z_data.get('HMM_State_Risk',0)+weights.get('gnn',0)*z_data.get('GNN_Structural_Risk',0)+weights.get('game_theory',0)*z_data.get('Game_Theory_Tension',0)), weights.get('stgp',0)*z_data.get('STGP_Risk',0), weights.get('hmm',0)*z_data.get('HMM_State_Risk',0), weights.get('gnn',0)*z_data.get('GNN_Structural_Risk',0), weights.get('game_theory',0)*z_data.get('Game_Theory_Tension',0)]}
            fig = go.Figure(go.Sunburst(ids=data['ids'], labels=data['labels'], parents=data['parents'], values=data['values'], branchvalues="total", hovertemplate='<b>%{label}</b><br>Contribution: %{value:.3f}<extra></extra>'))
            fig.update_layout(margin=dict(t=20,l=0,r=0,b=0), title_text=f"Risk Breakdown for Zone: {zone}", title_x=0.5, height=450)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in sunburst plot: {e}", exc_info=True)
            st.warning("Could not display Risk Contribution plot.")
            
    def _plot_forecast_with_uncertainty(self):
        st.markdown("**Analysis:** Projects risk over 72 hours. Shaded area is the 95% confidence interval.")
        try:
            fc_df, kpi_df = st.session_state.forecast_df, st.session_state.kpi_df
            if fc_df.empty or kpi_df.empty: st.info("No forecast data to display."); return
            zones = sorted(fc_df['Zone'].unique().tolist())
            defaults = kpi_df.nlargest(3, 'Integrated_Risk_Score')['Zone'].tolist()
            selected = st.multiselect("Select zones for forecast:", options=zones, default=defaults)
            if not selected: st.info("Please select at least one zone."); return
            fig = go.Figure()
            colors = px.colors.qualitative.Plotly
            for i, zone in enumerate(selected):
                zone_df = fc_df[fc_df['Zone'] == zone]
                if zone_df.empty: continue
                color, rgb = colors[i % len(colors)], px.colors.hex_to_rgb(colors[i % len(colors)])
                fig.add_trace(go.Scatter(x=np.concatenate([zone_df['Horizon (Hours)'], zone_df['Horizon (Hours)'][::-1]]), y=np.concatenate([zone_df['Upper_Bound'], zone_df['Lower_Bound'][::-1]]), fill='toself', fillcolor=f'rgba({",".join(map(str,rgb))}, 0.2)', line={'color':'rgba(255,255,255,0)'}, hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=zone_df['Horizon (Hours)'], y=zone_df['Combined Risk'], name=zone, line=dict(color=color, width=2), mode='lines+markers'))
            fig.update_layout(title='72-Hour Risk Forecast with 95% Confidence Interval', xaxis_title='Horizon (Hours)', yaxis_title='Projected Integrated Risk Score', legend_title_text='Zone', hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in forecast plot: {e}", exc_info=True)
            st.warning("Could not display forecast plot.")

    # --- metodologia, etc. ---
    def _render_methodology_tab(self):
        st.header("System Architecture & Methodology")
        st.markdown("This section provides a deep dive into the analytical engine powering the Phoenix v4.0 platform. It is designed for data scientists, analysts, and command staff who wish to understand the 'why' behind the system's predictions and prescriptions.")
        self._render_architecture_philosophy()
        self._render_prediction_engine()
        self._render_prescription_engine()
        self._render_incident_specific_weighting()
        self._render_kpi_glossary()

    def _render_architecture_philosophy(self):
        with st.expander("I. Architectural Philosophy: From Prediction to Prescription", expanded=True):
            st.markdown("""
            The fundamental goal of RedShield AI: Phoenix v4.0 is to engineer a paradigm shift in emergency response—from a traditional **reactive model** (dispatching units after an incident occurs) to a **proactive, prescriptive posture** (anticipating where incidents are likely to emerge and prescriptively positioning resources to minimize response times and maximize impact).
            
            To achieve this, the system is built on a philosophy of **Hierarchical Ensemble Modeling**. Instead of relying on a single "black box" algorithm, Phoenix v4.0 integrates a diverse portfolio of analytical techniques in a layered, "glass box" architecture. This creates a highly robust and resilient system where the weaknesses of any one model are offset by the strengths of others.
            
            The architecture is composed of four primary layers:
            1. **LAYER 1: Foundational Models.** Consists of well-established statistical models (Hawkes Processes, Bayesian Networks, Graph Laplacians) that create a robust baseline understanding of risk. This produces the `Ensemble_Risk_Score`.
            2. **LAYER 2: Advanced AI & Complexity Proxies.** Introduces computationally inexpensive but powerful proxies for cutting-edge models (ST-GPs, HMMs, GNNs) to capture deeper, more nuanced patterns that complement the foundational layer.
            3. **LAYER 3: Integrated Synthesis (Prediction).** The outputs of the first two layers are combined in a final, weighted synthesis to produce the ultimate `Integrated_Risk_Score`. This score represents the system's best **prediction** of risk.
            4. **LAYER 4: Prescriptive Optimization (Prescription).** The `Integrated_Risk_Score` and `Expected Incident Volume` are fed into an Operations Research (OR) engine. This layer moves beyond prediction to **prescription**, determining the *optimal* real-world action to take (e.g., ambulance allocation) to best mitigate the predicted risk.
            """)

    def _render_prediction_engine(self):
        with st.expander("II. The Prediction Engine: A Multi-Model Deep Dive", expanded=False):
            st.info("#### Core Principle: Different questions require different tools. No single model can capture all facets of urban risk.", icon="💡")
            st.markdown("---")
            st.markdown("#### **A. Stochastic & Statistical Models (The 'When')**")
            st.markdown("""
            * **Non-Homogeneous Poisson Process (NHPP):** This model forms the temporal backbone of our predictions. It understands that incident rates are not constant.
                - **Question it Answers:** *"What is the baseline probability of an incident at 3 AM on a Tuesday versus 6 PM on a Friday?"*
                - **Relevance:** Captures the predictable, cyclical nature of urban life, ensuring our baseline risk is sensitive to the time of day and day of the week.
                - **Mathematical Formulation:** The intensity function `λ(t)` is modeled as a function of time, often using a log-linear model with Fourier terms to capture cyclicity:
                    $$
                    \\lambda(t) = \\exp\\left( \\beta_0 + \\sum_{k=1}^{K} \\left[ \\beta_k \\cos\\left(\\frac{2\\pi kt}{T}\\right) + \\gamma_k \\sin\\left(\\frac{2\\pi kt}{T}\\right) \\right] \\right)
                    $$
                    where `β` and `γ` are coefficients learned from historical data and `T` is the period of the cycle (e.g., 24 hours or 168 hours).
            * **Hawkes Process (Self-Exciting Point Process):** This is the cornerstone of our violence and cascading accident models. It operates on the principle that certain events can trigger "aftershocks."
                - **Question it Answers:** *"Given a shooting just occurred, what is the immediate, elevated risk of another shooting in the same area?"* or *"After a major highway collision, what is the increased likelihood of secondary accidents due to traffic build-up?"*
                - **Relevance:** Critical for modeling retaliatory gang violence and chain-reaction traffic incidents. It directly powers the `Trauma Clustering Score`.
                - **Mathematical Formulation:** The conditional intensity `λ(t)` of an event at time `t` is defined as:
                    $$
                    \\lambda(t) = \\mu(t) + \\sum_{t_i < t} \\alpha \\cdot g(t - t_i)
                    $$
                    where `μ(t)` is the background rate from the NHPP, the sum is over past event times `tᵢ`, `α` is the branching ratio (strength of aftershock), and `g(t - tᵢ)` is the triggering kernel modeling the decaying influence of past events.
            * **Bayesian Networks:** These models represent our understanding of causal relationships. They combine static base rates with real-time environmental factors.
                - **Question it Answers:** *"How does a public holiday, combined with rainy weather and a major concert, collectively influence the probability of an incident?"*
                - **Relevance:** Allows the system to reason with expert knowledge and adapt to contextual factors like `Weather`, `Is Holiday`, and `Major Event`. It is a core driver of the baseline `Incident Probability`.
                - **Mathematical Formulation:** Based on the chain rule of probability, where the joint probability is the product of conditional probabilities: $P(X_1, ..., X_n) = \\prod_{i=1}^{n} P(X_i | \\text{Parents}(X_i))$. Our network models `P(IncidentRate | Weather, Holiday, ...)` to find the most likely baseline rate.
            """)
            st.markdown("---")
            st.markdown("#### **B. Spatiotemporal & Graph Models (The 'Where' and 'How It Spreads')**")
            st.markdown("""
            * **Spatiotemporal Gaussian Processes (ST-GPs):** Our `STGP_Risk` KPI is a proxy for this advanced technique. It models risk as a continuous fluid over the map.
                - **Question it Answers:** *"An incident occurred 500 meters from this zone's border. How much 'risk pressure' does that exert on this zone?"*
                - **Relevance:** Interpolates risk intelligently across the map, ensuring that proximity to danger is always accounted for, even across arbitrary zone boundaries.
                - **Mathematical Formulation:** The risk `f` at a spatiotemporal point `(s, t)` is modeled as a draw from a Gaussian Process:
                    $$
                    f(s, t) \\sim \\mathcal{GP}(m(s, t), k((s, t), (s', t')))
                    $$
                    where `m(s, t)` is the mean function and `k` is a spatiotemporal kernel, often a product of a spatial kernel (like RBF) and a temporal kernel:
                    $$
                    k((s, t), (s', t')) = \\sigma^2 \\exp\\left(-\\frac{\\|s-s'\|^2}{2l_s^2}\\right) \\exp\\left(-\\frac{|t-t'|^2}{2l_t^2}\\right)
                    $$
                    The lengthscales `l_s` and `l_t` control the "range" of influence in space and time.
            * **Graph Neural Networks (GNNs):** The city's road network and zone adjacencies are treated as a complex graph. A GNN learns a deep, structural understanding of each zone's role within this network.
                - **Question it Answers:** *"Is this zone inherently vulnerable simply due to its position as a major crossroads, regardless of recent events?"*
                - **Relevance:** Identifies long-term, structural vulnerabilities that may not be apparent from recent incident data alone. It powers the `GNN_Structural_Risk`, representing a zone's intrinsic risk.
                - **Mathematical Formulation (GCN Layer):** A GNN works by passing "messages" between connected nodes. A common layer operation is:
                    $$
                    H^{(l+1)} = \\sigma(\\hat{D}^{-\\frac{1}{2}} \\hat{A} \\hat{D}^{-\\frac{1}{2}} H^{(l)} W^{(l)})
                    $$
                    This means the new features for a node (`H^(l+1)`) are a transformed aggregate of its neighbors' previous features (`H^(l)`), where `Â` is the adjacency matrix with self-loops, `D̂` is its degree matrix, `W` is a learnable weight matrix, and `σ` is an activation function.
            * **Graph Laplacian Diffusion:** This technique models how effects (like traffic, panic, or police cordons) "spill over" from one zone to its neighbors through the road network.
                - **Question it Answers:** *"A major fire has shut down three blocks in Zone A. How does this increase the traffic-related accident risk in the adjacent Zone B?"*
                - **Relevance:** Essential for modeling the secondary effects of major incidents. It directly calculates the `Spatial Spillover Risk`.
                - **Mathematical Formulation:** The process is modeled by the heat diffusion equation on a graph. A single discrete step of this process is:
                    $$
                    r(t+1) = (I - \\epsilon L) r(t)
                    $$
                    where `r(t)` is the vector of zone risks, `I` is the identity matrix, `ε` is a small step size, and `L` is the normalized graph Laplacian matrix.
            """)
            st.markdown("---")
            st.markdown("#### **C. Complexity & Information Theory (The 'System State')**")
            st.markdown("""
            * **Lyapunov Exponents (Chaos Sensitivity Score):** A concept from Chaos Theory that measures a system's sensitivity to small changes. A high score means the system is in a fragile, unpredictable state.
                - **Question it Answers:** *"Is the city operating normally, or is it in a 'brittle' state where one small incident could cascade into a major crisis?"*
                - **Relevance:** This is a critical "instability alarm" for command staff. It doesn't predict a specific incident, but warns that the entire system is volatile.
                - **Mathematical Formulation (Conceptual):** It measures the exponential rate of divergence of nearby trajectories. If `δ(t)` is the separation between two trajectories over time, the largest Lyapunov exponent `λ` is estimated by:
                    $$
                    \\lambda \\approx \\frac{1}{t} \\ln \\frac{\\| \\delta(t) \\|}{\\| \\delta(0) \\|}
                    $$
                    A positive `λ` is an indicator of chaos.
            * **Kullback-Leibler (KL) Divergence (Anomaly Score):** An information theory metric that measures how much the current pattern of incidents deviates from the historical norm.
                - **Question it Answers:** *"Are we seeing the right number of incidents, but in all the wrong places today? Or are we seeing a bizarre new type of incident we've never seen before?"*
                - **Relevance:** Detects "pattern anomalies" that simple volume-based metrics would miss. A high score is a clear signal that "today is not a normal day."
                - **Mathematical Formulation:**
                    $$
                    D_{KL}(P || Q) = \\sum_{z \\in \\text{Zones}} P(z) \\log{\\frac{P(z)}{Q(z)}}
                    $$
            """)

    def _render_prescription_engine(self):
        with st.expander("III. The Prescription Engine: Optimal Resource Allocation", expanded=False):
            st.info("#### Core Principle: Moving from 'what will happen' to 'what is the best thing to do'.", icon="🎯")
            st.markdown("""
            The prescriptive engine uses the risk scores from the prediction layer as inputs into sophisticated Operations Research models. This ensures that resource allocation is not just intuitive, but mathematically optimal based on our objectives.
            
            * **Mixed-Integer Linear Programming (MILP):** This is the workhorse for `Linear Optimal` allocation. It finds the provably best way to assign a whole number of ambulances to zones.
                - **Objective:** Maximize the total amount of risk "covered" across the entire city.
                - **Relevance:** Excellent for finding the most efficient solution under a single, clear objective. It is fast and guarantees a mathematically optimal result for a linear problem.
                - **Mathematical Formulation (Simplified):**
                    $$
                    \\begin{aligned}
                    & \\text{maximize} && \\sum_{i \\in \\text{Zones}} R_i \\cdot c_i \\\\
                    & \\text{subject to} && \\sum_{i \\in \\text{Zones}} c_i \\leq N, \\quad c_i \\in \\mathbb{Z}^+
                    \\end{aligned}
                    $$
                    where `Rᵢ` is the risk score for zone `i`, `cᵢ` is the integer number of ambulances assigned, and `N` is the total available.
            * **Non-Linear Programming (NLP):** This is our most advanced model for `Non-Linear Optimal` allocation. It captures complex, real-world dynamics that linear models miss.
                - **Objective:** Minimize a "system dissatisfaction" function, which includes two key non-linear effects:
                    1. **Diminishing Returns (Logarithmic Utility):** The first ambulance sent to a zone provides a huge benefit; the fifth provides much less. The model understands this and avoids over-saturating a single high-risk zone if another zone has zero coverage.
                    2. **Congestion Penalties (Quadratic Penalty):** As the number of expected incidents in a zone vastly outpaces the allocated units, the "harm" (e.g., response time delay) grows exponentially, not linearly.
                - **Relevance:** This provides the most realistic and robust recommendations. It makes intelligent trade-offs that a human or a simpler model might miss, leading to a more resilient overall system posture.
                - **Mathematical Formulation (Simplified):**
                    $$
                    \\begin{aligned}
                    & \\text{minimize} && \\sum_{i \\in \\text{Zones}} \\left( w_1(R_i - R_i \\log(1+c_i)) + w_2 \\left( \\frac{E_i}{1+c_i} \\right)^2 \\right) \\\\
                    & \\text{subject to} && \\sum_{i \\in \\text{Zones}} c_i = N, \\quad c_i \\geq 0
                    \\end{aligned}
                    $$
                    where `Eᵢ` is expected incidents, the `log` term models **diminishing returns**, and the quadratic term models **congestion penalties**.
            * **Queueing Theory:** This mathematical theory is conceptually used to model system strain, particularly at hospitals.
                - **Relevance:** By understanding arrival rates (from our predictions) and service rates, we can better estimate wait times and the impact of hospital diversions, which feeds into the `Resource Adequacy Index`.
                - **Objective:** To mathematically model and predict wait times, congestion, and the probability of system saturation (e.g., at a hospital ER).
                - **Overall Relevance in the System:** Queueing theory provides a robust, theoretical foundation for the `Resource Adequacy Index` and informs routing decisions. Instead of a simple penalty for a busy hospital, it allows the system to calculate the *actual expected delay*, leading to smarter assignments.
                - **The Question it Answers:** "If we send another ambulance to Hospital X, what is the probability it will have to wait more than 15 minutes to offload the patient, given their current patient load and our predicted arrival rate of new incidents?"
                - **Mathematical Formulation (M/M/c Model):** For a system with `c` servers (e.g., ER beds), a Poisson arrival rate `λ` (from our incident predictions), and an exponential service rate `μ`, the probability of an arriving patient having to wait is given by the Erlang-C formula:
                    $$
                    P_{\text{wait}} = C(c, \lambda/\mu) = \frac{(\lambda/\mu)^c / c!}{ ((\lambda/\mu)^c / c!) + (1 - \lambda/(c\mu)) \sum_{k=0}^{c-1} (\lambda/\mu)^k / k!}
                    $$
                - **Mathematical Relevance:** This is a cornerstone of Operations Research, providing a powerful analytical framework to understand and optimize stochastic systems defined by random arrivals and service times, which perfectly describes an emergency response network.
            """)

    def _render_incident_specific_weighting(self):
        with st.expander("IV. Incident-Specific Model Weighting", expanded=False):
            st.markdown("""
            The system is not one-size-fits-all. The final `Integrated_Risk_Score` is a weighted sum of many model outputs, and these weights are dynamically influenced by the nature of the risk being assessed.
            
            #### **Trauma - Violence**
            * **Primary Predictive Models:** **Hawkes Processes** are paramount, as they explicitly model the retaliatory, self-exciting nature of violence. **GNN Structural Risk** is also critical for identifying long-term territorial hotspots.
                - **Objective:** To capture the self-exciting, retaliatory nature of violent crime, where one incident significantly increases the probability of another nearby.
                - **Overall Relevance:** This is the primary driver for short-term, high-intensity violence prediction, making it critical for proactive patrol and resource staging.
                - **Mathematical Formulation:** The conditional intensity `λ(t)` is:
                    $$
                    \\lambda(t) = \\mu(t) + \\sum_{t_i < t} \\alpha \\cdot g(t - t_i)
                    $$
                    where `μ(t)` is the background rate, `α` is the strength of aftershocks, and `g()` is the decay function.
            * **Primary Prescriptive Model:** **NLP** is often preferred to not only cover risk but also to avoid over-saturating one area, which can be crucial in fluid tactical situations.
                - **Objective:** To allocate resources in a way that is robust to the fluid and rapidly changing tactical situation common in violence response.
                - **Overall Relevance:** The NLP model's ability to handle diminishing returns prevents over-saturating a single hotspot, ensuring flexible coverage across multiple potential flashpoints.
            * **Primary Predictive Model: Graph Laplacians**
                - **Objective:** To model how traffic congestion and blockages from one incident "spill over" and increase accident risk in adjacent areas.
                - **Overall Relevance:** This is critical for predicting secondary incidents caused by the disruption of a primary one.
                - **Mathematical Formulation:** A discrete step of the heat diffusion equation on the city graph:
                    $$
                    r(t+1) = (I - \\epsilon L) r(t)
                    $$
                    where `L` is the graph Laplacian, modeling the flow of "risk pressure".
            #### **Trauma - Accidents**
            * **Primary Predictive Models:** **Bayesian Networks** (incorporating weather and traffic) and **Graph Laplacians** (modeling spillover from congestion) are the key drivers.
            * **Primary Prescriptive Model:** **NLP** is highly effective here as its built-in congestion penalty directly models the real-world consequence of traffic jams, leading to smarter staging decisions.
            #### **Medical Emergencies**
            * **Primary Predictive Models:** **Bayesian Networks** are crucial for incorporating environmental factors like heatwaves and air quality. Spatiotemporal models analyzing population density and demographics (e.g., age) are also key.
                - **Objective:** To probabilistically fuse multiple, distinct causal factors for accidents, such as weather, traffic, and events.
                - **Overall Relevance:** It provides a clear, interpretable model for how environmental conditions elevate accident risk system-wide.
                - **Mathematical Formulation:** It uses the chain rule of probability: $P(X_1, ..., X_n) = \\prod_{i=1}^{n} P(X_i | \\text{Parents}(X_i))$, allowing for inference on `P(Accident | Weather, TrafficLevel)`.
            * **Primary Prescriptive Model:** The choice of model is heavily influenced by **Hospital Divert Status**. When hospitals are under strain, the prescriptive models must weigh not just the risk of an incident, but the added travel time and risk of delay upon arrival, a factor that NLP can incorporate more naturally.
            """)

    def _render_kpi_glossary(self):
        with st.expander("V. Key Performance Indicator (KPI) Glossary", expanded=False):
            kpi_defs = {
                "Integrated Risk Score": """
                **The final, primary risk metric** used for all decisions, blending foundational and advanced AI models.
                - **Objective:** To create the single, unified, actionable risk score by blending the baseline with advanced AI signals.
                - **Overall Relevance:** This is the primary output of the predictive engine and drives all downstream tasks.
                - **Mathematical Formulation:** A weighted linear combination:
                    $$
                    R_{int} = \\alpha R_{ens} + \\beta R_{stgp} + \\gamma R_{gnn} + \\delta R_{tension} + ...
                    $$
                - **Mathematical Relevance:** A hierarchical ensemble that combines a stable base with specialized, sensitive signals for a balanced and powerful final prediction.
                """,
                "Ensemble Risk Score": """
                Blended risk score from the foundational (Layer 1) models.
                - **Objective:** To combine signals from foundational models into a stable baseline risk assessment.
                - **Overall Relevance:** It provides a robust, interpretable "first-pass" risk score.
                - **Mathematical Formulation:** A weighted average of normalized model outputs:
                    $$
                    R_{ens} = \\sum_{k=1}^{K} w_k \\cdot \\text{normalize}(M_k)
                    $$
                - **Mathematical Relevance:** A classic model ensembling technique to reduce variance and improve robustness.
                """,
                "GNN Structural Risk": """
                A zone's intrinsic vulnerability due to its position in the road network.
                - **Objective:** To identify long-term, endemic hotspots of violence based on their inherent network properties, independent of recent events.
                - **Overall Relevance:** It provides a stable, foundational layer of risk, preventing the system from ignoring historically dangerous areas that are momentarily quiet.
                - **Mathematical Formulation (PageRank):** A zone `zᵢ`'s importance is calculated iteratively based on the importance of its neighbors:
                    $$
                    PR(z_i) = \\frac{1-d}{N} + d \\sum_{z_j \\in N(z_i)} \\frac{PR(z_j)}{|N(z_j)|}
                    $$
                """,
                "STGP Risk": """
                Risk from proximity to recent, severe incidents (spatiotemporal correlation).
                - **Objective:** To quantify risk from proximity to recent, severe incidents, assuming risk decays smoothly over space and time.
                - **Overall Relevance:** It provides a more fluid, less-blocky view of risk that respects geographical closeness over arbitrary zone boundaries.
                - **Mathematical Formulation:** The posterior mean of a Spatiotemporal Gaussian Process, `f(s,t) ~ GP(m, k)`, conditioned on recent incident locations.
                - **Mathematical Relevance:** A non-parametric Bayesian method that provides a statistically powerful way to perform spatiotemporal interpolation.
                """,
                "Game Theory Tension": """
                A measure of a zone's contribution to system-wide resource competition.
                - **Objective:** To measure how much a single zone contributes to the overall "competition" for limited EMS resources.
                - **Overall Relevance:** It identifies which zones are the primary drivers of system-wide strain.
                - **Mathematical Formulation:** A heuristic capturing the interaction between risk and demand:
                    $$
                    T_i = R_i \\times E_i
                    $$
                - **Mathematical Relevance:** It acts as a proxy for a zone's "bidding power" in a conceptual resource competition, prioritizing areas where the consequences of under-resourcing are highest.
                """,
                "Chaos Sensitivity Score": """
                Measures system volatility and fragility. High score = 'The system is unstable.'
                - **Objective:** To provide an early warning of system instability.
                - **Overall Relevance:** It's the system's "check engine light." High risk + high chaos is a volatile, unpredictable crisis in the making.
                - **Mathematical Formulation (Heuristic):** A practical proxy for the Largest Lyapunov Exponent, calculated from the log of the mean absolute difference in the time series of incident counts: `log(mean(|count(t) - count(t-1)|))`.
                - **Mathematical Relevance:** Applies a core concept from dynamical systems theory to provide a meta-level insight into the system's state (its rate of change and volatility).
                """,
                "Anomaly Score": """
                Measures the 'strangeness' of the current incident pattern compared to history.
                - **Objective:** To quantify how "strange" the current spatial pattern of incidents is compared to the historical norm.
                - **Overall Relevance:** It detects novel threats or unusual shifts in behavior that would be missed by looking at volume alone.
                - **Mathematical Formulation (KL Divergence):**
                    $$
                    D_{KL}(P || Q) = \\sum_{z} P(z) \\log{\\frac{P(z)}{Q(z)}}
                    $$
                - **Mathematical Relevance:** A fundamental metric from information theory measuring the "surprise" when moving from a prior distribution (history) to a posterior one (today).
                """,
                "Resource Adequacy Index": """
                System-wide ratio of available units to expected demand, penalized by hospital strain.
                - **Objective:** To provide a single, holistic measure of the EMS system's ability to meet the currently predicted demand.
                - **Overall Relevance:** A top-line metric for commanders. A score of 60% indicates the system is severely overstretched.
                - **Mathematical Formulation:** A ratio of supply to demand, penalized by system strain factors:
                    $$
                    RAI = \\frac{\\text{AvailableUnits}}{\\sum E_i \\times (1 + k_{hosp} \\cdot \\text{HospDivert} + k_{traf} \\cdot \\text{Traffic})}
                    $$
                - **Mathematical Relevance:** A practical, operational metric that directly synthesizes predictive outputs (`Σ Eᵢ`) with real-time system status variables into an actionable health score.
                """
            }
            for kpi, definition in kpi_defs.items():
                c1, c2 = st.columns([1, 2])
                with c1: st.markdown(f"**{kpi}**")
                with c2: st.markdown(definition)
                st.markdown("---", unsafe_allow_html=True)

    # --- SIDEBAR AND ACTIONS ---

    def _render_sidebar(self):
        """Renders the sidebar for user controls and actions."""
        st.sidebar.title("Strategic Controls")
        st.sidebar.markdown("Adjust real-time factors to simulate different scenarios.")
        new_env = self._build_env_factors_from_sidebar()
        if new_env != st.session_state.env_factors:
            logger.info("EnvFactors updated, triggering rerun.")
            st.session_state.env_factors = new_env
            st.rerun()
        st.sidebar.divider()
        st.sidebar.header("Data & Reporting")
        self._sidebar_file_uploader()
        if st.sidebar.button("Generate & Download PDF Report", use_container_width=True):
            self._generate_report()

    def _build_env_factors_from_sidebar(self) -> EnvFactorsWithTolerance:
        """Builds an EnvFactors object from the current state of sidebar widgets."""
        env = st.session_state.env_factors
        with st.sidebar.expander("General Environmental Factors", expanded=True):
            is_holiday = st.checkbox("Is Holiday", value=env.is_holiday)
            weather = st.selectbox("Weather", ["Clear", "Rain", "Fog"], index=["Clear", "Rain", "Fog"].index(env.weather))
            aqi = st.slider("Air Quality Index (AQI)", 0.0, 500.0, env.air_quality_index, 5.0)
            heatwave = st.checkbox("Heatwave Alert", value=env.heatwave_alert)
        with st.sidebar.expander("Contextual & Event-Based Factors", expanded=True):
            day_type = st.selectbox("Day Type", ['Weekday', 'Friday', 'Weekend'], index=['Weekday', 'Friday', 'Weekend'].index(env.day_type))
            time_of_day = st.selectbox("Time of Day", ['Morning Rush', 'Midday', 'Evening Rush', 'Night'], index=['Morning Rush', 'Midday', 'Evening Rush', 'Night'].index(env.time_of_day))
            public_event = st.selectbox("Public Event Type", ['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'], index=['None', 'Sporting Event', 'Concert/Festival', 'Public Protest'].index(env.public_event_type))
            school_in_session = st.checkbox("School In Session", value=env.school_in_session)
        with st.sidebar.expander("System Strain & Response Factors", expanded=True):
            traffic = st.slider("General Traffic Level", CONSTANTS['TRAFFIC_MIN'], CONSTANTS['TRAFFIC_MAX'], env.traffic_level, 0.1)
            h_divert = st.slider("Hospital Divert Status (%)", 0, 100, int(env.hospital_divert_status * 100), 5)
            police_activity = st.selectbox("Police Activity Level", ['Low', 'Normal', 'High'], index=['Low', 'Normal', 'High'].index(env.police_activity))
        return EnvFactorsWithTolerance(
            is_holiday=is_holiday, weather=weather, traffic_level=traffic, major_event=(public_event != 'None'),
            population_density=env.population_density, air_quality_index=aqi, heatwave_alert=heatwave,
            day_type=day_type, time_of_day=time_of_day, public_event_type=public_event,
            hospital_divert_status=h_divert / 100.0, police_activity=police_activity, school_in_session=school_in_session
        )

    def _sidebar_file_uploader(self):
        """Handles logic for the historical data file uploader."""
        up_file = st.sidebar.file_uploader("Upload Historical Incidents (JSON)", type=["json"])
        if up_file:
            try:
                data = json.load(up_file)
                if not isinstance(data, list) or not all('location' in d and 'type' in d for d in data):
                    raise ValueError("Invalid JSON: Must be a list of incident objects.")
                st.session_state.historical_data = data
                st.sidebar.success(f"Loaded {len(data)} historical records.")
                st.rerun()
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Error loading uploaded data: {e}", exc_info=True)
                st.sidebar.error(f"Error parsing file: {e}")
            except Exception as e:
                logger.error(f"Unexpected error during file upload: {e}", exc_info=True)
                st.sidebar.error("An unexpected error occurred.")

    def _generate_report(self):
        """Generates and provides a download link for the PDF report."""
        with st.spinner("Generating Report..."):
            try:
                pdf_buffer = ReportGenerator.generate_pdf_report(
                    kpi_df=st.session_state.kpi_df, forecast_df=st.session_state.forecast_df,
                    allocations=st.session_state.allocations, env_factors=st.session_state.env_factors
                )
                if pdf_buffer.getbuffer().nbytes > 0:
                    st.sidebar.download_button(
                        label="Download PDF Report", data=pdf_buffer,
                        file_name=f"RedShield_Phoenix_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf", use_container_width=True)
                else: raise ValueError("Generated PDF buffer is empty.")
            except Exception as e:
                logger.error(f"Report generation failed: {e}", exc_info=True)
                st.sidebar.error(f"Report generation failed: {e}")

def main():
    """Main function to initialize and run the application."""
    try:
        config = load_config()
        data_manager = DataManager(config)
        engine = PredictiveAnalyticsEngine(data_manager, config)
        dashboard = Dashboard(data_manager, engine)
        dashboard.render()
    except Exception as e:
        logger.critical(f"A fatal error occurred during application startup: {e}", exc_info=True)
        st.error(f"A fatal application error occurred: {e}. Please check logs and configuration file.")

if __name__ == "__main__":
    main()
