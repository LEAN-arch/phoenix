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
    @staticmethod
    def _create_sparkline_plot(
        data: List[float], 
        normal_range: List[float], 
        current_value_text: str, 
        label: str, 
        color: str,
        high_is_bad: bool = True
    ) -> go.Figure:
        """
        [SME VISUALIZATION] Creates a sophisticated and elegant sparkline plot.
        This design is optimized for information density in a small space and includes a contextual y-axis.
        """
        fig = go.Figure()

        # 1. Normal Operating Range (Subtle background band)
        fig.add_shape(
            type="rect",
            xref="x", yref="y",
            x0=0, y0=normal_range[0], x1=len(data) - 1, y1=normal_range[1],
            fillcolor="#388E3C",  # Always green for "normal"
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        # 2. Main Trend Line
        fig.add_trace(go.Scatter(
            x=list(range(len(data))),
            y=data,
            mode='lines',
            line=dict(color=color, width=3),
            hoverinfo='none'
        ))

        # 3. Current Value Marker (The prominent final dot)
        fig.add_trace(go.Scatter(
            x=[len(data) - 1],
            y=[data[-1]],
            mode='markers',
            marker=dict(color=color, size=10, line=dict(width=2, color='white')),
            hoverinfo='none'
        ))

        # 4. Big Current Value Annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=f"<b>{current_value_text}</b>",
            showarrow=False,
            font=dict(size=28, color=color, family="Arial Black, sans-serif"),
            align="left",
            xanchor="left",
            yanchor="top"
        )
        
        # 5. Label Annotation
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.05,
            text=label,
            showarrow=False,
            font=dict(size=14, color="#666"),
            align="right",
            xanchor="right",
            yanchor="bottom"
        )
        
        # Determine y-axis range to give some padding
        plot_min = min(min(data), normal_range[0])
        plot_max = max(max(data), normal_range[1])
        padding = (plot_max - plot_min) * 0.15
        
        fig.update_layout(
            # --- Y-AXIS IS NOW VISIBLE ---
            yaxis=dict(
                range=[plot_min - padding, plot_max + padding],
                showticklabels=True,  # This makes the axis labels visible
                tickfont=dict(size=10, color="#999"),
                side='right',
                nticks=4, # Limit the number of ticks to avoid clutter
                showgrid=False # Keep the plot area clean
            ),
            xaxis=dict(visible=False),
            showlegend=False,
            plot_bgcolor='rgba(240, 240, 240, 0.95)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=5, r=40, t=5, b=5), # Increased right margin for axis labels
            height=120
        )
        return fig

    def _render_operational_command_tab(self):
        """Renders the main operational command view."""
        # --- This part is UNCHANGED ---
        self._render_system_status_bar()
        
        # --- SME Addition: Add sparkline plots in an expander for detailed trends ---
        with st.expander("Show System Trend Details"):
            kpi_df = st.session_state.kpi_df
            spark_data = st.session_state.sparkline_data
            
            if not kpi_df.empty and spark_data:
                try:
                    # --- Data Extraction ---
                    inc_val = len(st.session_state.current_incidents)
                    amb_val = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
                    risk_val = kpi_df['Integrated_Risk_Score'].max()
                    adeq_val = kpi_df['Resource Adequacy Index'].mean()

                    inc_data = spark_data.get('active_incidents', {'values': [inc_val]*5, 'range': [inc_val-1, inc_val+1]})
                    amb_data = spark_data.get('available_ambulances', {'values': [amb_val]*5, 'range': [amb_val-1, amb_val+1]})
                    risk_data = spark_data.get('max_risk', {'values': [risk_val]*5, 'range': [0, 1]})
                    adeq_data = spark_data.get('adequacy', {'values': [adeq_val]*5, 'range': [0, 1]})
                    
                    def get_status_color(val, normal_range, high_is_bad=True):
                        low, high = normal_range
                        if (high_is_bad and val > high) or (not high_is_bad and val < low): return "#D32F2F"
                        if (high_is_bad and val > low) or (not high_is_bad and val < high): return "#FBC02D"
                        return "#388E3C"

                    cols = st.columns(4)
                    with cols[0]:
                        st.plotly_chart(self._create_sparkline_plot(inc_data['values'], inc_data['range'], f"{inc_val}", "Active Incidents", get_status_color(inc_val, inc_data['range'])), use_container_width=True)
                    with cols[1]:
                        st.plotly_chart(self._create_sparkline_plot(amb_data['values'], amb_data['range'], f"{amb_val}", "Available Units", get_status_color(amb_val, amb_data['range'], False)), use_container_width=True)
                    with cols[2]:
                        st.plotly_chart(self._create_sparkline_plot(risk_data['values'], risk_data['range'], f"{risk_val:.3f}", "Max Zone Risk", get_status_color(risk_val, risk_data['range'])), use_container_width=True)
                    with cols[3]:
                        st.plotly_chart(self._create_sparkline_plot(adeq_data['values'], adeq_data['range'], f"{adeq_val:.1%}", "System Adequacy", get_status_color(adeq_val, adeq_data['range'], False)), use_container_width=True)
                
                except Exception as e:
                    logger.error(f"Error rendering sparkline plots: {e}", exc_info=True)
                    st.warning("Could not display trend details.")
            else:
                st.info("Trend data is not yet available.")
        
        # --- This part is UNCHANGED ---
        st.divider()
        col1, col2 = st.columns([3, 2])
        with col1:
            st.subheader("Live Operations Map")
            map_data = self._prepare_map_data(kpi_df=st.session_state.kpi_df, _zones_gdf=self.dm.zones_gdf)
            if map_data is not None:
                map_object = self._render_dynamic_map(map_gdf=map_data, incidents=st.session_state.current_incidents, _ambulances=self.dm.ambulances)
                if map_object: st_folium(map_object, use_container_width=True, height=600)
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

    @st.cache_data
    def _prepare_map_data(_self, kpi_df: pd.DataFrame, _zones_gdf: gpd.GeoDataFrame) -> Optional[gpd.GeoDataFrame]:
        """Prepares and caches the GeoDataFrame needed for map rendering."""
        if _zones_gdf.empty or kpi_df.empty:
            return None
        map_gdf = _zones_gdf.join(kpi_df.set_index('Zone'), on='name')
        map_gdf.reset_index(inplace=True)
        return map_gdf

    def _render_dynamic_map(self, map_gdf: gpd.GeoDataFrame, incidents: List[Dict], _ambulances: Dict) -> Optional[folium.Map]:
        """Renders the Folium map using pre-prepared, cached data. This function is NOT cached."""
        try:
            center = map_gdf.unary_union.centroid
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
            logger.error(f"Failed to render map from prepared data: {e}", exc_info=True)
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
        kpi_df = st.session_state.kpi_df
        if not kpi_df.empty:
            st.dataframe(kpi_df.set_index('Zone').style.format("{:.3f}").background_gradient(cmap='viridis'), use_container_width=True)
        else:
            st.info("KPI data not yet available.")
        st.divider()
        st.subheader("Advanced Analytical Visualizations")
        if not kpi_df.empty:
            # --- SME Addition: Expanded and reorganized tabs for new visualizations ---
            tab_titles = [
                "📍 Strategic Overview", 
                "🎯 Allocation Opportunity", 
                "⏱️ Risk Momentum", 
                "🧬 Critical Zone Anatomy", 
                "🧩 Zone Deep-Dive", 
                "🔭 72-Hour Forecast"
            ]
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tab_titles)
            with tab1:
                self._plot_vulnerability_quadrant(kpi_df)
            with tab2:
                # Call the new plot method
                self._plot_allocation_opportunity(kpi_df, st.session_state.allocations)
            with tab3:
                # Call the new plot method
                self._plot_risk_momentum(kpi_df)
            with tab4:
                # Call the new plot method
                self._plot_critical_zone_anatomy(kpi_df)
            with tab5:
                # This was the original "Risk Contribution Drill-Down"
                self._plot_risk_contribution_sunburst(kpi_df)
            with tab6:
                # This was the original "Risk Forecast & Uncertainty"
                self._plot_forecast_with_uncertainty()
        else:
            st.info("Advanced visualizations unavailable: Waiting for data...")

    def _plot_vulnerability_quadrant(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a Strategic Risk Matrix.
        This quadrant plot elegantly segments zones by their risk profile for high-level decision-making.
        """
        st.markdown("**Analysis:** This matrix segments zones by **structural vulnerability** vs. **dynamic risk**. The quadrants guide strategic focus: from immediate crisis response to long-term preventative action. Bubble size indicates expected incident volume, highlighting zones where high risk has the greatest potential impact on the population.")
        try:
            req = ['Ensemble Risk Score', 'GNN_Structural_Risk', 'Integrated_Risk_Score', 'Expected Incident Volume']
            if not all(c in kpi_df.columns for c in req):
                st.error("Data missing for Strategic Risk Matrix."); return

            # Use 75th percentile for a more meaningful "high risk" threshold
            x_threshold = kpi_df['Ensemble Risk Score'].quantile(0.75)
            y_threshold = kpi_df['GNN_Structural_Risk'].quantile(0.75)
            
            max_incidents = max(kpi_df['Expected Incident Volume'].max(), 1)
            sizeref = 2. * max_incidents / (40.**2)

            fig = go.Figure()

            # --- Background Quadrant Shading for immediate visual context ---
            x_max = kpi_df['Ensemble Risk Score'].max() * 1.1
            y_max = kpi_df['GNN_Structural_Risk'].max() * 1.1
            fig.add_shape(type="rect", xref="x", yref="y", x0=x_threshold, y0=y_threshold, x1=x_max, y1=y_max, fillcolor="rgba(229, 57, 53, 0.07)", line_width=0, layer="below")
            fig.add_shape(type="rect", xref="x", yref="y", x0=x_threshold, y0=0, x1=x_max, y1=y_threshold, fillcolor="rgba(255, 179, 0, 0.07)", line_width=0, layer="below")
            fig.add_shape(type="rect", xref="x", yref="y", x0=0, y0=y_threshold, x1=x_threshold, y1=y_max, fillcolor="rgba(25, 118, 210, 0.07)", line_width=0, layer="below")

            # --- Scatter plot with enhanced markers and a more engaging color scale ---
            fig.add_trace(go.Scatter(
                x=kpi_df['Ensemble Risk Score'],
                y=kpi_df['GNN_Structural_Risk'],
                mode='markers', # Text will be added selectively
                marker=dict(
                    size=kpi_df['Expected Incident Volume'],
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=6,
                    # --- Color Enhancement: Using a vibrant, perceptually uniform scale ---
                    color=kpi_df['Integrated_Risk_Score'],
                    colorscale='Plasma', 
                    showscale=True,
                    colorbar=dict(
                        title='Total<br>Risk',
                        x=1.15,
                        thickness=20,
                        tickfont=dict(size=10)
                    ),
                    # --- Aesthetic Enhancement: Add a line to make markers pop ---
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                customdata=kpi_df['Zone'],
                hovertemplate="<b>Zone: %{customdata}</b><br>Dynamic Risk: %{x:.3f}<br>Structural Risk: %{y:.3f}<extra></extra>"
            ))
            
            # --- Selective Labeling for Critical Zones ---
            crisis_zones = kpi_df[(kpi_df['Ensemble Risk Score'] >= x_threshold) & (kpi_df['GNN_Structural_Risk'] >= y_threshold)]
            if not crisis_zones.empty:
                fig.add_trace(go.Scatter(
                    x=crisis_zones['Ensemble Risk Score'],
                    y=crisis_zones['GNN_Structural_Risk'],
                    mode='text',
                    text=crisis_zones['Zone'],
                    textposition="top center",
                    textfont=dict(size=10, color="#333", family="Arial Black"),
                    showlegend=False,
                    hoverinfo='none'
                ))

            # --- Quadrant Lines and Annotations ---
            fig.add_vline(x=x_threshold, line_width=1.5, line_dash="longdash", line_color="rgba(0,0,0,0.2)")
            fig.add_hline(y=y_threshold, line_width=1.5, line_dash="longdash", line_color="rgba(0,0,0,0.2)")

            anno_defaults = dict(xref="paper", yref="paper", showarrow=False, font=dict(family="Arial, sans-serif", size=11, color="rgba(0,0,0,0.5)"))
            fig.add_annotation(x=0.98, y=0.98, text="<b>CRISIS ZONES</b><br>(High Dynamic, High Structural)", xanchor='right', yanchor='top', align='right', **anno_defaults)
            fig.add_annotation(x=0.98, y=0.02, text="<b>ACUTE HOTSPOTS</b><br>(High Dynamic, Low Structural)", xanchor='right', yanchor='bottom', align='right', **anno_defaults)
            fig.add_annotation(x=0.02, y=0.98, text="<b>LATENT THREATS</b><br>(Low Dynamic, High Structural)", xanchor='left', yanchor='top', align='left', **anno_defaults)
            fig.add_annotation(x=0.02, y=0.02, text="STABLE ZONES", xanchor='left', yanchor='bottom', align='left', **anno_defaults)

            # --- Final Layout Polish for a professional, "mission control" feel ---
            fig.update_layout(
                title_text="<b>Strategic Risk Matrix</b>",
                title_x=0.5,
                title_font=dict(size=20, family="Arial, sans-serif"),
                xaxis_title="Dynamic Risk (Events & Recency) →",
                yaxis_title="Structural Vulnerability (Intrinsic) →",
                height=550,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=False,
                xaxis=dict(gridcolor='#e5e5e5', zeroline=False, range=[0, kpi_df['Ensemble Risk Score'].max() * 1.1]),
                yaxis=dict(gridcolor='#e5e5e5', zeroline=False, range=[0, kpi_df['GNN_Structural_Risk'].max() * 1.1]),
                margin=dict(l=80, r=40, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in vulnerability quadrant plot: {e}", exc_info=True)
            st.warning("Could not display Strategic Risk Matrix.")
            
    # --- SME Addition: New High-Value Plot 1 ---
    def _plot_risk_fingerprints(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a radar chart comparing the risk 'fingerprints' of the top 5 zones.
        This provides a sophisticated, at-a-glance view of *why* different zones are risky.
        """
        st.markdown("**Analysis:** This radar chart visualizes the unique risk 'fingerprint' for the top 5 highest-risk zones. It helps answer: *'Is this zone's risk due to its inherent structure, recent activity, or system tension?'* A large, skewed shape indicates a specialized threat, while a large, balanced shape indicates a multi-faceted crisis.")
        try:
            risk_components = ['Ensemble Risk Score', 'GNN_Structural_Risk', 'STGP_Risk', 'HMM_State_Risk', 'Game_Theory_Tension']
            if not all(col in kpi_df.columns for col in risk_components):
                st.error("Data missing for Risk Fingerprints plot."); return

            df_top5 = kpi_df.nlargest(5, 'Integrated_Risk_Score')
            
            # Normalize each component 0-1 for fair comparison on the radar chart
            normalized_df = df_top5[risk_components].apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
            normalized_df['Zone'] = df_top5['Zone']

            fig = go.Figure()
            for i, row in normalized_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=row.values[:-1].tolist() + [row.values[0]],  # Close the loop
                    theta=risk_components + [risk_components[0]],
                    fill='toself',
                    name=row['Zone'],
                    hovertemplate=f"<b>Zone:</b> {row['Zone']}<br><b>Risk Type:</b> %{{theta}}<br><b>Normalized Score:</b> %{{r:.3f}}<extra></extra>"
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1])
                ),
                showlegend=True,
                title="Comparative Risk Fingerprints for Top 5 Zones",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in risk fingerprints plot: {e}", exc_info=True)
            st.warning("Could not display Risk Fingerprints plot.")

    # --- SME Addition: New High-Value Plot 2 ---
    def _plot_systemic_risk_composition(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a treemap to show the composition of risk types across the entire system.
        Size represents total risk magnitude, color represents the dominant risk type.
        """
        st.markdown("**Analysis:** This treemap provides a city-wide overview of risk. The **size** of each rectangle represents a zone's total `Integrated_Risk_Score`. The **color** indicates the dominant *type* of incident risk (Violence, Accident, or Medical). This allows commanders to instantly identify not just *how much* risk exists, but *what kind* of risk is driving it across the system.")
        try:
            risk_types = ['Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score', 'Integrated_Risk_Score']
            if not all(col in kpi_df.columns for col in risk_types):
                st.error("Data missing for Systemic Risk Composition plot."); return

            df_comp = kpi_df.copy()
            # Identify the dominant risk driver
            df_comp['Dominant_Risk'] = df_comp[['Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score']].idxmax(axis=1).str.replace(' Score', '').str.replace(' Clustering', '')
            
            fig = px.treemap(
                df_comp,
                path=[px.Constant("All Zones"), 'Dominant_Risk', 'Zone'],
                values='Integrated_Risk_Score',
                color='Dominant_Risk',
                color_discrete_map={
                    'Violence': '#D32F2F',
                    'Accident': '#FBC02D',
                    'Medical Surge': '#1E90FF',
                    '(?)': 'grey'
                },
                hover_data={'Integrated_Risk_Score': ':.3f'},
                title="Systemic Risk Composition"
            )
            fig.update_traces(textinfo="label+percent root", hovertemplate='<b>Zone:</b> %{label}<br><b>Total Integrated Risk:</b> %{value:.3f}<br><b>Dominant Type:</b> %{color}<extra></extra>')
            fig.update_layout(height=500, margin=dict(t=50, l=10, r=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in systemic risk composition plot: {e}", exc_info=True)
            st.warning("Could not display Systemic Risk Composition plot.")
   
    # --- SME Addition: New High-Value Plot 1 (Commercial Grade) ---
    def _plot_allocation_opportunity(self, kpi_df: pd.DataFrame, allocations: Dict[str, int]):
        """
        [SME VISUALIZATION] Plots a Strategic Deployment Matrix.
        This provides an immediate, actionable overview of resource allocation efficiency.
        """
        st.markdown("**Analysis:** This matrix identifies the most critical resource gaps and potential surpluses. It plots each zone based on its total risk and its current resource deficit. The goal is to move zones from the top-right quadrant (re-allocate **to** them) by using surplus units from the top-left quadrant (re-allocate **from** them).")
        try:
            df = kpi_df[['Zone', 'Integrated_Risk_Score', 'Expected Incident Volume']].copy()
            df['allocated_units'] = df['Zone'].map(allocations).fillna(0)
            risk_cov = self.config.get('model_params', {}).get('risk_coverage_per_unit', CONSTANTS['RISK_COVERAGE_PER_UNIT'])
            df['risk_covered'] = df['allocated_units'] * risk_cov
            df['resource_deficit'] = df['Integrated_Risk_Score'] - df['risk_covered']

            # Define a dynamic reference for bubble sizing to prevent errors
            max_incidents = max(df['Expected Incident Volume'].max(), 1)
            sizeref = 2. * max_incidents / (40.**2)

            mean_risk = df['Integrated_Risk_Score'].mean()
            
            fig = go.Figure()

            # Background Quadrant Colors for Intuitive Interpretation
            fig.add_shape(type="rect", xref="paper", yref="paper", x0=0.5, y0=0.5, x1=1, y1=1, line=dict(width=0), fillcolor="rgba(211, 47, 47, 0.1)", layer="below")
            fig.add_shape(type="rect", xref="paper", yref="paper", x0=0, y0=0.5, x1=0.5, y1=1, line=dict(width=0), fillcolor="rgba(30, 136, 229, 0.1)", layer="below")
            
            # Scatter Plot
            fig.add_trace(go.Scatter(
                x=df['Integrated_Risk_Score'], y=df['resource_deficit'],
                mode='markers', # Text will be added as a separate trace for better control
                marker=dict(
                    size=df['Expected Incident Volume'],
                    sizemode='area',
                    sizeref=sizeref,
                    sizemin=4,
                    color=df['resource_deficit'],
                    colorscale="OrRd",
                    showscale=True,
                    colorbar=dict(title="Resource Deficit", x=1.15)
                ),
                customdata=df[['Zone', 'allocated_units']],
                hovertemplate="<b>Zone: %{customdata[0]}</b><br>Total Risk: %{x:.3f}<br>Resource Deficit: %{y:.3f}<br>Units Allocated: %{customdata[1]}<extra></extra>"
            ))
            
            # Add text labels selectively to avoid clutter
            df_high_priority = df[df['resource_deficit'] > 0.1]
            fig.add_trace(go.Scatter(
                x=df_high_priority['Integrated_Risk_Score'], y=df_high_priority['resource_deficit'],
                mode='text',
                text=df_high_priority['Zone'],
                textposition="top center",
                textfont=dict(size=10, color='#444'),
                showlegend=False,
                hoverinfo='none'
            ))

            # Quadrant Lines and Annotations
            fig.add_vline(x=mean_risk, line_width=1, line_dash="dash", line_color="darkgrey")
            fig.add_hline(y=0, line_width=2, line_color="black")
            
            # CORRECTED and IMPROVED Annotations
            anno_font = dict(family="Arial, sans-serif", size=12, color="white")
            fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.98, text="<b>URGENT DEFICIT</b><br>(High Risk, High Need)", showarrow=False, font=anno_font, bgcolor="#D32F2F", xanchor='right', yanchor='top', borderpad=4, bordercolor="#D32F2F")
            fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.98, text="<b>POTENTIAL SURPLUS</b><br>(Low Risk, High Need)", showarrow=False, font=anno_font, bgcolor="#1E90FF", xanchor='left', yanchor='top', borderpad=4, bordercolor="#1E90FF")
            fig.add_annotation(xref="paper", yref="paper", x=0.98, y=0.02, text="<b>STABLE</b> (High Risk, Covered)", showarrow=False, font=dict(family="Arial, sans-serif", size=12, color="#333"), xanchor='right', yanchor='bottom', borderpad=4)
            fig.add_annotation(xref="paper", yref="paper", x=0.02, y=0.02, text="<b>ADEQUATE</b> (Low Risk, Covered)", showarrow=False, font=dict(family="Arial, sans-serif", size=12, color="#333"), xanchor='left', yanchor='bottom', borderpad=4)


            fig.update_layout(
                title_text="Strategic Deployment Matrix",
                xaxis_title="Zone Risk Profile →",
                yaxis_title="← Uncovered Risk (Deficit) →",
                height=550,
                plot_bgcolor='white',
                showlegend=False,
                xaxis=dict(gridcolor='#e5e5e5', zeroline=False),
                yaxis=dict(gridcolor='#e5e5e5', zeroline=True, zerolinewidth=2, zerolinecolor='black'),
                margin=dict(l=60, r=40, t=60, b=60)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in allocation opportunity plot: {e}", exc_info=True)
            st.warning("Could not display Allocation Opportunity plot.")
            
    def _plot_risk_momentum(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a Threat Vector Analysis chart (comet chart).
        This elegant design shows current magnitude, historical position, and trajectory with high-impact visuals.
        """
        st.markdown("**Analysis:** This chart reveals the trajectory of risk for the top zones. The **large colored dot** is the current risk. The **small grey dot** is the risk 6 hours ago. The connecting line and arrow show the path, or 'threat vector'. A **vibrant red arrow** indicates rapidly worsening conditions, demanding immediate attention.")
        try:
            top_zones_df = kpi_df.nlargest(10, 'Integrated_Risk_Score')
            risk_now = top_zones_df[['Zone', 'Integrated_Risk_Score']].set_index('Zone')
            # For demonstration, we simulate historical data.
            risk_6hr_ago = (risk_now * np.random.normal(1.0, 0.2, risk_now.shape)).clip(0, 1)
            
            plot_data = risk_now.join(risk_6hr_ago.rename(columns={'Integrated_Risk_Score': 'Risk_6hr_ago'}))
            plot_data = plot_data.sort_values('Integrated_Risk_Score', ascending=False).reset_index()
            plot_data['Momentum'] = plot_data['Integrated_Risk_Score'] - plot_data['Risk_6hr_ago']

            fig = go.Figure()
            
            # --- High-Contrast Action Colors ---
            INCREASING_COLOR = "#D32F2F"  # Material Design Red
            DECREASING_COLOR = "#1976D2"  # Material Design Blue

            # Add segments and arrows in a single loop
            for i, row in plot_data.iterrows():
                is_increasing = row['Momentum'] > 0
                arrow_color = INCREASING_COLOR if is_increasing else DECREASING_COLOR
                # Line segment (comet tail)
                fig.add_shape(type='line', x0=row['Risk_6hr_ago'], y0=row['Zone'], x1=row['Integrated_Risk_Score'], y1=row['Zone'], line=dict(color='#B0BEC5', width=2))
                # Arrowhead
                fig.add_annotation(ax=row['Risk_6hr_ago'], ay=row['Zone'], axref='x', ayref='y',
                                   x=row['Integrated_Risk_Score'], y=row['Zone'], xref='x', yref='y',
                                   showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2, arrowcolor=arrow_color)

            # Trace for the historical points
            fig.add_trace(go.Scatter(
                x=plot_data['Risk_6hr_ago'], y=plot_data['Zone'],
                mode='markers', name='6 Hrs Ago',
                marker=dict(color='#78909C', size=8),
                hovertemplate="<b>Zone:</b> %{y}<br><b>Past Risk:</b> %{x:.3f}<extra></extra>"
            ))

            # Trace for the current points, with color opacity encoding magnitude
            def get_rgba(is_increasing, risk_score):
                base_color = '211, 47, 47' if is_increasing else '25, 118, 210'
                opacity = 0.4 + (risk_score * 0.6) # Scale opacity from 0.4 to 1.0
                return f'rgba({base_color}, {opacity})'

            plot_data['marker_color'] = [get_rgba(row['Momentum'] > 0, row['Integrated_Risk_Score']) for i, row in plot_data.iterrows()]
            
            fig.add_trace(go.Scatter(
                x=plot_data['Integrated_Risk_Score'], y=plot_data['Zone'],
                mode='markers', name='Current',
                marker=dict(
                    color=plot_data['marker_color'],
                    size=16,
                    line=dict(width=1, color='rgba(0,0,0,0.6)')
                ),
                customdata=plot_data['Momentum'],
                hovertemplate="<b>Zone:</b> %{y}<br><b>Current Risk:</b> %{x:.3f}<br><b>Momentum (6hr):</b> %{customdata:+.3f}<extra></extra>"
            ))

            # --- Final Layout Polish for a professional, "mission control" feel ---
            fig.update_layout(
                title_text="<b>Threat Vector Analysis</b>",
                title_x=0.5,
                title_font=dict(size=20, family="Arial, sans-serif"),
                xaxis_title="Integrated Risk Score", yaxis_title=None,
                height=550,
                plot_bgcolor='white',
                paper_bgcolor='white',
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    traceorder="normal",
                    font=dict(size=12, color="#333"),
                    bgcolor='rgba(255,255,255,0.5)',
                    bordercolor="rgba(0,0,0,0.1)",
                    borderwidth=1
                ),
                yaxis=dict(
                    autorange="reversed",
                    showgrid=True,
                    gridcolor='rgba(221, 221, 221, 0.5)'
                ),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(221, 221, 221, 0.5)',
                    zeroline=False
                ),
                margin=dict(l=80, r=40, t=100, b=80)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in risk momentum plot: {e}", exc_info=True)
            st.warning("Could not display Risk Momentum plot.")
            
    # --- SME Addition: New High-Value Plot 3 (Commercial Grade, Improved) ---
    def _plot_critical_zone_anatomy(self, kpi_df: pd.DataFrame):
        """
        [SME VISUALIZATION] Plots a detailed Risk Composition Profile.
        This dissects the anatomy of risk for critical zones, guiding the type of response.
        """
        st.markdown("**Analysis:** This chart dissects the *composition* of risk for the most critical zones. The **colored segments** show the proportional contribution of each risk driver. The **grey lollipop marker** on the right shows the absolute `Integrated_Risk_Score`, providing crucial context on the overall magnitude of the threat.")
        try:
            risk_cols = ['Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score']
            if not all(col in kpi_df.columns for col in risk_cols): st.error("Data missing for Zone Anatomy plot."); return

            df_top = kpi_df.nlargest(7, 'Integrated_Risk_Score').copy()
            
            # Normalize the risk components for each zone
            df_top['total_component_risk'] = df_top[risk_cols].sum(axis=1) + 1e-9 # Avoid division by zero
            df_norm = df_top.copy()
            for col in risk_cols:
                df_norm[col] = (df_top[col] / df_top['total_component_risk']) * 100

            # Sort by total integrated risk for a clear hierarchy
            df_norm = df_norm.sort_values('Integrated_Risk_Score', ascending=True)
            
            fig = go.Figure()
            
            colors = {'Violence': '#D32F2F', 'Accident': '#FBC02D', 'Medical': '#1E90FF'}
            
            for risk_type, color in colors.items():
                col_name = f"{risk_type} Clustering Score" if risk_type != 'Medical' else 'Medical Surge Score'
                fig.add_trace(go.Bar(
                    y=df_norm['Zone'],
                    x=df_norm[col_name],
                    name=risk_type,
                    orientation='h',
                    marker=dict(color=color, line=dict(color='white', width=1.5)),
                    text=df_norm[col_name].apply(lambda x: f'{x:.0f}%' if x > 10 else ''), # Only show significant percentages
                    textposition='inside',
                    insidetextanchor='middle',
                    insidetextfont=dict(color='white', size=11, family='Arial Black'),
                    hovertemplate="<b>%{y}</b><br>%{name}: %{x:.1f}%<extra></extra>"
                ))

            # Add a secondary axis for the absolute Integrated Risk Score
            fig.add_trace(go.Scatter(
                x=[105] * len(df_norm), # Position to the right of the 100% bar
                y=df_norm['Zone'],
                mode='markers+text',
                marker=dict(color='#607D8B', size=10, symbol='circle', line=dict(width=1, color='white')),
                text=df_norm['Integrated_Risk_Score'].apply(lambda x: f'{x:.2f}'),
                textposition="middle right",
                textfont=dict(size=12, color='#37474F'),
                hovertemplate="<b>%{y}</b><br>Total Risk: %{text}<extra></extra>",
                showlegend=False
            ))
            
            # Add line segments for the "lollipop" effect
            for i, row in df_norm.iterrows():
                 fig.add_shape(type='line', x0=100, y0=row['Zone'], x1=105, y1=row['Zone'], line=dict(color='#B0BEC5', width=1.5))


            fig.update_layout(
                barmode='stack',
                title_text="<b>Risk Anatomy of Critical Zones</b>",
                title_x=0.5,
                xaxis=dict(
                    showgrid=False,
                    showline=False,
                    showticklabels=False,
                    zeroline=False,
                    range=[0, 115] # Extend range to accommodate annotations
                ),
                yaxis=dict(
                    showgrid=False,
                    showline=False,
                    showticklabels=True,
                    title=None,
                ),
                height=500,
                plot_bgcolor='white',
                paper_bgcolor='white',
                legend_title_text='Risk Driver',
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font_size=12),
                margin=dict(t=80, b=40, l=40, r=40)
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            logger.error(f"Error in critical zone anatomy plot: {e}", exc_info=True)
            st.warning("Could not display Critical Zone Anatomy plot.")
            
    ##END OF NEW PLOTS
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
        """
        [SME VISUALIZATION] Renders a comprehensive and structured glossary for all Key Performance Indicators.
        """
        with st.expander("V. Key Performance Indicator (KPI) Glossary", expanded=False):
            kpi_defs = {
                "Integrated Risk Score": {
                    "description": "The final, synthesized risk metric used for all operational decisions. It is the single source of truth for prioritizing zones and allocating resources.",
                    "question": "Which zone needs my attention the most right now, all factors considered?",
                    "relevance": "Drives the final resource allocation recommendations and provides the primary ranking of zones by overall threat level.",
                    "formula": r'''R_{\text{int}} = \sum_{i} w_i \cdot \text{RiskComponent}_i'''
                },
                "Ensemble Risk Score": {
                    "description": "A blended score from foundational statistical models (e.g., historical patterns, Bayesian inference). It represents the stable, baseline risk for a zone.",
                    "question": "What is the 'normal' or expected level of risk for this zone, given the current context (e.g., weather, day of week)?",
                    "relevance": "Provides a robust and less volatile risk assessment, preventing overreaction to minor, transient events.",
                    "formula": r'''R_{\text{ens}} = \sum_{k=1}^{K} w_k \cdot \text{normalize}(M_k)'''
                },
                "GNN Structural Risk": {
                    "description": "A measure of a zone's intrinsic, long-term vulnerability based on its position and connectivity within the city's road and social networks.",
                    "question": "Is this zone inherently dangerous or problematic, regardless of recent events?",
                    "relevance": "Identifies chronically at-risk areas that may require long-term strategic intervention (e.g., community policing, infrastructure changes) beyond daily resource allocation.",
                    "formula": r'''\text{PageRank}(z_i) = \frac{1-d}{N} + d \sum_{z_j \in N(z_i)} \frac{\text{PR}(z_j)}{|N(z_j)|}'''
                },
                "STGP Risk": {
                    "description": "A score representing the risk radiating from recent, severe incidents. It models risk as a fluid that decays over space and time from a point source.",
                    "question": "How much 'risk pressure' is a major incident in a neighboring zone putting on this one?",
                    "relevance": "Captures the spatiotemporal correlation of risk, ensuring that proximity to danger is accounted for, even across arbitrary zone boundaries.",
                    "formula": r'''f(s, t) \sim \mathcal{GP}(m(s, t), k((s, t), (s', t')))'''
                },
                "Game Theory Tension": {
                    "description": "A metric that quantifies a zone's contribution to system-wide resource competition. It's high when a high-risk zone also has a high expected volume of incidents.",
                    "question": "Which zones are causing the most strain and competition for our limited resources?",
                    "relevance": "Identifies which zones are the primary drivers of system-wide strain, helping to prioritize areas where under-resourcing would have the most severe consequences.",
                    "formula": r'''\text{Tension}_i = \text{Risk}_i \times \text{ExpectedIncidents}_i'''
                },
                "Chaos Sensitivity Score": {
                    "description": "A measure of system-wide volatility and fragility, based on concepts from Chaos Theory. It does not predict a specific incident, but rather the system's unpredictability.",
                    "question": "Is the city operating normally, or is it in a 'brittle' state where one small incident could cascade into a major crisis?",
                    "relevance": "Acts as a critical 'instability alarm' for command staff. A high score warns that the entire system is volatile and prone to cascading failures.",
                    "formula": r'''\lambda \approx \frac{1}{t} \ln \frac{\|\delta(t)\|}{\|\delta(0)\|}'''
                },
                "Anomaly Score": {
                    "description": "Measures the 'strangeness' of the current incident pattern (spatial and typological) compared to the historical norm for this time and day.",
                    "question": "Are we seeing unusual types of incidents or normal incidents in very unusual places?",
                    "relevance": "Detects novel threats or coordinated activity that simple volume-based metrics would miss. A high score is a clear signal that 'today is not a normal day.'",
                    "formula": r'''D_{KL}(P || Q) = \sum_{z} P(z) \log\frac{P(z)}{Q(z)}'''
                },
                "Resource Adequacy Index": {
                    "description": "A system-wide ratio of available units to the total expected demand, penalized by factors like hospital strain and traffic.",
                    "question": "As a whole, does my system have enough resources to handle the predicted demand for the next hour?",
                    "relevance": "Provides a top-line metric for command staff to understand overall system capacity. A low score indicates the system is severely overstretched and may require calling in additional resources.",
                    "formula": r'''\text{RAI} = \frac{\text{AvailableUnits}}{\sum E_i \times (1 + k_{\text{strain}})}'''
                }
            }
            
            for kpi, content in kpi_defs.items():
                st.markdown(f"**{kpi}**")
                st.markdown(f"*{content['description']}*")
                
                cols = st.columns([1, 2])
                with cols[0]:
                    st.markdown("**Question it Answers:**")
                    st.markdown(f"> {content['question']}")
                    
                    st.markdown("**Strategic Relevance:**")
                    st.markdown(f"> {content['relevance']}")

                with cols[1]:
                    st.markdown("**Mathematical Formulation:**")
                    st.latex(content['formula'])
                
                st.markdown("---")

    # --- SIDEBAR AND ACTIONS ---
# In main.py, inside the Dashboard class:
    def _render_sidebar(self):
        """Renders the sidebar for user controls and actions."""
        st.sidebar.title("Strategic Controls")
        st.sidebar.markdown("Adjust real-time factors to simulate different scenarios.")
        
        new_env = self._build_env_factors_from_sidebar()
        
        if new_env != st.session_state.env_factors:
            logger.info("EnvFactors updated, triggering rerun.")
            st.session_state.env_factors = new_env
            # If in simulation mode, we need to regenerate synthetic incidents with new factors
            if st.session_state.get('simulation_mode', False):
                st.session_state.current_incidents = self.dm._generate_synthetic_incidents(
                    st.session_state.env_factors, 
                    override_count=len(st.session_state.current_incidents)
                )
            st.rerun()

        # --- SME Fix: Introduce an explicit Simulation Mode ---
        st.sidebar.divider()
        st.sidebar.header("Scenario Simulation")
        
        # Initialize simulation_mode if it doesn't exist
        if 'simulation_mode' not in st.session_state:
            st.session_state.simulation_mode = False

        # The master toggle for simulation mode
        st.session_state.simulation_mode = st.sidebar.toggle(
            "Activate Simulation Mode", 
            value=st.session_state.simulation_mode,
            help="Turn this on to manually override live data with your own scenarios."
        )

        with st.sidebar.expander("Simulation Controls", expanded=st.session_state.simulation_mode):
            # The controls are disabled if simulation mode is off
            is_disabled = not st.session_state.simulation_mode

            # Control for Number of Incidents
            current_incident_count = len(st.session_state.get('current_incidents', []))
            new_incident_count = st.number_input(
                "Set Number of Active Incidents",
                min_value=0, value=current_incident_count, step=1,
                key="incident_simulator", disabled=is_disabled,
                help="Override the live data feed to simulate a different number of active incidents."
            )
            
            # Control for Number of Ambulances
            total_ambulances = len(self.dm.ambulances)
            current_available_ambulances = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
            new_ambulance_count = st.number_input(
                "Set Number of Available Ambulances",
                min_value=0, max_value=total_ambulances,
                value=current_available_ambulances, step=1,
                key="ambulance_simulator", disabled=is_disabled,
                help=f"Adjust the number of available units from the total fleet of {total_ambulances}."
            )

            # --- Logic to apply changes ONLY if simulation mode is ON and values have changed ---
            if st.session_state.simulation_mode:
                if new_incident_count != current_incident_count:
                    st.session_state.current_incidents = self.dm._generate_synthetic_incidents(
                        st.session_state.env_factors, override_count=new_incident_count
                    )
                    logger.info(f"User simulated new scenario with {new_incident_count} incidents.")
                    st.rerun()

                if new_ambulance_count != current_available_ambulances:
                    for i, amb_id in enumerate(self.dm.ambulances.keys()):
                        self.dm.ambulances[amb_id]['status'] = 'Disponible' if i < new_ambulance_count else 'En Misión'
                    logger.info(f"User simulated new scenario with {new_ambulance_count} available ambulances.")
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
        # --- CORRECTED: Added a unique, explicit key ---
        up_file = st.sidebar.file_uploader(
            "Upload Historical Incidents (JSON)",
            type=["json"],
            key="history_uploader"
        )
        if up_file:
            try:
                data = json.load(up_file)
                if not isinstance(data, list) or not all('location' in d and 'type' in d for d in data):
                    raise ValueError("Invalid JSON: Must be a list of incident objects.")
                st.session_state.historical_data = data
                st.sidebar.success(f"Loaded {len(data)} historical records.")
                st.rerun()
            except Exception as e:
                logger.error(f"Error loading uploaded data: {e}", exc_info=True)
                st.sidebar.error(f"Error loading data: {e}")

        # This part remains unchanged
        st.sidebar.divider()
        st.sidebar.header("Data & Reporting")
        self._sidebar_file_uploader()
        if st.sidebar.button("Generate & Download PDF Report", use_container_width=True):
            self._generate_report()
                ####END OF ADDITION
        st.sidebar.divider()
        st.sidebar.header("Data & Reporting")
        self._sidebar_file_uploader()
        if st.sidebar.button("Generate & Download PDF Report", use_container_width=True):
            self._generate_report()

    def _build_env_factors_from_sidebar(self) -> EnvFactorsWithTolerance:
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
        up_file = st.sidebar.file_uploader("Upload Historical Incidents (JSON)", type=["json"])
        if up_file:
            try:
                data = json.load(up_file)
                if not isinstance(data, list) or not all('location' in d and 'type' in d for d in data):
                    raise ValueError("Invalid JSON: Must be a list of incident objects.")
                st.session_state.historical_data = data
                st.sidebar.success(f"Loaded {len(data)} historical records.")
                st.rerun()
            except Exception as e:
                logger.error(f"Error loading uploaded data: {e}", exc_info=True)
                st.sidebar.error(f"Error loading data: {e}")

    def _generate_report(self):
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
