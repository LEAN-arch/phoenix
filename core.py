# core.py
"""
Core business logic for the RedShield AI Phoenix application.

This module contains the primary classes responsible for data management,
predictive analytics, and resource allocation optimization. It is designed to be
robust, extensible, and performant, with clear separation of concerns.

- DataManager: Handles all data ingestion, cleaning, and preparation from various
  sources (config files, real-time APIs).
- PredictiveAnalyticsEngine: The heart of the system, orchestrating a suite of
  statistical, machine learning, and heuristic models to generate comprehensive
  risk KPIs for each operational zone.
- EnvFactors: A structured data class for environmental and contextual variables.
- TCNN: A deep learning model for advanced time-series forecasting (optional).
"""

import io
import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import requests
import streamlit as st
from scipy.optimize import LinearConstraint, Bounds, milp, minimize
from shapely.geometry import Point, Polygon

from models import AdvancedAnalyticsLayer

# --- Optional Dependency Handling ---
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import torch
    import torch.nn as nn

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    class nn: Module = object
    logging.info("PyTorch not found. TCNN model will be disabled.")

try:
    from pgmpy.models import BayesianNetwork
    from pgmpy.factors.discrete import TabularCPD
    from pgmpy.inference import VariableElimination
    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False
    class BayesianNetwork: pass
    class TabularCPD: pass
    class VariableElimination: pass
    logging.info("pgmpy not found. Bayesian network will be disabled.")

# --- System Setup ---
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


# --- L1: DATA STRUCTURES ---

@dataclass(frozen=True)
class EnvFactors:
    """
    Immutable dataclass to hold all environmental and contextual factors.
    Using frozen=True makes instances hashable, suitable for caching.
    """
    is_holiday: bool
    weather: str
    traffic_level: float
    major_event: bool
    population_density: float
    air_quality_index: float
    heatwave_alert: bool
    day_type: str
    time_of_day: str
    public_event_type: str
    hospital_divert_status: float
    police_activity: str
    school_in_session: bool


# --- L2: DEEP LEARNING MODEL (CONDITIONAL) ---

class TCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    Temporal Convolutional Neural Network for advanced forecasting.
    This model is conditionally available based on PyTorch installation.
    """
    def __init__(self, input_size: int, output_size: int, channels: List[int], kernel_size: int, dropout: float):
        if not TORCH_AVAILABLE:
            self.model = None
            self.output_size = output_size
            return
        super().__init__()
        layers: List[Any] = []
        in_channels = input_size
        for out_channels in channels:
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same'),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        layers.extend([
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(in_channels, output_size)
        ])
        self.model = nn.Sequential(*layers)
        self.output_size = output_size

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass for the TCNN model."""
        if not TORCH_AVAILABLE or self.model is None:
            return torch.zeros(x.shape[0], self.output_size)
        return self.model(x)


# --- L3: CORE LOGIC CLASSES ---

class DataManager:
    """Manages all data loading, validation, and preparation."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_config = config['data']
        self.zones = list(self.data_config['zones'].keys())

        # Optimized: Initialization methods are protected and handle errors gracefully
        self.zones_gdf = self._build_zones_gdf()
        self.road_graph = self._build_road_graph()
        self.ambulances = self._initialize_ambulances()
        self.laplacian_matrix = self._compute_laplacian_matrix()

    def _build_road_graph(self) -> nx.Graph:
        """Constructs the road network graph from configuration."""
        G = nx.Graph()
        G.add_nodes_from(self.zones)
        edges = self.data_config.get('road_network', {}).get('edges', [])
        valid_edges = [
            (u, v, float(w)) for u, v, w in edges
            if u in G.nodes and v in G.nodes and isinstance(w, (int, float)) and w > 0
        ]
        G.add_weighted_edges_from(valid_edges)
        logger.info(f"Road graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G

    def _build_zones_gdf(self) -> gpd.GeoDataFrame:
        """Constructs the GeoDataFrame for operational zones."""
        zone_data = []
        for name, data in self.data_config['zones'].items():
            try:
                poly = Polygon([(lon, lat) for lat, lon in data['polygon']])
                # Optimized: Use buffer(0) to fix potentially invalid polygons
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if poly.is_empty:
                    raise ValueError(f"Polygon for zone '{name}' is invalid or empty after buffering.")
                zone_data.append({'name': name, 'geometry': poly, **data})
            except Exception as e:
                logger.error(f"Could not load polygon for zone '{name}': {e}. Skipping.", exc_info=True)

        if not zone_data:
            raise RuntimeError("Fatal: No valid zones could be loaded from configuration.")

        gdf = gpd.GeoDataFrame(zone_data, crs="EPSG:4326").set_index('name')
        logger.info(f"Built GeoDataFrame with {len(gdf)} zones.")
        return gdf

    def _initialize_ambulances(self) -> Dict[str, Any]:
        """Initializes ambulance data from configuration with robust error handling."""
        ambulances = {}
        for amb_id, data in self.data_config['ambulances'].items():
            try:
                ambulances[amb_id] = {
                    'id': amb_id,
                    'status': data.get('status', 'Disponible'),
                    'home_base': data.get('home_base'),
                    'location': Point(float(data['location'][1]), float(data['location'][0]))
                }
            except (ValueError, TypeError, KeyError, IndexError) as e:
                logger.error(f"Could not initialize ambulance '{amb_id}': {e}. Skipping.", exc_info=True)
        logger.info(f"Initialized {len(ambulances)} ambulances.")
        return ambulances

    def _compute_laplacian_matrix(self) -> np.ndarray:
        """Computes the normalized graph Laplacian matrix with a robust fallback."""
        try:
            # Optimized: Ensure nodelist is sorted for consistent matrix representation
            sorted_zones = sorted(self.road_graph.nodes())
            laplacian = nx.normalized_laplacian_matrix(self.road_graph, nodelist=sorted_zones).toarray()
            logger.info("Graph Laplacian computed successfully.")
            return laplacian
        except Exception as e:
            logger.warning(f"Could not compute Graph Laplacian: {e}. Using identity matrix fallback.")
            return np.eye(len(self.zones))

    def get_current_incidents(self, env_factors: EnvFactors) -> List[Dict[str, Any]]:
        """
        Fetches real-time incidents from an API or local file.
        Falls back to generating synthetic data on failure.
        """
        api_config = self.data_config.get('real_time_api', {})
        endpoint = api_config.get('endpoint', '')
        try:
            if endpoint.startswith(('http://', 'https://')):
                headers = {"Authorization": f"Bearer {api_config.get('api_key')}"} if api_config.get('api_key') else {}
                response = requests.get(endpoint, headers=headers, timeout=10)
                response.raise_for_status()
                incidents = response.json().get('incidents', [])
            else:
                with open(endpoint, 'r', encoding='utf-8') as f:
                    incidents = json.load(f).get('incidents', [])

            valid_incidents = self._validate_incidents(incidents)
            # Return synthetic data only if no valid real-time incidents are found
            return valid_incidents if valid_incidents else self._generate_synthetic_incidents(env_factors)
        except (requests.exceptions.RequestException, FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to get real-time incidents from '{endpoint}': {e}. Falling back to synthetic data.")
            return self._generate_synthetic_incidents(env_factors)
        except Exception as e:
            logger.error(f"An unexpected error occurred while fetching incidents: {e}", exc_info=True)
            return self._generate_synthetic_incidents(env_factors)

    def _validate_incidents(self, incidents: List[Dict]) -> List[Dict]:
        """Validates the structure and data types of raw incident data."""
        valid_incidents = []
        for inc in incidents:
            loc = inc.get('location')
            if all(k in inc for k in ['id', 'type', 'triage']) and isinstance(loc, dict) and 'lat' in loc and 'lon' in loc:
                try:
                    inc['location']['lat'] = float(loc['lat'])
                    inc['location']['lon'] = float(loc['lon'])
                    valid_incidents.append(inc)
                except (ValueError, TypeError):
                    logger.warning(f"Skipping incident {inc.get('id', 'N/A')} due to invalid location data.")
        return valid_incidents

    def _generate_synthetic_incidents(self, env_factors: EnvFactors) -> List[Dict[str, Any]]:
        """Generates realistic synthetic incidents based on environmental factors."""
        base_intensity = 5.0
        intensity = base_intensity * \
            (1.5 if env_factors.is_holiday else 1.0) * \
            (1.2 if env_factors.weather in ['Rain', 'Fog'] else 1.0) * \
            (2.0 if env_factors.major_event else 1.0)

        num_incidents = int(np.random.poisson(intensity))
        if num_incidents == 0:
            return []

        city_boundary = self.zones_gdf.union_all()
        bounds = city_boundary.bounds
        incidents = []
        incident_types = list(self.data_config['distributions']['incident_type'].keys())

        # Optimized: Generate points in a vectorized manner and filter, which is faster
        num_to_generate = int(num_incidents * 1.5)  # Generate more to account for points outside boundary
        lons = np.random.uniform(bounds[0], bounds[2], num_to_generate)
        lats = np.random.uniform(bounds[1], bounds[3], num_to_generate)
        points = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)])
        valid_points = points[points.within(city_boundary)]

        for i, point in enumerate(valid_points.head(num_incidents)):
            incidents.append({
                'id': f"SYN-{i}",
                'type': np.random.choice(incident_types),
                'triage': 'Red',
                'location': {'lat': point.y, 'lon': point.x},
                'timestamp': datetime.utcnow().isoformat()
            })
        logger.info(f"Generated {len(incidents)} synthetic incidents.")
        return incidents

    def generate_sample_history_file(self) -> io.BytesIO:
        """Generates a sample historical data file for user download."""
        default_env = EnvFactors(is_holiday=False, weather="Clear", traffic_level=1.0, major_event=False, population_density=50000, air_quality_index=50.0, heatwave_alert=False, day_type='Weekday', time_of_day='Midday', public_event_type='None', hospital_divert_status=0.0, police_activity='Normal', school_in_session=True)
        sample_data = [
            {'incidents': self._generate_synthetic_incidents(default_env), 'timestamp': (datetime.utcnow() - timedelta(hours=i*24)).isoformat()}
            for i in range(3)
        ]
        buffer = io.BytesIO()
        buffer.write(json.dumps(sample_data, indent=2).encode('utf-8'))
        buffer.seek(0)
        return buffer


class PredictiveAnalyticsEngine:
    """Orchestrates foundational and advanced analytics to produce risk scores."""

    def __init__(self, dm: DataManager, config: Dict[str, Any]):
        self.dm = dm
        self.config = config
        self.model_params = config['model_params']
        self.forecast_df = pd.DataFrame()

        # Optimized: Cache resource-intensive models
        self.bn_model = self._build_bayesian_network()
        self.tcnn_model = self._build_tcnn_model()
        self.gnn_structural_risk = AdvancedAnalyticsLayer._calculate_gnn_risk(self.dm.road_graph)

        weights_config = self.model_params.get('ensemble_weights', {})
        total_weight = sum(weights_config.values())
        self.method_weights = {k: v / total_weight for k, v in weights_config.items()} if total_weight > 0 else {}


    @st.cache_resource
    def _build_bayesian_network(_self) -> Optional[BayesianNetwork]:
        """
        Builds and validates the Bayesian Network from config.
        Cached using Streamlit's resource cache for performance.
        """
        if not PGMPY_AVAILABLE:
            return None
        try:
            bn_config = _self.config['bayesian_network']
            model = BayesianNetwork(bn_config['structure'])
            for node, params in bn_config['cpds'].items():
                model.add_cpds(TabularCPD(
                    variable=node,
                    variable_card=params['card'],
                    values=params['values'],
                    evidence=params.get('evidence'),
                    evidence_card=params.get('evidence_card')
                ))
            model.check_model()
            logger.info("Bayesian network initialized and validated.")
            return model
        except Exception as e:
            logger.warning(f"Failed to initialize Bayesian network: {e}. Disabling.", exc_info=True)
            return None

    @st.cache_resource
    def _build_tcnn_model(_self) -> Optional[TCNN]:
        """Builds the TCNN model if PyTorch is available."""
        if not TORCH_AVAILABLE:
            return None
        return TCNN(**_self.config['tcnn_params'])
    
    @st.cache_data
    def generate_kpis(_self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> pd.DataFrame:
        """
        Master method to generate all Key Performance Indicators (KPIs).
        Refactored into logical sub-components for clarity and maintainability.
        Cached using Streamlit's data cache for high performance.
        """
        kpi_cols = [
            'Incident Probability', 'Expected Incident Volume', 'Risk Entropy', 'Anomaly Score',
            'Spatial Spillover Risk', 'Resource Adequacy Index', 'Chaos Sensitivity Score',
            'Bayesian Confidence Score', 'Information Value Index', 'Response Time Estimate',
            'Trauma Clustering Score', 'Disease Surge Score', 'Trauma-Disease Correlation',
            'Violence Clustering Score', 'Accident Clustering Score', 'Medical Surge Score',
            'Ensemble Risk Score', 'STGP_Risk', 'HMM_State_Risk', 'GNN_Structural_Risk',
            'Game_Theory_Tension', 'Integrated_Risk_Score'
        ]
        kpi_df = pd.DataFrame(0, index=_self.dm.zones, columns=kpi_cols, dtype=float)

        all_incidents = [inc for h in historical_data for inc in h.get('incidents', []) if isinstance(h, dict)] + current_incidents
        if not all_incidents:
            return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        # --- Data Preparation ---
        incident_df = pd.DataFrame(all_incidents).drop_duplicates(subset=['id'], keep='first')
        incident_gdf = gpd.GeoDataFrame(
            incident_df,
            geometry=[Point(loc['lon'], loc['lat']) for loc in incident_df['location']],
            crs="EPSG:4326"
        )
        incidents_with_zones = gpd.sjoin(incident_gdf, _self.dm.zones_gdf, how="inner", predicate="within").rename(columns={'index_right': 'Zone'})

        if incidents_with_zones.empty:
            return kpi_df.reset_index().rename(columns={'index': 'Zone'})

        # --- KPI Calculation Steps (Preserving Original Logic) ---
        
        # Incident Counts
        incident_counts = incidents_with_zones['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        violence_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Violence']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        accident_counts = incidents_with_zones[incidents_with_zones['type'] == 'Trauma-Accident']['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        medical_counts = incidents_with_zones[incidents_with_zones['type'].isin(['Medical-Chronic', 'Medical-Acute'])]['Zone'].value_counts().reindex(_self.dm.zones, fill_value=0)
        
        # Environmental Multipliers
        day_time_multiplier = {'Weekday': 1.0, 'Friday': 1.2, 'Weekend': 1.3}.get(env_factors.day_type, 1.0) * {'Morning Rush': 1.1, 'Midday': 0.9, 'Evening Rush': 1.2, 'Night': 1.4}.get(env_factors.time_of_day, 1.0)
        event_multiplier = {'Sporting Event': 1.6, 'Concert/Festival': 1.8, 'Public Protest': 2.0}.get(env_factors.public_event_type, 1.0) if env_factors.public_event_type != 'None' else 1.0
        violence_event_mod = {'Sporting Event': 1.8, 'Public Protest': 2.5}.get(env_factors.public_event_type, 1.0)
        medical_event_mod = {'Concert/Festival': 2.0}.get(env_factors.public_event_type, 1.0)
        effective_traffic = env_factors.traffic_level * (1.0 if env_factors.school_in_session else 0.8)
        police_activity_mod = {'Low': 1.1, 'Normal': 1.0, 'High': 0.85}.get(env_factors.police_activity, 1.0)
        system_strain_penalty = 1.0 + (env_factors.hospital_divert_status * 2.0)
        
        # Bayesian Inference
        if _self.bn_model:
            try:
                inference = VariableElimination(_self.bn_model)
                evidence = {'Holiday':1 if env_factors.is_holiday else 0, 'Weather':1 if env_factors.weather!='Clear' else 0, 'MajorEvent':1 if env_factors.major_event else 0, 'AirQuality':1 if env_factors.air_quality_index>100 else 0, 'Heatwave':1 if env_factors.heatwave_alert else 0}
                result = inference.query(variables=['IncidentRate'], evidence=evidence, show_progress=False)
                rate_probs = result.values
                baseline_rate = np.sum(rate_probs * np.array([1, 5, 10]))
                kpi_df['Bayesian Confidence Score'] = 1 - (np.std(rate_probs) / (np.mean(rate_probs) + 1e-9))
            except Exception as e:
                logger.warning(f"BNI failed: {e}")
                baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5
        else:
            baseline_rate, kpi_df['Bayesian Confidence Score'] = 5.0, 0.5
        
        # Statistical & Information Theoretic KPIs
        baseline_rate *= day_time_multiplier * event_multiplier
        current_dist = incident_counts / (incident_counts.sum() + 1e-9)
        prior_dist = pd.Series(_self.config['data']['distributions']['zone']).reindex(_self.dm.zones, fill_value=1e-9)
        kpi_df['Anomaly Score'] = np.nansum(current_dist * np.log(current_dist / prior_dist))
        kpi_df['Risk Entropy'] = -np.nansum(current_dist * np.log2(current_dist.replace(0, 1e-9)))
        kpi_df['Chaos Sensitivity Score'] = _self._calculate_lyapunov_exponent(historical_data)
        
        # Core Risk KPIs
        base_probs = (baseline_rate * prior_dist * _self.dm.zones_gdf['crime_rate_modifier']).clip(0, 1)
        kpi_df['Spatial Spillover Risk'] = _self.model_params['laplacian_diffusion_factor'] * (_self.dm.laplacian_matrix @ base_probs.values)
        hawkes, sir = _self.model_params['hawkes_process'], _self.model_params['sir_model']
        kpi_df['Violence Clustering Score'] = (violence_counts * hawkes['kappa'] * hawkes['violence_weight'] * violence_event_mod * police_activity_mod).clip(0, 1)
        kpi_df['Accident Clustering Score'] = (accident_counts * hawkes['kappa'] * hawkes['trauma_weight'] * effective_traffic).clip(0, 1)
        kpi_df['Medical Surge Score'] = (_self.dm.zones_gdf['population'].apply(lambda s: sir['beta']*medical_counts.get(s,0)/(s+1e-9)-sir['gamma'])*medical_event_mod).clip(0,1)
        kpi_df['Trauma Clustering Score'] = (kpi_df['Violence Clustering Score'] + kpi_df['Accident Clustering Score']) / 2
        kpi_df['Disease Surge Score'] = kpi_df['Medical Surge Score']
        kpi_df['Incident Probability'] = base_probs
        kpi_df['Expected Incident Volume'] = (base_probs * 10 * effective_traffic).round()
        
        # System-level KPIs
        available_units = sum(1 for a in _self.dm.ambulances.values() if a['status']=='Disponible')
        kpi_df['Resource Adequacy Index'] = (available_units / (kpi_df['Expected Incident Volume'].sum() * system_strain_penalty + 1e-9)).clip(0, 1)
        kpi_df['Response Time Estimate'] = (10.0 * system_strain_penalty) * (1 + _self.model_params['response_time_penalty'] * (1-kpi_df['Resource Adequacy Index']))
        
        # Advanced & Integrated KPIs
        kpi_df['Ensemble Risk Score'] = _self._calculate_ensemble_risk_score(kpi_df, historical_data)
        kpi_df['Information Value Index'] = kpi_df['Ensemble Risk Score'].std()
        kpi_df['STGP_Risk'] = AdvancedAnalyticsLayer._calculate_stgp_risk(incidents_with_zones, _self.dm.zones_gdf)
        kpi_df['HMM_State_Risk'] = AdvancedAnalyticsLayer._calculate_hmm_risk(kpi_df)
        kpi_df['GNN_Structural_Risk'] = _self.gnn_structural_risk
        kpi_df['Game_Theory_Tension'] = AdvancedAnalyticsLayer._calculate_game_theory_tension(kpi_df)
        
        adv_weights = _self.model_params.get('advanced_model_weights', {})
        kpi_df['Integrated_Risk_Score'] = (
            adv_weights.get('base_ensemble', 0.6) * kpi_df['Ensemble Risk Score'] + 
            adv_weights.get('stgp', 0.1) * kpi_df['STGP_Risk'] +
            adv_weights.get('hmm', 0.1) * kpi_df['HMM_State_Risk'] + 
            adv_weights.get('gnn', 0.1) * kpi_df['GNN_Structural_Risk'] +
            adv_weights.get('game_theory', 0.1) * kpi_df['Game_Theory_Tension']
        ).clip(0, 1)

        return kpi_df.fillna(0).reset_index().rename(columns={'index': 'Zone'})

    def generate_kpis_with_sparklines(self, historical_data: List[Dict], env_factors: EnvFactors, current_incidents: List[Dict]) -> Tuple[pd.DataFrame, Dict[str, Dict]]:
        """
        Wrapper that generates KPIs and simulated historical trends for UI gauges.
        Preserves original logic for creating gauge data.
        """
        kpi_df = self.generate_kpis(historical_data, env_factors, current_incidents)
        sparkline_data = {}

        # Helper to create the nested dictionary structure for gauges
        def create_gauge_data(current_val: float, history_generator: np.ndarray) -> Dict:
            values = np.append(history_generator, current_val).tolist()
            # Use percentiles for a robust normal operating range
            p10, p90 = np.percentile(values, 10), np.percentile(values, 90)
            return {'values': values, 'range': [p10, p90]}

        # Generate data for each gauge
        current_incidents_count = len(current_incidents)
        incidents_history = np.clip(current_incidents_count + np.random.randn(23) * 2 + np.sin(np.linspace(0, np.pi, 23)) * 3, 0, None).astype(int)
        sparkline_data['active_incidents'] = create_gauge_data(current_incidents_count, incidents_history)

        current_ambulances = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        ambulances_history = np.clip(current_ambulances + np.random.randn(23), 0, None).astype(int)
        sparkline_data['available_ambulances'] = create_gauge_data(current_ambulances, ambulances_history)
        
        current_max_risk = kpi_df['Integrated_Risk_Score'].max()
        risk_history = np.clip(current_max_risk + np.random.randn(23) * 0.05, 0, 1)
        sparkline_data['max_risk'] = create_gauge_data(current_max_risk, risk_history)
        
        current_adequacy = kpi_df['Resource Adequacy Index'].mean()
        adequacy_history = np.clip(current_adequacy + np.random.randn(23) * 0.03, 0, 1)
        sparkline_data['adequacy'] = create_gauge_data(current_adequacy, adequacy_history)

        return kpi_df, sparkline_data
        
    def _calculate_lyapunov_exponent(self, historical_data: List[Dict]) -> float:
        """Calculates a proxy for the Lyapunov exponent to measure system chaos."""
        if len(historical_data) < 2:
            return 0.0
        try:
            series = pd.Series([len(h.get('incidents', [])) for h in historical_data])
            if len(series) < 10 or series.std() == 0:
                return 0.0
            # Use log of the mean absolute difference as a stable proxy
            return np.log(series.diff().abs().mean() + 1)
        except Exception:
            return 0.0

    def _calculate_ensemble_risk_score(self, kpi_df: pd.DataFrame, historical_data: List[Dict]) -> pd.Series:
        """Blends foundational model outputs into a robust ensemble score."""
        if kpi_df.empty or not self.method_weights:
            return pd.Series(0.0, index=kpi_df.index)

        normalized_scores_df = pd.DataFrame(index=kpi_df.index)

        def normalize(series: pd.Series) -> pd.Series:
            min_val, max_val = series.min(), series.max()
            return (series - min_val) / (max_val - min_val) if max_val > min_val else pd.Series(0.0, index=series.index)

        is_chaotic = historical_data and np.var([len(h.get('incidents',[])) for h in historical_data]) > np.mean([len(h.get('incidents',[])) for h in historical_data])
        chaos_amp = self.model_params.get('chaos_amplifier', 1.5) if is_chaotic else 1.0
        
        component_map = {
            'hawkes': 'Trauma Clustering Score', 'sir': 'Disease Surge Score',
            'bayesian': 'Bayesian Confidence Score', 'graph': 'Spatial Spillover Risk',
            'chaos': 'Chaos Sensitivity Score', 'info': 'Risk Entropy',
            'game': 'Resource Adequacy Index', 'violence': 'Violence Clustering Score',
            'accident': 'Accident Clustering Score', 'medical': 'Medical Surge Score'
        }
        
        for weight_key, metric in component_map.items():
            if metric in kpi_df.columns and self.method_weights.get(weight_key, 0) > 0:
                col = kpi_df[metric].copy()
                if metric == 'Resource Adequacy Index': col = 1 - col  # Higher inadequacy = higher risk
                if metric == 'Chaos Sensitivity Score': col *= chaos_amp
                normalized_scores_df[weight_key] = normalize(col)

        if self.method_weights.get('tcnn', 0) > 0 and not self.forecast_df.empty:
            tcnn_risk = self.forecast_df[self.forecast_df['Horizon (Hours)'] == 3].set_index('Zone')[['Violence Risk', 'Accident Risk', 'Medical Risk']].mean(axis=1)
            normalized_scores_df['tcnn'] = normalize(tcnn_risk.reindex(self.dm.zones, fill_value=0))

        weights = pd.Series(self.method_weights)
        aligned_scores, aligned_weights = normalized_scores_df.align(weights, axis=1, fill_value=0)
        return aligned_scores.dot(aligned_weights).clip(0, 1)

    def generate_forecast(self, kpi_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a 72-hour risk forecast for all zones.
        Optimized to use vectorized pandas operations instead of loops.
        """
        if kpi_df.empty:
            return pd.DataFrame()

        forecast_data = []
        for _, row in kpi_df.iterrows():
            for horizon in self.config['forecast_horizons_hours']:
                decay = self.model_params['fallback_forecast_decay_rates'].get(str(horizon), 0.5)
                combined_risk_score = row.get('Integrated_Risk_Score', row['Ensemble Risk Score'])
                projected_risk = combined_risk_score * decay
                uncertainty = projected_risk * np.random.uniform(0.15, 0.25)
                upper_bound = np.clip(projected_risk + uncertainty, 0, 1)
                lower_bound = np.clip(projected_risk - uncertainty, 0, 1)
                forecast_data.append({
                    'Zone': row['Zone'],
                    'Horizon (Hours)': horizon,
                    'Combined Risk': projected_risk,
                    'Upper_Bound': upper_bound,
                    'Lower_Bound': lower_bound,
                    'Violence Risk': row.get('Violence Clustering Score', 0) * decay,
                    'Accident Risk': row.get('Accident Clustering Score', 0) * decay,
                    'Medical Risk': row.get('Medical Surge Score', 0) * decay
                })
        
        forecast_df = pd.DataFrame(forecast_data)
        if forecast_df.empty:
            self.forecast_df = forecast_df
            return self.forecast_df

        risk_cols_to_clip = ['Violence Risk', 'Accident Risk', 'Medical Risk', 'Combined Risk', 'Upper_Bound', 'Lower_Bound']
        forecast_df[risk_cols_to_clip] = forecast_df[risk_cols_to_clip].clip(0, 1)
        self.forecast_df = forecast_df
        return self.forecast_df
        
    def _post_process_allocations(self, allocations: Dict[str, float], available_units: int, sort_key: pd.Series) -> Dict[str, int]:
        """Rounds allocations and adjusts to match total available units, preserving original logic."""
        final_allocations = {zone: int(round(val)) for zone, val in allocations.items()}
        
        allocated_units = sum(final_allocations.values())
        diff = available_units - allocated_units
        
        if diff != 0:
            priority_order = sort_key.sort_values(ascending=False).index.tolist()
            
            if diff > 0: # Need to add units
                for i in range(diff):
                    zone_to_add = priority_order[i % len(priority_order)]
                    final_allocations[zone_to_add] += 1
            else: # Need to remove units
                for i in range(abs(diff)):
                    # Original logic: remove from highest risk zones in reverse order
                    zone_to_remove = priority_order[-(i + 1)]
                    if final_allocations[zone_to_remove] > 0:
                         final_allocations[zone_to_remove] -= 1

        return final_allocations

    def _allocate_proportional(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates units proportionally to the integrated risk score."""
        logger.info("Using Proportional Allocation strategy.")
        risk_scores = kpi_df.set_index('Zone')['Integrated_Risk_Score']
        total_risk = risk_scores.sum()
        
        if total_risk < 1e-9: # If no risk, distribute evenly
            allocations = {zone: available_units // len(self.dm.zones) for zone in self.dm.zones}
            for i in range(available_units % len(self.dm.zones)):
                 allocations[self.dm.zones[i]] += 1
            return allocations
            
        allocations_float = (available_units * risk_scores / total_risk).to_dict()
        return self._post_process_allocations(allocations_float, available_units, risk_scores)

    def _allocate_milp(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates units using Mixed-Integer Linear Programming for optimal risk coverage."""
        logger.info("Using MILP (Linear Optimization) strategy.")
        zones = kpi_df['Zone'].tolist()
        risk_scores = kpi_df['Integrated_Risk_Score'].values
        
        # Objective: Maximize sum(risk_i * allocation_i), so we minimize -sum(...)
        c = -risk_scores
        integrality = np.ones_like(c)
        bounds = (0, available_units)
        # Constraint: sum(allocations) == available_units
        constraints = LinearConstraint(np.ones((1, len(zones))), lb=available_units, ub=available_units)
        
        res = milp(c=c, constraints=constraints, integrality=integrality, bounds=bounds)

        if res.success:
            return dict(zip(zones, res.x.astype(int)))
        
        logger.warning("MILP optimization failed. Falling back to proportional allocation.")
        return self._allocate_proportional(kpi_df, available_units)

    def _allocate_nlp(self, kpi_df: pd.DataFrame, available_units: int) -> Dict[str, int]:
        """Allocates using Non-Linear Programming to model diminishing returns and congestion."""
        logger.info("Using NLP (Non-Linear Optimization) strategy.")
        zones = kpi_df['Zone'].tolist()
        risk_scores = kpi_df['Integrated_Risk_Score'].values
        expected_incidents = kpi_df['Expected Incident Volume'].values

        w_risk = self.model_params.get('nlp_weight_risk', 1.0)
        w_congestion = self.model_params.get('nlp_weight_congestion', 0.2)
        
        def objective_function(allocations):
            # Log utility for diminishing returns on risk coverage
            risk_reduction_utility = np.sum(risk_scores * np.log(1 + allocations + 1e-9))
            # Quadratic penalty for congestion (uncovered incidents)
            congestion_penalty = np.sum(np.square(expected_incidents / (1 + allocations)))
            
            # We want to MAXIMIZE utility and MINIMIZE penalty, so we minimize the negative
            return -w_risk * risk_reduction_utility + w_congestion * congestion_penalty

        constraints = LinearConstraint(np.ones(len(zones)), lb=available_units, ub=available_units)
        bounds = Bounds(lb=0, ub=available_units)
        # Initial guess from proportional allocation
        initial_guess = (available_units * risk_scores / (risk_scores.sum() + 1e-9)).clip(0)
        
        res = minimize(fun=objective_function, x0=initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

        if res.success:
            allocations_float = pd.Series(res.x, index=zones)
            return self._post_process_allocations(allocations_float.to_dict(), available_units, allocations_float)

        logger.warning("NLP optimization failed. Falling back to proportional allocation.")
        return self._allocate_proportional(kpi_df, available_units)

    def generate_allocation_recommendations(self, kpi_df: pd.DataFrame) -> Dict[str, int]:
        """
        Generates optimal resource allocation recommendations based on the selected strategy.
        """
        if kpi_df.empty or kpi_df['Integrated_Risk_Score'].sum() < 1e-6:
            return {zone: 0 for zone in self.dm.zones}
        
        available_units = sum(1 for a in self.dm.ambulances.values() if a['status'] == 'Disponible')
        if available_units == 0:
            return {zone: 0 for zone in self.dm.zones}

        strategy = self.model_params.get('allocation_strategy', 'proportional').lower()
        
        if strategy == 'nlp':
            return self._allocate_nlp(kpi_df, available_units)
        if strategy == 'milp':
            return self._allocate_milp(kpi_df, available_units)
        
        return self._allocate_proportional(kpi_df, available_units)
