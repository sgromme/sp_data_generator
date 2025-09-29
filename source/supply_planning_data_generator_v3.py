
# Creating a new supply chain data generator
# This generator will create synthetic supply chain data for testing purposes.
# the demand/product will feed the other entities







""" 
Item -----------\
                  >---- Demand, Supply, Inventory
Location -------/
   |
   ---> Resource (capacity at that location)


"""


from xmlrpc import client
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from openai import OpenAI
from openai import APIStatusError, APIConnectionError, RateLimitError, OpenAIError
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import os, time


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
print("OpenAI API Key loaded:", openai_api_key is not None)

# Initialize OpenAI client, where should this be placed?
client = OpenAI(api_key=openai_api_key)

class SupplyPlanningDataGenerator:
    """Generate realistic supply planning data for testing optimization models."""
    
    def __init__(self, 
                 seed: Optional[int] = None,
                 openai_api_key: Optional[str] = None):
        """
        Initialize the data generator with optional seed for reproducibility.
        
        Args:
            seed: Random seed for reproducibility
            openai_api_key: API key for OpenAI services (if using GenAI features)
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        self.openai_api_key = openai_api_key
    
    def generate_products(self, 
                         num_products: int = 5, 
                         category_distribution: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Generate product data with realistic attributes.
        
        Args:
            num_products: Number of products to generate
            category_distribution: Dictionary mapping categories to their probability
                                  e.g. {'Electronics': 0.3, 'Clothing': 0.4, 'Food': 0.3}
        
        Returns:
            DataFrame with product information
        """
        if category_distribution is None:
            category_distribution = {
                'Electronics': 0.3, 
                'Clothing': 0.4, 
                'Home Goods': 0.2, 
                'Food': 0.1
            }
        
        categories = list(category_distribution.keys())
        probabilities = list(category_distribution.values())

        #generating product name with OpenAI , also name will be based on the category
        categories = np.random.choice(categories, size=num_products, p=probabilities)

        prompt = f"Generate {num_products} realistic and specific product names and wholesale costs for the category: {categories}. Return in this format name:cost;name:cost; ...  , the cost should not have a dollar sign, and no explanations."

        # Call OpenAI API with retries
        response = self.call_with_retries(prompt, 4, 1.0)

        productnames_costs = [name_cost.strip() for name_cost in response.split(';')]

        # Split names and costs
        productnames = [ item.split(':')[0].strip() for item in productnames_costs]
        productcosts = [ float(item.split(':')[1].strip()) for item in productnames_costs]
  
        # if the productnames is not the correct number product, recalculate
        if len(productnames) != num_products| len(productcosts) != num_products:
            print("Product names or costs generation failed or returned incorrect number, using fallback names.")
            productnames = [f'Product {i}' for i in range(1, num_products + 1)]
            productcosts = [np.round(np.random.uniform(5, 100, num_products), 2) for i in range(1, num_products + 1)]
 
        data = {
            'product_id': [f'P{i:04d}' for i in range(1, num_products + 1)],
            'product_name': productnames,
            'product_cost': productcosts,
            'category': categories #,
     #       'unit_cost': np.round(np.random.uniform(5, 100, num_products), 2),
     #       'setup_cost': np.random.randint(100, 1000, num_products),
     #       'setup_time': np.random.randint(30, 240, num_products),
     #       'production_time': np.random.randint(5, 60, num_products),
     #       'min_production_qty': np.random.randint(10, 100, num_products),
     #       'shelf_life_days': np.random.choice([30, 60, 90, 180, 365, 730, 1095], num_products)
        }
        
        # Add realistic product weights
        data['weight_kg'] = np.round(np.random.exponential(5, num_products), 2)
        data['weight_kg'] = np.clip(data['weight_kg'], 0.1, 50)
        
        # Add realistic volumes
        data['volume_m3'] = np.round(data['weight_kg'] * np.random.uniform(0.001, 0.01, num_products), 4)
        
        return pd.DataFrame(data)

    def generate_item_loc_resource(self, products_df: pd.DataFrame, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate item-location-resource relationships.
        Each product can be produced at certain factories and stored at certain warehouses.
        
        Returns:
            DataFrame with item-location-resource relationships
        """
        # This function can be expanded based on specific requirements.
        # Accumulate records in a list and construct a single DataFrame at the end
        records = []

        # Loop through products and facilities to create relationships
        for _, product in products_df.iterrows():
            for _, facility in facilities_df.iterrows():
                if facility['facility_type'] == 'Factory':
                    records.append({
                        'product_id': product['product_id'],
                        'facility_id': facility['facility_id'],
                        'resource_type': 'Production',
                        'throughput_rate': int(np.random.randint(100, 1000)),
                        'lead_time': int(np.random.randint(1, 5)),
                        'cost_per_unit': float(np.round(np.random.uniform(5, 20), 2)),
                        'setup_cost': int(np.random.randint(100, 1000)),
                        'setup_time': int(np.random.randint(30, 240)),
                        'production_time': int(np.random.randint(5, 60)),
                        'min_production_qty': int(np.random.randint(10, 100)),
                        'safety_stock': int(np.random.randint(10, 100))
                    })

        if records:
            df = pd.DataFrame.from_records(records)
        else:
            df = pd.DataFrame(columns=['product_id', 'facility_id', 'resource_type',
                                       'throughput_rate', 'lead_time', 'cost_per_unit',
                                       'setup_cost', 'setup_time', 'production_time',
                                       'min_production_qty', 'safety_stock'])

        return df

    def generate_facilities(self, num_facilities: int = 10, frequency: str = 'D') -> pd.DataFrame:
        """
        Generate facilities with resource-specific capacities.
        - production_capacity: applies mainly to Factory
        - storage_capacity: applies to Warehouse/DC
        - throughput_capacity: handling/ship-pick capacity per period for Warehouse/DC
        (replaces generic 'capacity')

        Returns:
            facilities_df: columns include facility_id, facility_name, facility_type,
                        latitude, longitude, operating_cost,
                        production_capacity, storage_capacity, throughput_capacity
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(getattr(self, "random_seed", None))

        # Determine periods per year based on frequency
        if frequency == 'D':
            periods_per_year = 365.0
        elif frequency == 'W':
            periods_per_year = 52.0
        elif frequency == 'M':
            periods_per_year = 12.0

        facility_types = ['Factory', 'Warehouse', 'Distribution Center']
        data = {
            'facility_id': [f'F{i:03d}' for i in range(1, num_facilities + 4)],
            'facility_name': [f'Facility {i}' for i in range(1, num_facilities + 4)],
            'facility_type': rng.choice(facility_types, num_facilities + 3),
            'latitude': np.random.uniform(30, 50, num_facilities + 3),
            'longitude': np.random.uniform(-120, -70, num_facilities + 3),
            'operating_cost': np.random.randint(10000/periods_per_year, 50000/periods_per_year  , num_facilities + 3)
        }

        
        df = pd.DataFrame(data)

        # Initialize resource capacities
        df['production_capacity'] = 0
        df['storage_capacity'] = 0
        df['throughput_capacity'] = 0  # <- replaces generic 'capacity'

        # Set capacities by facility type
        for i, row in df.iterrows():
            if row['facility_type'] == 'Factory':
                df.loc[i, 'production_capacity'] = np.random.randint(1000/periods_per_year, 5000/periods_per_year)
                df.loc[i, 'storage_capacity']   = np.random.randint(500/periods_per_year, 2000/periods_per_year)
                df.loc[i, 'throughput_capacity'] = 0  # throughput mainly at WH/DC
            elif row['facility_type'] == 'Warehouse':
                df.loc[i, 'production_capacity'] = 0
                df.loc[i, 'storage_capacity']   = np.random.randint(5000/periods_per_year, 15000/periods_per_year)
                df.loc[i, 'throughput_capacity'] = np.random.randint(5000/periods_per_year, 20000/periods_per_year)
            else:  # Distribution Center
                df.loc[i, 'production_capacity'] = 0
                df.loc[i, 'storage_capacity']   = np.random.randint(2000/periods_per_year, 8000/periods_per_year)
                df.loc[i, 'throughput_capacity'] = np.random.randint(7000/periods_per_year, 22000/periods_per_year) 

        return df  
    
    def generate_transportation_matrix(self, facilities_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate transportation costs and times between facilities.
        
        Args:
            facilities_df: DataFrame with facility information
            
        Returns:
            DataFrame with transportation costs and times
        """
        facility_ids = facilities_df['facility_id'].tolist()
        n_facilities = len(facility_ids)
        
        data = []
        for i in range(n_facilities):
            for j in range(n_facilities):
                if i != j:
                    # Calculate distance based on lat/long (simplified)
                    lat1, lon1 = facilities_df.iloc[i][['latitude', 'longitude']]
                    lat2, lon2 = facilities_df.iloc[j][['latitude', 'longitude']]
                    
                    # Simple distance calculation (not accurate geographic distance)
                    distance = np.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2) * 111  # rough km conversion
                    
                    # Add some randomness for real-world variation in routes
                    distance = distance * np.random.uniform(0.8, 1.2)
                    
                    data.append({
                        'from_facility': facility_ids[i],
                        'to_facility': facility_ids[j],
                        'distance_km': round(distance, 2),
                        'transport_cost': round(distance * np.random.uniform(0.5, 2.0), 2),
                        'transport_time_hours': round(distance / np.random.uniform(40, 80), 2)
                    })
        
        return pd.DataFrame(data)
    
    def generate_demand_data(self, 
                         products_df: pd.DataFrame,
                         facilities_df: pd.DataFrame,
                         capacity_variation: int = 1,
                         start_date: str = '2023-01-01',
                         periods: int = 52,
                         frequency: str = 'W',
                         seasonality: bool = True,
                         trend: bool = True,
                         noise_level: float = 0.2) -> pd.DataFrame:
        """
        Generate demand at Warehouse/DC nodes and size FACTORY production_capacity
        using capacity_variation. Throughput capacity is not modified.

        Also sets facilities_df['initial_inventory'] (aggregate units per facility).

        capacity_variation (FACTORIES ONLY):
            -1 : production_capacity < typical period load
            0 : production_capacity ≈ mean period load
            1 : production_capacity > peak period load
            2 : random per-factory among {-1,0,1}  (default for other values)
        """
        import numpy as np
        import pandas as pd

        rng = np.random.default_rng(getattr(self, "random_seed", None))

         # Determine periods per year based on frequency
        if frequency == 'D':
            periods_per_year = 365.0
        elif frequency == 'W':
            periods_per_year = 52.0
        elif frequency == 'M':
            periods_per_year = 12.0

        # -------- 0) helpers --------
        def _periods_per(two_weeks=True) -> float:
            """Return how many 'periods' ≈ two weeks, based on frequency."""
            freq = (frequency or '').upper()
            days_per = 7.0
            if freq.startswith('D'):
                days_per = 1.0
            elif freq.startswith('W'):
                days_per = 7.0
            elif freq.startswith('M'):
                days_per = 30.0
            # two weeks worth of periods:
            return (14.0 / days_per) if two_weeks else (7.0 / days_per)

        def _pick_rule(cv):
            if cv in (-1, 0, 1):
                return cv
            return rng.choice([-1, 0, 1])

        # -------- 1) Demand generation at WH/DC (unchanged in spirit) --------
        date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)

        product_ids = products_df['product_id'].tolist()
        # demand only at Factory 
        wh_types = {'Factory'}
        warehouse_ids = facilities_df.loc[
            facilities_df['facility_type'].isin(wh_types), 'facility_id'
        ].tolist()

        records = []

        # Example category base volumes (keep your own if you already have them)
        category_volumes = {
            'Electronics': np.random.randint(50, 200),
            'Clothing':    np.random.randint(100, 500),
            'Home Goods':  np.random.randint(30, 150),
            'Food':        np.random.randint(200, 800)
        }

        for pid in product_ids:
            prow = products_df.loc[products_df['product_id'] == pid].iloc[0]
            category = prow.get('category', 'General')
            base_volume = category_volumes.get(category, 100)

            for fid in warehouse_ids:
                facility_scale = rng.uniform(0.5, 1.5)
                base = base_volume * facility_scale

                t = np.arange(len(date_range))
                trend_component = (0.1 * base * (t / max(1, len(t)-1))) if trend else 0.0

                if seasonality:
                    if frequency.upper().startswith('D'):
                        seasonal = 0.1 * base * np.sin(2 * np.pi * t / 7)
                    elif frequency.upper().startswith('W'):
                        seasonal = 0.3 * base * np.sin(2 * np.pi * t / 13)
                    else:
                        seasonal = 0.4 * base * np.sin(2 * np.pi * t / 12)
                else:
                    seasonal = 0.0

                noise = rng.normal(0.0, noise_level * base, len(date_range))
                series = base + trend_component + seasonal + noise
                series = np.maximum(series, 0).round().astype(int)

                for d, q in zip(date_range, series):
                    records.append({'date': d, 'product_id': pid, 'facility_id': fid, 'demand': int(q)})

        demand_df = pd.DataFrame(records)

        # -------- 2) Network per-period totals and per-WH/DC stats --------
        per_fac_period = (
            demand_df.groupby(['facility_id', 'date'], as_index=False)['demand']
            .sum()
            .rename(columns={'demand': 'period_demand'})
        )
        network_period = (
            per_fac_period.groupby('date', as_index=False)['period_demand']
            .sum()
            .rename(columns={'period_demand': 'network_period_demand'})
        )

        # Guard: degenerate case
        if network_period['network_period_demand'].sum() == 0:
            # Ensure columns exist, but don't attempt sizing without demand
            if 'production_capacity' not in facilities_df.columns:
                facilities_df['production_capacity'] = 0
            if 'initial_inventory' not in facilities_df.columns:
                facilities_df['initial_inventory'] = 0
            return demand_df

        # -------- 3) Size FACTORY production_capacity (capacity_variation) --------
        factory_mask = facilities_df['facility_type'] == 'Factory'
        factory_ids = facilities_df.loc[factory_mask, 'facility_id'].tolist()

        if factory_ids:
            # Use existing production_capacity as seed weights if available/sum>0
            if 'production_capacity' in facilities_df.columns:
                seed = facilities_df.loc[factory_mask, 'production_capacity'].fillna(0).astype(float).values
            else:
                seed = np.zeros(len(factory_ids), dtype=float)

            if np.isfinite(seed).all() and seed.sum() > 0:
                weights = seed / seed.sum()
            else:
                raw_w = rng.uniform(0.7, 1.3, size=len(factory_ids))
                weights = raw_w / raw_w.sum()

            weight_s = pd.Series(weights, index=factory_ids)

            net_mean = float(network_period['network_period_demand'].mean())
            net_med  = float(network_period['network_period_demand'].median())
            net_max  = float(network_period['network_period_demand'].max())
            net_q75  = float(network_period['network_period_demand'].quantile(0.75))

            if 'production_capacity' not in facilities_df.columns:
                facilities_df['production_capacity'] = 0

            cv_value = capacity_variation if capacity_variation in (-1, 0, 1, 2) else 2

            for fac in factory_ids:
                w = float(weight_s[fac])
                mean_f = max(1.0, w * net_mean)
                med_f  = max(1.0, w * net_med)
                max_f  = max(1.0, w * net_max)
                q75_f  = max(1.0, w * net_q75)

                rule = _pick_rule(cv_value if cv_value != 2 else 2)
                if rule == 2:  # random per factory
                    rule = _pick_rule(2)

                if rule == -1:
                    cap = int(max(1, np.floor(0.85 * min(med_f, q75_f))))  # below typical
                elif rule == 0:
                    cap = int(max(1, round(mean_f)))                        # ~mean
                else:  # rule == 1
                    cap = int(max(1, np.ceil(1.15 * max_f)))               # > peak

                facilities_df.loc[facilities_df['facility_id'] == fac, 'production_capacity'] = int(cap/periods_per_year)

        # -------- 4) Add initial inventory for ALL facilities (aggregate units) --------
        # Rule of thumb: WH/DC get ~ two weeks of mean period demand;
        # factories get ~ 0.5 * their implied mean per-period production * two-weeks equivalent.
        two_weeks_in_periods = _periods_per(two_weeks=True)  # e.g., 2 for weekly, 14 for daily, ~0.47 for monthly

        # Start/ensure column
        if 'initial_inventory' not in facilities_df.columns:
            facilities_df['initial_inventory'] = 0

        # a) WH/DC inventory from their own mean period demand
        fac_means = per_fac_period.groupby('facility_id')['period_demand'].mean()
        for idx, row in facilities_df.iterrows():
            fid = row['facility_id']
            ftype = row['facility_type']
            if ftype in wh_types:
                mean_d = float(fac_means.get(fid, 0.0))
                init_inv = int(max(0, round(mean_d * two_weeks_in_periods)))
                facilities_df.at[idx, 'initial_inventory'] = int(init_inv/periods_per_year)

        # b) Factory inventory from implied mean production load (based on weights)
        if factory_ids:
            # reuse weights from above; if not computed (no factories), this block is skipped
            for idx, row in facilities_df.loc[factory_mask].iterrows():
                fid = row['facility_id']
                # implied mean production per period:
                implied_mean = float(weight_s[fid]) * float(network_period['network_period_demand'].mean())
                init_inv = int(max(0, round(0.5 * implied_mean * two_weeks_in_periods)))
                facilities_df.at[idx, 'initial_inventory'] = int(init_inv/periods_per_year)

        # NOTE: throughput_capacity is intentionally untouched here.

        return demand_df
    
    def generate_bill_of_materials(self, 
                                 products_df: pd.DataFrame, 
                                 max_components: int = 5,
                                 component_overlap: float = 0.3) -> pd.DataFrame:
        """
        Generate bill of materials (BOM) data for manufacturing.
        
        Args:
            products_df: DataFrame with product information
            max_components: Maximum number of components per product
            component_overlap: Probability of component reuse across products
            
        Returns:
            DataFrame with BOM data
        """
        
        # Generate components
        num_unique_components = int(len(products_df) * 2)  # 2x components as products
        components = [f'C{i:04d}' for i in range(1, num_unique_components + 1)]
        
        # Generate component properties
        component_data = {
            'component_id': components,
            'component_name': [f'Component {i}' for i in range(1, num_unique_components + 1)],
            'unit_cost': np.round(np.random.uniform(1, 50, num_unique_components), 2),
            'lead_time_days': np.random.randint(1, 30, num_unique_components)
        }
        components_df = pd.DataFrame(component_data)
        
        # Generate BOM relationships
        bom_data = []
        for _, product in products_df.iterrows():
            # Determine number of components for this product
            num_components = np.random.randint(2, max_components + 1)
            
            # Select components
            if np.random.random() < component_overlap and len(bom_data) > 0:
                # Reuse some components from existing BOMs
                existing_components = pd.DataFrame(bom_data)['component_id'].unique()
                if len(existing_components) > 0:
                    num_reused = min(np.random.randint(1, num_components), len(existing_components))
                    reused_components = np.random.choice(existing_components, num_reused, replace=False)
                    
                    for component_id in reused_components:
                        quantity = np.random.randint(1, 10)
                        bom_data.append({
                            'product_id': product['product_id'],
                            'component_id': component_id,
                            'quantity': quantity
                        })
                    
                    # Fill the rest with new components
                    remaining = num_components - num_reused
                    if remaining > 0:
                        new_components = np.random.choice(components, remaining, replace=False)
                        for component_id in new_components:
                            quantity = np.random.randint(1, 10)
                            bom_data.append({
                                'product_id': product['product_id'],
                                'component_id': component_id,
                                'quantity': quantity
                            })
            else:
                # Select all new components
                selected_components = np.random.choice(components, num_components, replace=False)
                for component_id in selected_components:
                    quantity = np.random.randint(1, 10)
                    bom_data.append({
                        'product_id': product['product_id'],
                        'component_id': component_id,
                        'quantity': quantity
                    })
        
        return pd.DataFrame(bom_data), components_df
    
    def generate_workforce_data(self, 
                              facilities_df: pd.DataFrame,
                              start_date: str = '2023-01-01',
                              periods: int = 52,
                              frequency: str = 'W') -> pd.DataFrame:
        """
        Generate workforce availability and cost data.
        
        Args:
            facilities_df: DataFrame with facility information
            start_date: Starting date for time series
            periods: Number of periods to generate
            frequency: Frequency string (D=daily, W=weekly, M=monthly)
            
        Returns:
            DataFrame with workforce data
        """
        factory_ids = facilities_df[facilities_df['facility_type'] == 'Factory']['facility_id'].tolist()
        date_range = pd.date_range(start=start_date, periods=periods, freq=frequency)
        
        data = []
        
        skill_levels = ['junior', 'intermediate', 'senior']
        
        for facility_id in factory_ids:
            # Base workforce size for this facility
            base_workforce = np.random.randint(20, 100)
            
            # Generate workforce fluctuations over time
            for date in date_range:
                # Add some seasonal and random variation
                day_of_year = date.dayofyear
                seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * day_of_year / 365)
                random_factor = np.random.uniform(0.9, 1.1)
                
                # Different workforce for different skill levels
                for skill in skill_levels:
                    if skill == 'junior':
                        skill_count = int(base_workforce * 0.5 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(15, 25)
                    elif skill == 'intermediate':
                        skill_count = int(base_workforce * 0.3 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(25, 40)
                    else:  # senior
                        skill_count = int(base_workforce * 0.2 * seasonal_factor * random_factor)
                        skill_cost = np.random.uniform(40, 60)
                    
                    data.append({
                        'date': date,
                        'facility_id': facility_id,
                        'skill_level': skill,
                        'available_workers': skill_count,
                        'hourly_cost': round(skill_cost, 2)
                    })
        
        return pd.DataFrame(data)
    
    def visualize_demand_patterns(self, demand_df: pd.DataFrame, product_ids: Optional[List[str]] = None):
        """
        Visualize demand patterns for selected products.
        
        Args:
            demand_df: DataFrame with demand data
            product_ids: List of product IDs to visualize (if None, select a random sample)
        """
        if product_ids is None:
            product_ids = random.sample(list(demand_df['product_id'].unique()), 3)
            
        plt.figure(figsize=(12, 8))
        
        for product_id in product_ids:
            product_demand = demand_df[demand_df['product_id'] == product_id]
            pivoted = product_demand.pivot_table(
                index='date', columns='facility_id', values='demand', aggfunc='sum'
            )
            total_demand = pivoted.sum(axis=1)
            plt.plot(total_demand.index, total_demand.values, label=f'Product {product_id}')
            
        plt.title('Demand Patterns by Product')
        plt.xlabel('Date')
        plt.ylabel('Total Demand')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def call_with_retries(self, prompt, max_retries:Optional[int] = 4, base_delay:Optional[float] =1.0) -> str:
        for attempt in range(max_retries):
            try:
                resp = client.responses.create(model="gpt-5", input=prompt)

                # Prefer the unified text helper if available
                if hasattr(resp, "output_text") and resp.output_text:
                    return resp.output_text

                # Fallbacks for other shapes (older/newer SDKs)
                if getattr(resp, "output", None):
                    parts = []
                    for item in resp.output or []:
                        for c in getattr(item, "content", []) or []:
                            if getattr(c, "type", None) in ("output_text", "text"):
                                parts.append(getattr(c, "text", ""))
                    if parts:
                        return "\n".join(parts)

                if getattr(resp, "choices", None):
                    # Legacy/chat-like shape
                    return resp.choices[0].message.content

                # If we reach here, we didn’t find text in any expected place
                raise ValueError("No text found in response payload.")

            except RateLimitError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            except APIConnectionError as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(base_delay * (2 ** attempt))
            except APIStatusError as e:
                # 4xx/5xx from the API – print and stop (usually not retryable except 429/5xx)
                raise RuntimeError(f"API status error {e.status_code}: {e.response}") from e
            except OpenAIError as e:
                # Generic SDK error – surface it
                raise RuntimeError(f"OpenAI error: {e}") from e

    def record_parameters(self, 
                          num_products: int,
                          num_facilities: int,
                          periods: int,
                          capacity_variation: int,
                          frequency: str,
                          start_date: str) -> pd.DataFrame:
        """
        Record the parameters used for dataset generation.
        """
        params = {
            "num_products": num_products,
            "num_facilities": num_facilities,
            "periods": periods,
            "capacity_variation": capacity_variation,
            "frequency": frequency,
            "start_date": start_date
        }
        return pd.DataFrame([params])

    def generate_full_dataset(self, 
                            num_products: int = 20,
                            num_facilities: int = 5,
                            periods: int = 52,
                            capacity_variation: int = 1,
                            frequency: str = 'W',
                            start_date: str = '2023-01-01') -> Dict[str, pd.DataFrame]:
        """
        Generate a complete, integrated dataset for supply chain planning.
        
        Args:
            num_products: Number of products to generate
            num_facilities: Number of facilities to generate
            periods: Number of time periods
            capacity_variation: Capacity variation strategy (-1, 0, 1, 2)
            frequency: Time frequency (D, W, M)
            start_date: Starting date for time series
            
        Returns:
            Dictionary of DataFrames containing the complete dataset
        """

        # Record parameters
        parameter_df = self.record_parameters(num_products, num_facilities, periods, capacity_variation, frequency, start_date)

        # Generate products
        products_df = self.generate_products(num_products)
        
        # Generate facilities
        facilities_df = self.generate_facilities(num_facilities, frequency)

        # Generate item-location-resource relationships (currently empty)
        ilr_df = self.generate_item_loc_resource(products_df, facilities_df)
        
        # Generate transportation matrix
        transport_df = self.generate_transportation_matrix(facilities_df)
        
        # Generate demand data
        demand_df = self.generate_demand_data(
            products_df=products_df,
            facilities_df=facilities_df,
            capacity_variation=capacity_variation,
            start_date=start_date,
            periods=periods,
            frequency=frequency
        )
        
        # Generate bill of materials
        bom_df, components_df = self.generate_bill_of_materials(products_df)
        
        # Generate workforce data
        workforce_df = self.generate_workforce_data(
            facilities_df=facilities_df,
            start_date=start_date,
            periods=periods,
            frequency=frequency
        )
        
        return {
            'parameters': parameter_df,
            'products': products_df,
            'item_location_resource': ilr_df,
            'facilities': facilities_df,
            'transportation': transport_df,
            'demand': demand_df,
            'bill_of_materials': bom_df,
            'components': components_df,
            'workforce': workforce_df
        }
    
    def export_to_excel(self, dataset: Dict[str, pd.DataFrame], filename: str = 'supply_planning_data.xlsx', working_directory:str = None):
        """
        Export the generated dataset to Excel.
        
        Args:
            dataset: Dictionary of DataFrames to export
            filename: Excel filename
            working_directory: Directory to save the file, if None uses the current script directory
        """
        # Get the directory of the current script or use the inputed 
        # working directory
        # and write to the same directory
        if not working_directory:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            file_path = os.path.join(script_dir, filename)
        else:
            file_path = os.path.join(working_directory, filename)
        
        with pd.ExcelWriter(file_path) as writer:
            for sheet_name, df in dataset.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def export_to_csv_directory(self, dataset: Dict[str, pd.DataFrame],  working_directory:str = None):
        """
        Export the generated dataset to a directory as csv files.
        
        Args:
            dataset: Dictionary of DataFrames to export
            working_directory: Directory to save the csv files, if None uses the current script directory
        """
        # Get the directory of the current script or use the inputed 
        # working directory
        # and write to the same directory
        if working_directory is None:
            working_directory = os.path.dirname(os.path.abspath(__file__))
    
        excel_dir = os.path.join(working_directory, "csv_data")
        os.makedirs(excel_dir, exist_ok=True)

        # Convert each sheet to a separate parquet file
        for sheet_name, df in dataset.items():
            # Clean sheet name for filename (avoid spaces/slashes etc.)
            safe_name = sheet_name.replace(" ", "_").replace("/", "-")
            file_path = os.path.join(excel_dir, f"{safe_name}.csv")
            df.to_csv(file_path, index=False)
            
    def export_to_parquet(self ,dataset: Dict[str, pd.DataFrame], working_directory:str = None  ):
        # Create parquet subfolder
        # ,base_dir: str, excel_file: str
        if working_directory is None:
            working_directory = os.path.dirname(os.path.abspath(__file__))
    
        parquet_dir = os.path.join(working_directory, "parquet_data")
        os.makedirs(parquet_dir, exist_ok=True)

        # Convert each sheet to a separate parquet file
        for sheet_name, df in dataset.items():
            # Clean sheet name for filename (avoid spaces/slashes etc.)
            safe_name = sheet_name.replace(" ", "_").replace("/", "-")
            file_path = os.path.join(parquet_dir, f"{safe_name}.parquet")

            df.to_parquet(file_path, engine="pyarrow", index=False)
            print(f"Saved: {file_path}")
        

# Example usage
if __name__ == "__main__":
    generator = SupplyPlanningDataGenerator(seed=42)
    dataset = generator.generate_full_dataset(capacity_variation=1)
    generator.export_to_excel(dataset)
    #generator.export_to_parquet(dataset)    
    # Visualize some data only in Jupyter
    #generator.visualize_demand_patterns(dataset['demand'])


