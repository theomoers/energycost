#!/usr/bin/env python3
"""
Custom cost data generator for PyPSA-Earth

Author: theo
Date: 2025-09-04
"""

import pandas as pd
import numpy as np
import os
import argparse
import logging
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CostDataGenerator:

    def __init__(self, base_cost_dir, output_dir):
        self.base_cost_dir = Path(base_cost_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.columns = [
            'technology', 'parameter', 'value', 'unit', 
            'source', 'further description', 'currency_year'
        ]
        
        self.base_data = self._load_base_cost_data()
        
    def _load_base_cost_data(self):
        """
        Load base cost data from existing cost files.

        returns: Dictionary mapping years to cost dfs
        """
        base_data = {}
        
        cost_files = list(self.base_cost_dir.glob('costs_*.csv'))
        
        if not cost_files:
            raise FileNotFoundError(f"No cost files found in {self.base_cost_dir}")
        
        for cost_file in cost_files:
            year_str = cost_file.stem.split('_')[-1]
            try:
                year = int(year_str)
                df = pd.read_csv(cost_file)
                base_data[year] = df
                logger.info(f"Loaded base cost data for year {year} with {len(df)} entries")
            except ValueError:
                logger.warning(f"Could not extract year from filename: {cost_file} or other issue")
                continue
        
        if not base_data:
            raise ValueError("No valid cost files found")
        
        return base_data
    
    def get_available_technologies(self):
        """
        Get list of all available technologies from base data.
        
        Returns:
            Sorted list of unique technology names
        """
        technologies = set()
        for df in self.base_data.values():
            technologies.update(df['technology'].unique())
        return sorted(list(technologies))
    
    def get_technology_parameters(self, technology):
        """
        Get list of parameters for a specific technology.
            
        Returns:
            List of parameter names for the technology
        """
        parameters = set()
        for df in self.base_data.values():
            tech_data = df[df['technology'] == technology]
            parameters.update(tech_data['parameter'].unique())
        return sorted(list(parameters))
    
    def _apply_cost_projection(self, base_value, start_year, target_year, 
                              annual_change_rate):
        """
        Apply cost projection based on annual change rate.
        """
        years_diff = target_year - start_year
        return base_value * (1 + annual_change_rate) ** years_diff
    
    def generate_cost_data(self, 
                          target_years,
                          technology_overrides=None,
                          base_year=None):
        """
        Generate custom cost data for specified years.
        
        Args:
            target_years: List of years to generate cost data for
            technology_overrides: Dictionary of technology parameter overrides
                Format: {
                    'technology_name': {
                        'parameter_name': {
                            'base_value': float,
                            'annual_change_rate': float,  # optional
                            'unit': str,  # optional
                            'source': str,  # optional
                            'description': str  # optional
                        }
                    }
                }
            base_year: Base year for projections (defaults to earliest available year)
            
        Returns:
            Dictionary mapping years to generated cost dfs
        """
        if base_year is None:
            base_year = min(self.base_data.keys())
        
        if base_year not in self.base_data:
            raise ValueError(f"Base year {base_year} not found in base data")
        
        generated_data = {}
        
        for target_year in target_years:
            logger.info(f"Generating cost data for year {target_year}")
            
            closest_year = min(self.base_data.keys(), key=lambda x: abs(x - target_year))
            df = self.base_data[closest_year].copy()
            
            if technology_overrides:
                df = self._apply_technology_overrides(
                    df, technology_overrides, base_year, target_year
                )
            
            generated_data[target_year] = df
            
        return generated_data
    
    def _apply_technology_overrides(self, 
                                   df,
                                   technology_overrides,
                                   base_year,
                                   target_year):
        """
        Apply technology parameter overrides to the DataFrame.
            
        Returns:
            Modified DataFrame
        """
        df = df.copy()
        
        for tech_name, params in technology_overrides.items():
            for param_name, param_config in params.items():
                mask = (df['technology'] == tech_name) & (df['parameter'] == param_name)
                existing_rows = df[mask]
                
                base_value = param_config['base_value']
                annual_change_rate = param_config.get('annual_change_rate', 0.0)
                
                new_value = self._apply_cost_projection(
                    base_value, base_year, target_year, annual_change_rate
                )
                
                if len(existing_rows) > 0:
                    df.loc[mask, 'value'] = new_value
                    
                    if 'unit' in param_config:
                        df.loc[mask, 'unit'] = param_config['unit']
                    if 'source' in param_config:
                        df.loc[mask, 'source'] = param_config['source']
                    else:
                        df.loc[mask, 'source'] = 'Custom override'
                    if 'description' in param_config:
                        df.loc[mask, 'further description'] = param_config['description']
                        
                    logger.info(f"Updated {tech_name}.{param_name}: {base_value} -> {new_value:.4f}")
                    
                else:
                    new_row = {
                        'technology': tech_name,
                        'parameter': param_name,
                        'value': new_value,
                        'unit': param_config.get('unit', ''),
                        'source': param_config.get('source', 'Custom override'),
                        'further description': param_config.get('description', 'Custom parameter'),
                        'currency_year': param_config.get('currency_year', target_year)
                    }
                    
                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    logger.warning(f"!!! Added new parameter {tech_name}.{param_name}: {new_value:.4f}")
        
        return df
    
    def save_cost_data(self, cost_data, 
                      filename_prefix="costs"):
        for year, df in cost_data.items():
            output_file = self.output_dir / f"{filename_prefix}_{year}.csv"
            
            df = df[self.columns]
            df = df.sort_values(['technology', 'parameter']).reset_index(drop=True)
            df.to_csv(output_file, index=False)
            logger.info(f"Saved cost data for {year} to {output_file} ({len(df)} entries)")
    
def main():
    # for command line stuff (not really needed since we use yaml_cost_generator.py)
    parser = argparse.ArgumentParser(description='Generate custom PyPSA-Earth cost data')
    parser.add_argument('--base-dir', required=True,
                       help='Directory containing base cost files')
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for custom cost files')
    parser.add_argument('--years', nargs='+', type=int,
                       help='Target years for cost data generation')
    parser.add_argument('--base-year', type=int,
                       help='Base year for projections (default: earliest available)')
    parser.add_argument('--list-technologies', action='store_true',
                       help='List available technologies and exit')
    parser.add_argument('--show-params', type=str,
                       help='Show parameters for specific technology')
    
    args = parser.parse_args()
    
    try:
        generator = CostDataGenerator(args.base_dir, args.output_dir)
    except Exception as e:
        logger.error(f"Failed to initialize generator: {e}")
        return 1
    
    if args.list_technologies:
        technologies = generator.get_available_technologies()
        print("Available technologies:")
        for tech in technologies:
            print(f"  - {tech}")
        return 0
    
    if args.show_params:
        try:
            params = generator.get_technology_parameters(args.show_params)
            print(f"Parameters for technology '{args.show_params}':")
            for param in params:
                print(f"  - {param}")
        except Exception as e:
            logger.error(f"Failed to get parameters: {e}")
            return 1
        return 0
    
    if not args.years:
        logger.error("Please specify target years with --years")
        return 1
    
    try:
        cost_data = generator.generate_cost_data(
            target_years=args.years,
            technology_overrides=None,
            base_year=args.base_year
        )
        
        generator.save_cost_data(cost_data)
        
        logger.info("Cost data generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to generate cost data: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
