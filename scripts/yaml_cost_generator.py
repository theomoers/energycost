#!/usr/bin/env python3
"""
Manages yaml input and cost generator

Author: theo
Date: 2025-09-04
"""

import os
import sys
import yaml
import logging
import shutil
from datetime import datetime
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from generate_custom_costs import CostDataGenerator
from generate_summaries import generate_cost_summaries

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file: {e}")
        raise 

def validate_config(config):
    """Validate the configuration structure."""
    required_keys = ['enabled', 'scenario_name', 'base_year', 'target_years']
    
    for key in required_keys:
        if key not in config:
            logger.error(f"Missing required configuration key: {key}")
            return False
    
    if not isinstance(config['target_years'], list):
        logger.error("'target_years' must be a list")
        return False
    
    return True

def apply_validation_checks(config, overrides):
    """Apply validation checks based on configuration settings."""
    if not config.get('validation', {}).get('check_negative_values', True):
        return
    
    warn_extreme = config.get('validation', {}).get('warn_extreme_changes', True)
    min_cost = config.get('validation', {}).get('min_reasonable_cost', 0.01)
    
    for tech, params in overrides.items():
        for param, settings in params.items():
            base_value = settings.get('base_value', 0)
            annual_rate = settings.get('annual_change_rate', 0)
            
            if base_value < min_cost:
                logger.warning(f"{tech}.{param}: Base value {base_value} is very low")
            
            if warn_extreme and abs(annual_rate) > 0.2:
                logger.warning(f"{tech}.{param}: Extreme annual change rate {annual_rate:.1%}")
            
            # Project to final year to check for negative values
            final_year = max(config['target_years'])
            years_to_project = final_year - config['base_year']
            final_value = base_value * ((1 + annual_rate) ** years_to_project)
            
            if final_value < 0:
                logger.error(f"{tech}.{param}: Projection leads to negative value in {final_year}")

def process_technology_overrides(config):
    """Process technology overrides from config, only including enabled ones."""
    if not config.get('enabled', False):
        logger.info("Custom cost overrides are disabled in configuration")
        return {}
    
    tech_overrides = config.get('technology_overrides', {})
    processed_overrides = {}
    
    for tech_name, tech_params in tech_overrides.items():
        processed_params = {}
        
        for param_name, param_config in tech_params.items():
            if param_config.get('enabled', False):
                processed_params[param_name] = {
                    'base_value': param_config['base_value'],
                    'annual_change_rate': param_config['annual_change_rate'],
                    'unit': param_config.get('unit', ''),
                    'source': param_config.get('source', 'Custom YAML configuration'),
                    'description': param_config.get('description', 'Custom parameter override')
                }
                logger.info(f"Enabled override: {tech_name}.{param_name}")
            else:
                logger.debug(f"Skipped disabled override: {tech_name}.{param_name}")
        
        if processed_params:
            processed_overrides[tech_name] = processed_params
    
    logger.info(f"Processed {len(processed_overrides)} technology overrides")
    return processed_overrides

def generate_from_yaml_config(config_path=None):
    """
    Generate cost data based on YAML configuration file.
    """
    if config_path is None:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "custom_cost_config.yaml"
        logger.info(f"No config path provided, using default: {config_path}")
    
    config_path = Path(config_path).resolve()
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        logger.info("Please create a configuration file using the template.")
        return
    
    config = load_yaml_config(str(config_path))
    if not validate_config(config):
        logger.error("Configuration validation failed")
        return
    
    overrides = process_technology_overrides(config)
    
    if not overrides and config.get('enabled', False):
        logger.warning("No enabled technology overrides found in configuration")
        logger.info("Cost files will be generated using default values only")
    
    if overrides:
        apply_validation_checks(config, overrides)
    
    base_cost_dir = Path("/shared/share_cki25/data/costs/default-costs") 
    base_output_dir = Path("/shared/share_cki25/data/costs/custom-data")
    
    scenario_name = config['scenario_name']
    scenario_dir = base_output_dir / scenario_name
    scenario_dir.mkdir(parents=True, exist_ok=True)
    
    config_filename = Path(config_path).name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scenario_config_path = scenario_dir / f"config_{timestamp}_{config_filename}"
    
    try:
        shutil.copy2(config_path, scenario_config_path)
        logger.info(f"Copied configuration file to: {scenario_config_path}")
        
        # Also create a symlink or copy with a simple name for easy access
        simple_config_path = scenario_dir / f"config_{config_filename}"
        if simple_config_path.exists():
            simple_config_path.unlink()
        shutil.copy2(config_path, simple_config_path)
        
    except Exception as e:
        logger.warning(f"Could not copy configuration file: {e}")
    
    logger.info(f"Using scenario directory: {scenario_dir}")
    
    generator = CostDataGenerator(
        base_cost_dir=str(base_cost_dir),
        output_dir=str(scenario_dir)
    )
    
    target_years = config['target_years']
    base_year = config['base_year']
    
    logger.info(f"Generating cost data for scenario '{scenario_name}'")
    logger.info(f"Target years: {target_years}")
    logger.info(f"Base year: {base_year}")
    
    output_settings = config.get('output_settings', {})
    filename_prefix = output_settings.get('filename_prefix', 'costs_custom')
    include_baseline = output_settings.get('include_baseline', True)
    
    try:
        if overrides:
            result_dataframes = generator.generate_cost_data(
                target_years=target_years,
                technology_overrides=overrides,
                base_year=base_year
            )
            
            for year, df in result_dataframes.items():
                output_file = scenario_dir / f"{filename_prefix}_{year}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Generated custom cost file: {output_file}")
        
        # Generate baseline files for comparison if requested
        if include_baseline and overrides:
            baseline_dataframes = generator.generate_cost_data(
                target_years=target_years,
                technology_overrides={},
                base_year=base_year
            )
            
            for year, df in baseline_dataframes.items():
                baseline_file = scenario_dir / f"costs_baseline_{year}.csv"
                df.to_csv(baseline_file, index=False)
                logger.info(f"Generated baseline cost file: {baseline_file}")
        
        # If no overrides, just generate standard files
        if not overrides:
            standard_dataframes = generator.generate_cost_data(
                target_years=target_years,
                technology_overrides={},
                base_year=base_year
            )
            
            for year, df in standard_dataframes.items():
                output_file = scenario_dir / f"costs_{year}.csv"
                df.to_csv(output_file, index=False)
                logger.info(f"Generated standard cost file: {output_file}")
            
    except Exception as e:
        logger.error(f"Error generating cost data: {e}")
        return
    
    logger.info("Cost data generation completed")
    
    logger.info("Generating cost summaries and visualizations...")
    try:
        generate_cost_summaries(scenario_dir, config)
    except Exception as e:
        logger.warning(f"Failed to generate cost summaries: {e}")
        logger.info("Cost data was generated successfully, but summary generation failed")
    
    enabled_techs = len([t for t, params in config.get('technology_overrides', {}).items() 
                        if any(p.get('enabled', False) for p in params.values())])
    
    if config.get('enabled', False) and overrides:
        logger.info(f"Custom overrides were enabled")
        for tech, params in overrides.items():
            enabled_params = [p for p in params.keys()]
            logger.info(f"  - {tech}: {', '.join(enabled_params)}")
    else:
        logger.info(f"Custom overrides were disabled")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate custom cost data for PyPSA-Earth using YAML configuration'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to YAML configuration file (default: ../custom_cost_config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        generate_from_yaml_config(args.config)
    except Exception as e:
        logger.error(f"Failed to generate cost data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
