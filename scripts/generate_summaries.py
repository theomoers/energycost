#!/usr/bin/env python3
"""
Creates plots and summaries of generated cost data to visualize
the implications of configuration changes.

Author: theo
Date: 2025-09-04
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

plt.style.use('default')
sns.set_palette("husl")

KEY_TECHNOLOGIES = {
    'Solar': ['solar-utility', 'solar-rooftop residential', 'solar-rooftop commercial', 'solar',
              'solar-rooftop'],
    'Solar Thermal' : ['decentral solar thermal', 'central solar thermal'],
    'Wind': ['onwind', 'offwind', 'offwind-float'],
    'Batteries': ['Lithium-Ion-NMC-store', 'Lithium-Ion-LFP-store', 'battery storage', 
                  'Battery electric (passenger cars)', 'Battery electric (trucks)', 'home battery storage', 
                  'iron-air battery'],
    'H2 Electrolysis': ['SOEC', 'Alkaline electrolyzer large size', 'Alkaline electrolyzer small size',
                        'PEM electrolyzer small size', 'electrolysis'],
    'Coal': ['coal', 'lignite'],
    'Gas': ['CCGT', 'OCGT', 'gas'],
    'Nuclear': ['nuclear'],
    'Oil': ['oil']
}

KEY_PARAMETERS = ['FOM', 'VOM', 'investment', 'efficiency', 'lifetime', 'discount rate', 'CO2 intensity', 'fuel']
def load_cost_data(scenario_dir, config):

    cost_files = {}
    enabled = config.get('enabled', True)
    scenario_path = Path(scenario_dir)
    
    # Determine which files to load based on config
    if enabled:
        # Look for custom files first, then baseline files
        file_patterns = ['costs_custom_*.csv', 'costs_*.csv']
    else:
        file_patterns = ['costs_baseline_*.csv', 'costs_*.csv']
    
    for pattern in file_patterns:
        files = list(scenario_path.glob(pattern))
        if files:
            break
    
    if not files:
        logger.warning("No cost files found in scenario directory")
        return {}
    
    for file_path in files:
        try:
            year_str = file_path.stem.split('_')[-1]
            year = int(year_str)
            
            df = pd.read_csv(file_path)
            cost_files[year] = df
            logger.info(f"Loaded cost data for {year} ({len(df)} entries)")
            
        except (ValueError, Exception) as e:
            logger.warning(f"Skipping {file_path}: {e}")
            continue
    
    return cost_files


def extract_technology_data(cost_data):

    tech_data = {}
    
    for year, df in cost_data.items():
        year_data = {}
        
        for tech_group, tech_names in KEY_TECHNOLOGIES.items():
            group_data = {}
            
            for param in KEY_PARAMETERS:
                param_data = []
                
                for tech_name in tech_names:
                    tech_rows = df[(df['technology'] == tech_name) & 
                                 (df['parameter'] == param)]
                    
                    if not tech_rows.empty:
                        # Take the first match if multiple exist
                        value = tech_rows.iloc[0]['value']
                        unit = tech_rows.iloc[0]['unit']
                        param_data.append({
                            'technology': tech_name,
                            'value': value,
                            'unit': unit
                        })
                
                if param_data:
                    group_data[param] = param_data
            
            if group_data:
                year_data[tech_group] = group_data
        
        tech_data[year] = year_data
    
    return tech_data


def plot_cost_evolution(tech_data, summary_dir):
    """Create plots showing cost evolution over time for each technology group."""
    years = sorted(tech_data.keys())
    
    # Parameters to plot with their display names
    plot_params = {
        'FOM': 'Fixed O&M Cost',
        'VOM': 'Variable O&M Cost', 
        'investment': 'Investment Cost',
        'lifetime': 'Lifetime',
        'discount rate': 'Discount Rate',
        'CO2 intensity': 'CO2 Intensity',
        'fuel': 'Fuel Cost'
    }
    
    # Create a separate figure for each technology group
    for tech_group, tech_names in KEY_TECHNOLOGIES.items():
        # Check if this technology group has any data
        has_data = False
        for year in years:
            if tech_group in tech_data[year]:
                has_data = True
                break
        
        if not has_data:
            logger.info(f"No data found for technology group: {tech_group}")
            continue
        
        # Count available parameters for this technology group
        available_params = []
        for param in plot_params.keys():
            param_found = False
            for year in years:
                if (tech_group in tech_data[year] and 
                    param in tech_data[year][tech_group]):
                    param_found = True
                    break
            if param_found:
                available_params.append(param)
        
        if not available_params:
            continue
        
        # Create subplot grid - aim for roughly square layout
        n_params = len(available_params)
        if n_params <= 2:
            rows, cols = 1, n_params
        elif n_params <= 4:
            rows, cols = 2, 2
        elif n_params <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        fig.suptitle(f'{tech_group} Technology Evolution Over Time', fontsize=16, fontweight='bold')
        
        # Handle single subplot case
        if n_params == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes) if n_params > 1 else [axes]
        else:
            axes = axes.flatten()
        
        # Create consistent color mapping for all technologies in this group
        tech_colors = {}
        color_cycle = plt.cm.tab10(np.linspace(0, 1, len(tech_names)))
        for i, tech_name in enumerate(tech_names):
            tech_colors[tech_name] = color_cycle[i]
        
        plot_idx = 0
        
        for param in available_params:
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Plot all technologies in this group for the current parameter
            for tech_name in tech_names:
                values = []
                plot_years = []
                units = None
                
                for year in years:
                    if (tech_group in tech_data[year] and 
                        param in tech_data[year][tech_group]):
                        
                        param_data = tech_data[year][tech_group][param]
                        tech_entries = [entry for entry in param_data 
                                      if entry['technology'] == tech_name]
                        
                        if tech_entries:
                            values.append(tech_entries[0]['value'])
                            plot_years.append(year)
                            if units is None:
                                units = tech_entries[0]['unit']
                
                if len(values) > 0:  # Plot if we have any data points
                    color = tech_colors[tech_name]
                    if len(values) == 1:
                        ax.scatter(plot_years, values, label=tech_name, s=50, color=color,
                                   alpha=0.6)
                    else:
                        ax.plot(plot_years, values, marker='o', linewidth=2, 
                               label=tech_name, markersize=4, color=color,
                               alpha=0.6)

            # Format the subplot
            param_title = plot_params[param]
            if units:
                param_title += f' ({units})'
            
            ax.set_title(param_title, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel(param_title)
            ax.grid(True, alpha=0.3)
            
            plot_idx += 1
        
        # Remove empty subplots
        for idx in range(plot_idx, len(axes)):
            fig.delaxes(axes[idx])
        
        if tech_names:
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], color=tech_colors[tech_name], 
                                    marker='o', linewidth=2, label=tech_name)
                             for tech_name in tech_names]
            fig.legend(handles=legend_elements, loc='lower center', 
                      bbox_to_anchor=(0.5, -0.05), fontsize=10, ncol=min(4, len(tech_names)))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        safe_group_name = tech_group.lower().replace(' ', '_').replace('&', 'and')
        plt.savefig(summary_dir / f'{safe_group_name}_evolution.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated evolution plot for {tech_group}")


def plot_technology_group_comparison(tech_data, summary_dir):
    """Create comparison plots across all technology groups for key parameters."""
    years = sorted(tech_data.keys())
    
    # Key parameters for comparison
    comparison_params = ['FOM', 'VOM', 'investment']
    
    for param in comparison_params:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'{param} Cost Comparison Across Technology Groups', fontsize=16, fontweight='bold')
        
        # Plot average values with ranges for each technology group
        for tech_group in KEY_TECHNOLOGIES.keys():
            avg_values = []
            min_values = []
            max_values = []
            
            for year in years:
                if tech_group in tech_data[year] and param in tech_data[year][tech_group]:
                    param_data = tech_data[year][tech_group][param]
                    values = [entry['value'] for entry in param_data]
                    if values:
                        avg_values.append(np.mean(values))
                        min_values.append(np.min(values))
                        max_values.append(np.max(values))
                    else:
                        avg_values.append(np.nan)
                        min_values.append(np.nan)
                        max_values.append(np.nan)
                else:
                    avg_values.append(np.nan)
                    min_values.append(np.nan)
                    max_values.append(np.nan)
            
            # Only plot if we have valid data
            valid_indices = [i for i, v in enumerate(avg_values) if not np.isnan(v)]
            if valid_indices:
                valid_years = [years[i] for i in valid_indices]
                valid_avg = [avg_values[i] for i in valid_indices]
                valid_min = [min_values[i] for i in valid_indices]
                valid_max = [max_values[i] for i in valid_indices]
                
                # Determine line style based on technology type
                if tech_group in ['Solar', 'Wind', 'Batteries', 'H2 Electrolysis']:
                    linestyle = '-'
                    marker = 'o'
                    alpha_fill = 0.2
                else:
                    linestyle = '--'
                    marker = 's'
                    alpha_fill = 0.15
                
                # Plot the average line
                line = ax.plot(valid_years, valid_avg, marker=marker, linewidth=2, 
                              label=tech_group, linestyle=linestyle, markersize=6)
                
                # Add the cone (range) using fill_between
                color = line[0].get_color()
                ax.fill_between(valid_years, valid_min, valid_max, 
                               alpha=alpha_fill, color=color)
        
        ax.set_title(f'Average {param} Cost by Technology Group (with ranges)', fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel(f'{param} Cost')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(summary_dir / f'{param.lower()}_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated {param} comparison plot")


def create_efficiency_lifetime_summary(tech_data, summary_dir):
    """Create summary of efficiency and lifetime data."""
    years = sorted(tech_data.keys())
    latest_year = max(years)
    
    summary_data = []
    
    for tech_group, tech_names in KEY_TECHNOLOGIES.items():
        for tech_name in tech_names:
            if (tech_group in tech_data[latest_year]):
                tech_group_data = tech_data[latest_year][tech_group]
                
                efficiency_data = tech_group_data.get('efficiency', [])
                lifetime_data = tech_group_data.get('lifetime', [])
                
                efficiency_entries = [e for e in efficiency_data if e['technology'] == tech_name]
                lifetime_entries = [e for e in lifetime_data if e['technology'] == tech_name]
                
                efficiency = efficiency_entries[0]['value'] if efficiency_entries else 'N/A'
                efficiency_unit = efficiency_entries[0]['unit'] if efficiency_entries else ''
                
                lifetime = lifetime_entries[0]['value'] if lifetime_entries else 'N/A'
                lifetime_unit = lifetime_entries[0]['unit'] if lifetime_entries else ''
                
                if efficiency != 'N/A' or lifetime != 'N/A':
                    summary_data.append({
                        'Technology Group': tech_group,
                        'Technology': tech_name,
                        'Efficiency': f"{efficiency} {efficiency_unit}" if efficiency != 'N/A' else 'N/A',
                        'Lifetime': f"{lifetime} {lifetime_unit}" if lifetime != 'N/A' else 'N/A'
                    })
    
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        df_summary.to_csv(summary_dir / 'efficiency_lifetime_summary.csv', index=False)
        logger.info("Generated efficiency and lifetime summary")


def create_cost_change_summary(tech_data, summary_dir):
    """Create summary of cost changes over time."""
    years = sorted(tech_data.keys())
    if len(years) < 2:
        logger.warning("Need at least 2 years of data to calculate cost changes")
        return
    
    first_year = min(years)
    last_year = max(years)
    
    change_data = []
    
    for tech_group, tech_names in KEY_TECHNOLOGIES.items():
        for tech_name in tech_names:
            for param in ['FOM', 'VOM', 'investment', 'lifetime', 'discount rate', 'CO2 intensity', 'fuel']:
                # Get first year value
                first_value = None
                if (tech_group in tech_data[first_year] and 
                    param in tech_data[first_year][tech_group]):
                    param_data = tech_data[first_year][tech_group][param]
                    tech_entries = [e for e in param_data if e['technology'] == tech_name]
                    if tech_entries:
                        first_value = tech_entries[0]['value']
                
                # Get last year value
                last_value = None
                if (tech_group in tech_data[last_year] and 
                    param in tech_data[last_year][tech_group]):
                    param_data = tech_data[last_year][tech_group][param]
                    tech_entries = [e for e in param_data if e['technology'] == tech_name]
                    if tech_entries:
                        last_value = tech_entries[0]['value']
                
                if first_value is not None and last_value is not None and first_value != 0:
                    change_percent = ((last_value - first_value) / first_value) * 100
                    annual_rate = ((last_value / first_value) ** (1 / (last_year - first_year)) - 1) * 100
                    
                    change_data.append({
                        'Technology Group': tech_group,
                        'Technology': tech_name,
                        'Parameter': param,
                        f'{first_year} Value': first_value,
                        f'{last_year} Value': last_value,
                        'Total Change (%)': round(change_percent, 2),
                        'Annual Rate (%)': round(annual_rate, 2)
                    })
    
    if change_data:
        df_changes = pd.DataFrame(change_data)
        df_changes.to_csv(summary_dir / 'cost_changes_summary.csv', index=False)
        
        logger.info("Generated cost changes summary")


def generate_cost_summaries(scenario_dir, config):
    logger.info(f"Generating cost summaries for scenario in {scenario_dir}")
    
    try:
        scenario_path = Path(scenario_dir)
        summary_dir = scenario_path / "summaries"
        summary_dir.mkdir(exist_ok=True)
        
        cost_data = load_cost_data(scenario_dir, config)
        if not cost_data:
            logger.error("No cost data found to analyze")
            return
        
        tech_data = extract_technology_data(cost_data)
        plot_cost_evolution(tech_data, summary_dir)
        plot_technology_group_comparison(tech_data, summary_dir)
        create_efficiency_lifetime_summary(tech_data, summary_dir)
        create_cost_change_summary(tech_data, summary_dir)
        
        logger.info(f"Cost summaries generated successfully in {summary_dir}")
        
    except Exception as e:
        logger.error(f"Failed to generate cost summaries: {e}")
        raise
