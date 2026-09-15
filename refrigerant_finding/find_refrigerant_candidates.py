#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import logging
import re
from scipy.spatial.distance import euclidean
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directories
OUTPUT_DIR = "refrigerant_finding/candidates"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Font settings for plots
plt.rcParams['font.family'] = 'Nimbus Sans'

# Define key refrigerant properties based on REFRIGERANT_PROPERTIES.md
KEY_PROPERTIES = [
    "MaxPartialCharge",
    "SMR_VSA1",
    "MinEStateIndex",
    "MolMR",
    "FpDensityMorgan3",
    "MinPartialCharge",
    "fr_halogen",
    "fr_alkyl_halide",
    "TPSA",
    "NumHAcceptors",
    "VSA_EState1",
    "LabuteASA",
    "MolLogP",
    "NumHeteroatoms",
    "NumRotatableBonds"
]

# Human-readable property names for visualization
PROPERTY_READABLE_NAMES = {
    "MaxPartialCharge": "Max Partial Charge",
    "SMR_VSA1": "Surface Area (MR)",
    "MinEStateIndex": "Min E-State Index",
    "MolMR": "Molar Refractivity",
    "FpDensityMorgan3": "Struct. Complexity",
    "MinPartialCharge": "Min Partial Charge",
    "fr_halogen": "Halogen Count",
    "fr_alkyl_halide": "Alkyl Halide Count",
    "TPSA": "Polar Surface Area",
    "NumHAcceptors": "H-Bond Acceptors",
    "VSA_EState1": "E-State Surface Area",
    "LabuteASA": "Accessible Surface Area",
    "MolLogP": "Lipophilicity",
    "NumHeteroatoms": "Heteroatom Count",
    "NumRotatableBonds": "Rotatable Bonds"
}

# Pre-defined refrigerant property means from REFRIGERANT_PROPERTIES.md
PREDEFINED_REFRIGERANT_MEANS = {
    "MaxPartialCharge": 0.4,
    "SMR_VSA1": 16.5,
    "MinEStateIndex": -4.1,
    "MolMR": 17.9,
    "FpDensityMorgan3": 1.4,
    "MinPartialCharge": -0.2,
    "fr_halogen": 4.0,
    "fr_alkyl_halide": 3.3,
    "TPSA": 0.0,
    "NumHAcceptors": 0.0,
    "VSA_EState1": 40.2,
    "LabuteASA": 39.4,
    "MolLogP": 2.1,
    "NumHeteroatoms": 4.0,
    "NumRotatableBonds": 0.2
}

def extract_means_from_md_file(md_file_path):
    """Extract refrigerant mean values from the markdown documentation file."""
    
    # Initialize with default values in case parsing fails
    extracted_means = PREDEFINED_REFRIGERANT_MEANS.copy()
    
    try:
        with open(md_file_path, 'r') as file:
            content = file.read()
            
            # Extract property blocks with regex
            for prop in KEY_PROPERTIES:
                pattern = rf'### \d+\. {prop} \(JS Distance: [\d\.]+\).*?- \*\*Refrigerants?\*\*: ([\d\.\-]+)'
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    try:
                        value = float(match.group(1))
                        extracted_means[prop] = value
                        logger.info(f"Extracted {prop} = {value} from documentation")
                    except ValueError:
                        logger.warning(f"Could not convert {match.group(1)} to float for {prop}")
        
        return extracted_means
    except Exception as e:
        logger.error(f"Error parsing markdown file: {e}")
        logger.info("Using default predefined means")
        return PREDEFINED_REFRIGERANT_MEANS

def calculate_descriptors(mol):
    """Calculate all RDKit descriptors for a molecule."""
    descriptor_values = {}
    
    if mol is None:
        return {desc: np.nan for desc in KEY_PROPERTIES}
    
    for desc_name, desc_function in Descriptors.descList:
        try:
            if desc_name in KEY_PROPERTIES:
                value = desc_function(mol)
                descriptor_values[desc_name] = value
        except Exception as e:
            descriptor_values[desc_name] = np.nan
            
    return descriptor_values

def get_mols_from_smiles(smiles_list):
    """Convert a list of SMILES strings to RDKit molecules."""
    molecules = []
    valid_count = 0
    problematic_smiles = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_count += 1
            else:
                logger.warning(f"Failed to parse SMILES at index {idx}: '{smiles}'")
                problematic_smiles.append((idx, smiles, "Failed to parse"))
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Error processing SMILES at index {idx}: '{smiles}'. Error: {error_message}")
            problematic_smiles.append((idx, smiles, error_message))
            mol = None
            
        molecules.append(mol)
    
    logger.info(f"Processed {len(smiles_list)} SMILES strings, {valid_count} valid molecules")
    return molecules, problematic_smiles

def compute_properties(smiles_list, category_name):
    """Compute key properties for a list of SMILES strings."""
    mols, problematic_smiles = get_mols_from_smiles(smiles_list)
    
    results = []
    for i, mol in enumerate(tqdm(mols, desc=f"Computing properties for {category_name}")):
        if mol is not None:
            try:
                # Add heavy atom count specifically for filtering
                heavy_atom_count = Chem.Descriptors.HeavyAtomCount(mol)
                
                # Calculate all descriptors
                desc_values = calculate_descriptors(mol)
                desc_values['smiles'] = smiles_list[i]
                desc_values['category'] = category_name
                desc_values['HeavyAtomCount'] = heavy_atom_count
                desc_values['MolName'] = f"{category_name}_{i}"
                
                results.append(desc_values)
            except Exception as e:
                logger.error(f"Error calculating descriptors for molecule at index {i} (SMILES: {smiles_list[i]}): {str(e)}")
    
    return pd.DataFrame(results)

def normalize_properties(df, properties):
    """Normalize properties to 0-1 range for distance calculation."""
    df_norm = df.copy()
    
    for prop in properties:
        if prop in df.columns:
            # Check if property has any non-NaN values
            if df[prop].notna().any():
                # Log property statistics before normalization
                logger.info(f"Normalizing {prop}: min={df[prop].min()}, max={df[prop].max()}, NaN count={df[prop].isna().sum()}")
                
                # Handle properties with specific issues
                if prop in ['MaxPartialCharge', 'MinPartialCharge']:
                    # These properties often need special handling
                    # Replace inf/-inf with NaN first
                    df[prop] = df[prop].replace([np.inf, -np.inf], np.nan)
                    
                # Get min/max, ignoring NaN values
                min_val = df[prop].min()
                max_val = df[prop].max()
                
                # Check for infinity values that would break normalization
                if np.isinf(min_val) or np.isinf(max_val):
                    logger.warning(f"Infinity values found in {prop}. Setting normalized values to 0.")
                    df_norm[prop] = 0
                    continue
                
                # Avoid division by zero
                if max_val > min_val:
                    # Normalize and handle NaN values
                    df_norm[prop] = (df[prop] - min_val) / (max_val - min_val)
                    # Replace any resulting NaN or inf values in normalized data
                    df_norm[prop] = df_norm[prop].replace([np.inf, -np.inf], np.nan).fillna(0)
                else:
                    df_norm[prop] = 0  # If all values are the same
                    logger.warning(f"Property {prop} has identical min and max values. Setting normalized values to 0.")
            else:
                logger.warning(f"Property {prop} has all NaN values. Setting normalized values to 0.")
                df_norm[prop] = 0
        else:
            logger.warning(f"Property {prop} not found in dataset.")
                
    return df_norm

def calculate_refrigerant_similarity(df, refrigerant_means, properties):
    """
    Calculate similarity (as Euclidean distance) between each molecule
    and the mean properties of refrigerants.
    """
    df_similarity = df.copy()
    
    # Create a vector of refrigerant means
    refrigerant_vector = np.array([refrigerant_means.get(prop, 0) for prop in properties])
    
    # Calculate distance for each molecule
    distances = []
    for _, row in df.iterrows():
        # Create vector of molecule properties
        molecule_vector = np.array([row.get(prop, 0) for prop in properties])
        
        # Replace NaN and Inf with 0 for distance calculation
        molecule_vector = np.nan_to_num(molecule_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Make sure refrigerant vector is also free of NaN/Inf values
        refrigerant_vector_safe = np.nan_to_num(refrigerant_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate Euclidean distance
        try:
            distance = euclidean(molecule_vector, refrigerant_vector_safe)
        except Exception as e:
            logger.warning(f"Error calculating distance: {e}. Using maximum distance.")
            distance = float('inf')  # Use a large value to rank this molecule last
            
        distances.append(distance)
    
    df_similarity['RefrigerantDistance'] = distances
    return df_similarity

def plot_top_candidates(candidates_df, refrigerant_means, norm_refrigerant_means, output_dir, top_molecules, n_top=20):
    """Create visualizations for top refrigerant candidates."""
    # Plot 1: Top candidates by distance
    top_candidates = candidates_df.sort_values('RefrigerantDistance').head(n_top)
    
    plt.figure(figsize=(12, 8))
    # Use different colors for refrigerants and candidates
    sns.barplot(x='RefrigerantDistance', y='MolName', hue='molecule_type', data=top_candidates, palette={'refrigerant': 'blue', 'candidate': 'orange'})
    plt.title(f'Top {n_top} Molecules Most Similar to Refrigerants')
    plt.xlabel('Distance to Refrigerant Profile (lower is better)')
    plt.legend(title='Molecule Type')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_candidates_distance.png', dpi=300)
    plt.close()
    
    # Plot 2: Property comparison for top 5 candidates
    top5 = top_candidates.head(5)
    
    # Prepare data for radar chart
    properties_to_plot = [p for p in KEY_PROPERTIES if p in refrigerant_means]
    
    # Get number of variables
    N = len(properties_to_plot)
    
    # Create angle for each property
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 14), subplot_kw=dict(polar=True))
    
    # Add refrigerant mean properties - use normalized values
    # For predefined means, we already have the normalized version in norm_refrigerant_means
    refrigerant_values = [norm_refrigerant_means.get(p, 0) for p in properties_to_plot]
    refrigerant_values += refrigerant_values[:1]  # Close the loop
    ax.plot(angles, refrigerant_values, 'o-', linewidth=2, label='Refrigerant Mean', color='black')
    ax.fill(angles, refrigerant_values, alpha=0.1)
    
    # Add top candidates - use normalized values for radar chart
    for i, (_, molecule) in enumerate(top5.iterrows()):
        # Use column name with _normalized suffix to get normalized values
        values = [molecule.get(f"{p}_normalized", 0) if not np.isnan(molecule.get(f"{p}_normalized", 0)) else 0 
                 for p in properties_to_plot]
        values += values[:1]  # Close the loop
        
        if molecule['molecule_type'] == 'refrigerant':
            color = 'blue'
            linestyle = '--'
            label = f"Known Refrigerant {i+1}"
        else:
            color = 'orange'
            linestyle = '-'
            label = f"Candidate {i+1}"
            
        ax.plot(angles, values, 'o-', linewidth=1, label=label, color=color, linestyle=linestyle)
    
    # Add property labels with human-readable names
    readable_labels = [PROPERTY_READABLE_NAMES.get(p, p) for p in properties_to_plot]
    plt.xticks(angles[:-1], readable_labels, size=18, fontweight='bold')
    
    # Add radial axis labels
    plt.yticks([0.25, 0.5, 0.75], ["0.25", "0.50", "0.75"], size=12)
    
    # Set y-axis limits for normalized values (0-1)
    ax.set_ylim(0, 1)
    
    # Add legend with larger font
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    plt.title('Property Comparison: Top Candidates vs Refrigerant Mean (Normalized)', size=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/top_candidates_radar.png', dpi=300)
    plt.close()
    
    # Plot 3: Distribution of distances
    plt.figure(figsize=(12, 8))
    
    # Separate histograms for refrigerants and candidates
    sns.histplot(candidates_df[candidates_df['molecule_type'] == 'refrigerant']['RefrigerantDistance'], 
                kde=True, color='blue', label='Known Refrigerants', alpha=0.6)
    sns.histplot(candidates_df[candidates_df['molecule_type'] == 'candidate']['RefrigerantDistance'], 
                kde=True, color='orange', label='Database Molecules', alpha=0.6)
    
    # Use the maximum distance of molecules in the top_molecules DataFrame
    # This represents the exact threshold for molecules included in top_n_molecules.csv
    threshold = top_molecules['RefrigerantDistance'].max()
    
    # Draw the threshold line
    plt.axvline(threshold, color='r', linestyle='--', 
                label=f'Top {len(top_molecules)} threshold')
    
    plt.title('Distribution of Distances to Refrigerant Profile')
    plt.xlabel('Distance (lower is better)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/distance_distribution.png', dpi=300)
    plt.close()
    
    # Plot 4: Scatter plot of two key properties - use unnormalized values
    plt.figure(figsize=(12, 8))
    
    # Get ALL refrigerants
    refrigerants = candidates_df[candidates_df['molecule_type'] == 'refrigerant']
    
    # Get ALL database molecules (those that passed the filter)
    all_database_molecules = candidates_df[candidates_df['molecule_type'] == 'candidate']
    
    # Plot ALL database molecules as background with distance-based coloring
    scatter = plt.scatter(all_database_molecules['MolLogP'], all_database_molecules['fr_halogen'], 
                       c=all_database_molecules['RefrigerantDistance'], cmap='viridis_r',
                       s=30, alpha=0.3, label='All Database Molecules')
    
    # Get ONLY the database molecules (non-refrigerants) from the top molecules
    # These are the ones saved to top_n_molecules.csv file
    top_database_molecules = top_molecules[top_molecules['molecule_type'] == 'candidate']
    num_top_candidates = len(top_database_molecules)
    
    # Highlight the top database molecules
    plt.scatter(top_database_molecules['MolLogP'], top_database_molecules['fr_halogen'], 
                color='gold', s=80, alpha=0.8, edgecolor='black', 
                label=f'Top {num_top_candidates} Database Molecules')
    
    # Mark ALL known refrigerants with stars
    plt.scatter(refrigerants['MolLogP'], refrigerants['fr_halogen'], 
                color='red', s=120, marker='*', alpha=0.8, label='Known Refrigerants')
    
    # Mark refrigerant mean
    plt.scatter([refrigerant_means.get('MolLogP', 0)], 
                [refrigerant_means.get('fr_halogen', 0)], 
                color='blue', s=200, marker='X', label='Refrigerant Mean')
    
    plt.colorbar(scatter, label='Distance to Refrigerant Profile')
    plt.title('MolLogP vs Halogen Count - All Database Molecules with Top Candidates Highlighted')
    plt.xlabel('MolLogP')
    plt.ylabel('Halogen Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/property_scatter.png', dpi=300)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Find potential refrigerant candidates by comparing with known refrigerants.')
    parser.add_argument('--database_file', type=str, default='molecular_formulas.csv',
                        help='Path to the molecular database file')
    parser.add_argument('--refrigerants_file', type=str, default='refrigerant_finding/refrigerants.csv',
                        help='Path to the known refrigerants file')
    parser.add_argument('--min_atoms', type=int, default=5,
                        help='Minimum number of heavy atoms (default: 5)')
    parser.add_argument('--max_atoms', type=int, default=10,
                        help='Maximum number of heavy atoms (default: 10)')
    parser.add_argument('--n_candidates', type=int, default=100,
                        help='Number of top candidates to return (default: 100)')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--use_calculated_means', action='store_true',
                        help='Use means calculated from refrigerants.csv instead of predefined means from REFRIGERANT_PROPERTIES.md')
    parser.add_argument('--output_dir', type=str, default='refrigerant_finding/candidates',
                        help='Directory to save output files (default: refrigerant_finding/candidates)')
    parser.add_argument('--properties_doc', type=str, default='refrigerant_finding/REFRIGERANT_PROPERTIES.md',
                        help='Path to the refrigerant properties documentation file')
    parser.add_argument('--unique', action='store_true',
                        help='Keep only the first occurrence of each unique SMILES in the database')
    
    args = parser.parse_args()
    
    # Set output directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output will be saved to: {OUTPUT_DIR}")
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load refrigerants (always load, regardless of which mean method we're using)
    logger.info("Loading refrigerants...")
    refrigerants_df = pd.read_csv(args.refrigerants_file)

    # Remove duplicates from refrigerants before processing
    logger.info("Removing duplicate refrigerants...")
    refrigerants_df = refrigerants_df.drop_duplicates(subset=['SMILES'])
    logger.info(f"After removing duplicates: {len(refrigerants_df)} unique refrigerants")

    # Process refrigerants to get their properties
    logger.info("Computing properties for refrigerants...")
    refrigerants_properties = compute_properties(
        refrigerants_df['SMILES'].tolist(), 
        category_name='refrigerant'
    )

    # Get refrigerant means - either from documentation or calculated from data
    if args.use_calculated_means:
        logger.info("Using means calculated from refrigerants.csv...")
        
        # Calculate mean values for refrigerant properties
        refrigerant_means = {}
        for prop in KEY_PROPERTIES:
            if prop in refrigerants_properties.columns:
                refrigerant_means[prop] = refrigerants_properties[prop].mean()
        
        logger.info("Calculated refrigerant mean properties:")
    else:
        logger.info("Using predefined means from REFRIGERANT_PROPERTIES.md...")
        # Extract means from the documentation file
        refrigerant_means = extract_means_from_md_file(args.properties_doc)
        logger.info("Predefined refrigerant mean properties:")
    
    for prop, value in refrigerant_means.items():
        logger.info(f"  {prop}: {value:.4f}")
    
    # Save refrigerant means for reference
    pd.DataFrame([refrigerant_means]).to_csv(f"{OUTPUT_DIR}/refrigerant_mean_properties.csv", index=False)
    
    # Load molecular database
    logger.info("Loading molecular database...")
    try:
        molecules_df = pd.read_csv(args.database_file)
    except Exception as e:
        logger.error(f"Error loading database file: {e}")
        return
    
    # Optionally deduplicate SMILES in the database
    if args.unique:
        original_count = len(molecules_df)
        molecules_df = molecules_df.drop_duplicates(subset=['smiles'])
        logger.info(f"Keeping only unique SMILES in database: {original_count} → {len(molecules_df)} molecules")
    
    # Filter database molecules
    logger.info(f"Filtering database molecules with {args.min_atoms}-{args.max_atoms} atoms...")
    
    # Get molecules with valid SMILES
    molecules_with_smiles = molecules_df[molecules_df['smiles'].notna()]
    
    # Compute properties for all molecules to get atom counts
    logger.info("Computing atom counts for database molecules...")
    all_mols = []
    valid_smiles = []
    
    for idx, smiles in enumerate(tqdm(molecules_with_smiles['smiles'], desc="Processing SMILES")):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                heavy_atom_count = Chem.Descriptors.HeavyAtomCount(mol)
                if args.min_atoms <= heavy_atom_count <= args.max_atoms:
                    all_mols.append(mol)
                    valid_smiles.append(smiles)
        except Exception as e:
            continue
    
    logger.info(f"Found {len(valid_smiles)} molecules with {args.min_atoms}-{args.max_atoms} heavy atoms")
    
    # Compute properties for filtered molecules
    database_properties = compute_properties(valid_smiles, category_name='database')
    
    # Normalize properties for better comparison
    properties_to_normalize = [p for p in KEY_PROPERTIES if p in database_properties.columns]
    logger.info(f"Normalizing {len(properties_to_normalize)} properties for comparison")
    
    # Combine database and refrigerant properties for normalization and comparison
    combined_properties = pd.concat([refrigerants_properties, database_properties], ignore_index=True)
    normalized_df = normalize_properties(combined_properties, properties_to_normalize)
    
    # Split back
    norm_refrigerants = normalized_df[normalized_df['category'] == 'refrigerant']
    norm_database = normalized_df[normalized_df['category'] == 'database']
    
    if args.use_calculated_means:
        # Calculate normalized means for refrigerants
        norm_refrigerant_means = {}
        for prop in properties_to_normalize:
            norm_refrigerant_means[prop] = norm_refrigerants[prop].mean()
    else:
        # Normalize the predefined means using the same scale factors
        norm_refrigerant_means = {}
        for prop in properties_to_normalize:
            if prop in refrigerant_means:
                min_val = combined_properties[prop].min()
                max_val = combined_properties[prop].max()
                if max_val > min_val:
                    norm_refrigerant_means[prop] = (refrigerant_means[prop] - min_val) / (max_val - min_val)
                else:
                    norm_refrigerant_means[prop] = 0
    
    # Calculate similarity to refrigerant profile for database molecules
    logger.info("Calculating similarity to refrigerant profile for database molecules...")
    database_with_similarity = calculate_refrigerant_similarity(
        norm_database, 
        norm_refrigerant_means,
        properties_to_normalize
    )
    
    # Calculate similarity to refrigerant profile for existing refrigerants
    logger.info("Calculating similarity to refrigerant profile for existing refrigerants...")
    refrigerants_with_similarity = calculate_refrigerant_similarity(
        norm_refrigerants, 
        norm_refrigerant_means,
        properties_to_normalize
    )
    
    # Mark the type of molecule
    database_with_similarity['molecule_type'] = 'candidate'
    refrigerants_with_similarity['molecule_type'] = 'refrigerant'
    
    # Create a set of known refrigerant SMILES for quick lookups
    refrigerant_smiles_set = set(refrigerants_with_similarity['smiles'])

    # Remove any molecules from database that are already in refrigerants
    database_with_similarity = database_with_similarity[~database_with_similarity['smiles'].isin(refrigerant_smiles_set)]

    # Combine database and refrigerant results
    all_molecules_with_similarity = pd.concat([database_with_similarity, refrigerants_with_similarity], ignore_index=True)
    
    # Rank by similarity (lower distance = more similar)
    ranked_molecules = all_molecules_with_similarity.sort_values('RefrigerantDistance')
    
    # Add back the original unnormalized property values from both datasets
    database_properties['molecule_type'] = 'candidate'
    refrigerants_properties['molecule_type'] = 'refrigerant'

    # Ensure database properties don't contain refrigerants
    database_properties = database_properties[~database_properties['smiles'].isin(refrigerant_smiles_set)]

    original_properties = pd.concat([database_properties, refrigerants_properties], ignore_index=True)

    # Make sure original_properties has unique SMILES strings
    original_properties = original_properties.drop_duplicates(subset=['smiles'])
    
    ranked_molecules_with_orig = ranked_molecules.merge(
        original_properties[['smiles', 'molecule_type'] + properties_to_normalize], 
        on=['smiles', 'molecule_type'],
        suffixes=('_normalized', '')
    )

    # Save top candidates
    top_n = min(args.n_candidates, len(ranked_molecules_with_orig))
    top_molecules = ranked_molecules_with_orig.head(top_n)
    
    top_molecules.to_csv(f"{OUTPUT_DIR}/top_{top_n}_molecules.csv", index=False)
    logger.info(f"Saved top {top_n} molecules to {OUTPUT_DIR}/top_{top_n}_molecules.csv")
    
    # Save all candidates with scores
    ranked_molecules_with_orig.to_csv(f"{OUTPUT_DIR}/all_molecules_ranked.csv", index=False)
    logger.info(f"Saved all {len(ranked_molecules_with_orig)} ranked molecules to {OUTPUT_DIR}/all_molecules_ranked.csv")
    
    # Create visualizations using unnormalized values
    logger.info("Creating visualizations...")
    plot_top_candidates(ranked_molecules_with_orig, refrigerant_means, norm_refrigerant_means, OUTPUT_DIR, top_molecules)
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 