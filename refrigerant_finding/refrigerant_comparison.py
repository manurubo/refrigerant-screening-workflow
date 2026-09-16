import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tableone import TableOne
from tqdm import tqdm
import logging
from scipy.stats import entropy
import argparse
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directories
OUTPUT_DIR = "refrigerant_finding/statistical_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Font settings for plots
plt.rcParams['font.family'] = 'Nimbus Sans'

# Get all available RDKit descriptors
ALL_DESCRIPTORS = [desc[0] for desc in Descriptors.descList]
logger.info(f"Total number of RDKit descriptors: {len(ALL_DESCRIPTORS)}")

def jensen_shannon_distance(distribution1, distribution2):
    """Calculate Jensen-Shannon distance between two distributions."""
    # Ensure the distributions sum to 1
    distribution1 = np.array(distribution1)
    distribution2 = np.array(distribution2)
    
    # Handle empty distributions
    if np.sum(distribution1) == 0 or np.sum(distribution2) == 0:
        return 1.0  # Maximum distance
    
    distribution1 = distribution1 / np.sum(distribution1)
    distribution2 = distribution2 / np.sum(distribution2)
    
    # Calculate the average distribution
    avg_distribution = (distribution1 + distribution2) / 2
    
    # Calculate the Jensen-Shannon divergence
    js_divergence = (entropy(distribution1, avg_distribution) + entropy(distribution2, avg_distribution)) / 2
    
    # The square root of the divergence is the Jensen-Shannon distance
    js_distance = np.sqrt(js_divergence)
    return js_distance

def calculate_descriptors(mol):
    """Calculate all available RDKit descriptors for a molecule."""
    descriptor_values = {}
    
    if mol is None:
        return {desc: np.nan for desc in ALL_DESCRIPTORS}
    
    for desc_name, desc_function in Descriptors.descList:
        try:
            value = desc_function(mol)
            descriptor_values[desc_name] = value
        except Exception as e:
            # Capture more specific information about which descriptor failed
            error_msg = str(e)
            if "explicit valence" in error_msg.lower():
                # This is a valence error, get more details about the molecule
                try:
                    smiles = Chem.MolToSmiles(mol)
                    logger.error(f"Explicit valence error for descriptor '{desc_name}' on molecule {smiles}: {error_msg}")
                except:
                    logger.error(f"Explicit valence error for descriptor '{desc_name}': {error_msg}")
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
    
    if problematic_smiles:
        logger.error(f"Found {len(problematic_smiles)} problematic SMILES strings:")
        for idx, smiles, error in problematic_smiles:
            logger.error(f"  Index {idx}: '{smiles}' - {error}")
    
    logger.info(f"Processed {len(smiles_list)} SMILES strings, {valid_count} valid molecules")
    return molecules, problematic_smiles

def compute_all_descriptors(smiles_list, category_name):
    """Compute all descriptors for a list of SMILES strings."""
    mols, problematic_smiles = get_mols_from_smiles(smiles_list)
    
    results = []
    for i, mol in enumerate(tqdm(mols, desc=f"Computing descriptors for {category_name}")):
        if mol is not None:
            try:
                desc_values = calculate_descriptors(mol)
                desc_values['smiles'] = smiles_list[i]
                desc_values['category'] = category_name
                # Add heavy atom count specifically for filtering
                desc_values['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
                results.append(desc_values)
            except Exception as e:
                logger.error(f"Error calculating descriptors for molecule at index {i} (SMILES: {smiles_list[i]}): {str(e)}")
    
    if problematic_smiles:
        # Write problematic SMILES to a file for reference
        problem_file = f"{OUTPUT_DIR}/problematic_smiles_{category_name}.csv"
        pd.DataFrame(problematic_smiles, columns=['Index', 'SMILES', 'Error']).to_csv(problem_file, index=False)
        logger.info(f"Saved list of problematic SMILES to {problem_file}")
    
    return pd.DataFrame(results)

def get_size_filtered_molecules(database_mols_df, refrigerant_mols_df, n_samples, buffer_factor=1.2):
    """
    Filter database molecules to match the size range of refrigerants.
    
    Args:
        database_mols_df: DataFrame with database molecules
        refrigerant_mols_df: DataFrame with refrigerant molecules
        n_samples: Number of molecules to sample
        buffer_factor: Factor to expand size range by (e.g., 1.2 = 20% larger range)
        
    Returns:
        DataFrame with filtered database molecules
    """
    # Get size range of refrigerants
    min_size = refrigerant_mols_df['HeavyAtomCount'].min()
    max_size = refrigerant_mols_df['HeavyAtomCount'].max()
    
    # Apply buffer to range
    adjusted_min_size = max(1, int(min_size / buffer_factor))
    adjusted_max_size = int(max_size * buffer_factor)
    
    logger.info(f"Refrigerant size range: {min_size}-{max_size} heavy atoms")
    logger.info(f"Adjusted size range for filtering: {adjusted_min_size}-{adjusted_max_size} heavy atoms")
    
    # Filter database molecules by size
    filtered_df = database_mols_df[
        (database_mols_df['HeavyAtomCount'] >= adjusted_min_size) & 
        (database_mols_df['HeavyAtomCount'] <= adjusted_max_size)
    ]
    
    logger.info(f"Found {len(filtered_df)} molecules in database within size range")
    
    # Sample from filtered molecules
    if len(filtered_df) < n_samples:
        logger.warning(f"Only {len(filtered_df)} molecules available in size range, using all of them")
        return filtered_df
    else:
        return filtered_df.sample(n=n_samples)

def calculate_js_distances(df, categories, features):
    """Calculate Jensen-Shannon distances between categories for each feature."""
    results = []
    
    for feature in tqdm(features, desc="Calculating JS distances"):
        # Skip features where all values are the same
        if df[feature].nunique() <= 1:
            continue
            
        for cat1 in categories:
            for cat2 in categories:
                if cat1 >= cat2:  # Only calculate for unique pairs
                    continue
                    
                data1 = df[df['category'] == cat1][feature].dropna()
                data2 = df[df['category'] == cat2][feature].dropna()
                
                if len(data1) < 2 or len(data2) < 2:
                    continue
                    
                # For categorical features
                if df[feature].nunique() <= 20:
                    # Get unique values
                    unique_values = df[feature].dropna().unique()
                    
                    # Calculate probabilities
                    probs1 = []
                    probs2 = []
                    for val in unique_values:
                        count1 = (data1 == val).sum()
                        count2 = (data2 == val).sum()
                        probs1.append(count1 / len(data1))
                        probs2.append(count2 / len(data2))
                        
                    js_dist = jensen_shannon_distance(probs1, probs2)
                else:
                    # For continuous features, use histograms
                    hist1, edges = np.histogram(data1, bins=30, density=True)
                    hist2, _ = np.histogram(data2, bins=edges, density=True)
                    
                    js_dist = jensen_shannon_distance(hist1, hist2)
                
                results.append({
                    'Feature': feature,
                    'Category1': cat1,
                    'Category2': cat2,
                    'JS_Distance': js_dist
                })
    
    js_distances = pd.DataFrame(results)
    return js_distances

def plot_feature_distribution(df, feature, output_dir, js_distance=None):
    """Plot the distribution of a feature for each category."""
    # Determine if the feature is discrete or continuous
    is_discrete = df[feature].nunique() <= 20
    
    categories = df['category'].unique()
    colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a'][:len(categories)]
    
    if is_discrete:
        # Discrete feature - use countplot
        plt.figure(figsize=(12, 8))
        ax = sns.countplot(data=df, x=feature, hue='category', palette=colors)
        plt.xticks(rotation=45)
        plt.title(f'Distribution of {feature}')
        
        # Add counts on top of bars
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                       (p.get_x() + p.get_width()/2., p.get_height()), 
                       ha='center', va='bottom', fontsize=8)
    else:
        # Continuous feature - use KDE
        plt.figure(figsize=(12, 8))
        
        # For each category, create a KDE plot
        for i, category in enumerate(categories):
            category_data = df[df['category'] == category][feature].dropna()
            if len(category_data) >= 2:  # Need at least 2 points for KDE
                sns.kdeplot(data=category_data, label=category, color=colors[i])
            else:
                logger.info(f"Not enough data points for KDE plot of {feature} in category {category}")
                
        plt.title(f'Distribution of {feature}')
        plt.legend()
    
    # Add JS distance in the plot if provided
    if js_distance is not None:
        plt.text(0.95, 0.95, f'JS Distance: {js_distance:.4f}',
                transform=plt.gca().transAxes, 
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{feature}.png', dpi=300)
    plt.close()

def plot_size_distributions(refrigerants_df, database_df, filtered_df, output_dir):
    """
    Create plots comparing the molecule size distributions of refrigerants, 
    the full database sample, and the size-filtered sample.
    Creates both density and count plots.
    """
    # Get overall min and max for consistent binning
    all_data = pd.concat([
        refrigerants_df['HeavyAtomCount'], 
        filtered_df['HeavyAtomCount']
    ])
    if database_df is not None:
        all_data = pd.concat([all_data, database_df['HeavyAtomCount']])
    
    min_atoms = int(all_data.min())
    max_atoms = int(all_data.max())
    
    # Create consistent bins with integer boundaries
    bins = np.arange(min_atoms, max_atoms + 2) - 0.5  # +2 because we want inclusive upper bound and -0.5 for bin centering
    
    # 1. Create density plot
    plt.figure(figsize=(12, 8))
    sns.histplot(refrigerants_df['HeavyAtomCount'], 
                 label='Refrigerants', 
                 color='#1b9e77',
                 kde=True, 
                 stat='density',
                 bins=bins)
    
    if database_df is not None:
        sns.histplot(database_df['HeavyAtomCount'], 
                     label='Database (Random Sample)', 
                     color='#d95f02',
                     kde=True, 
                     stat='density',
                     bins=bins)
    
    sns.histplot(filtered_df['HeavyAtomCount'], 
                 label='Size-Matched Database', 
                 color='#7570b3',
                 kde=True, 
                 stat='density',
                 bins=bins)
    
    plt.title('Distribution of Molecule Sizes (Heavy Atom Count) - Density')
    plt.xlabel('Heavy Atom Count')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/molecule_size_comparison_density.png', dpi=300)
    plt.close()
    
    # 2. Create count plot
    plt.figure(figsize=(12, 8))
    sns.histplot(refrigerants_df['HeavyAtomCount'], 
                 label='Refrigerants', 
                 color='#1b9e77',
                 kde=False,  # No KDE on count plot
                 stat='count',
                 bins=bins)
    
    if database_df is not None:
        sns.histplot(database_df['HeavyAtomCount'], 
                     label='Database (Random Sample)', 
                     color='#d95f02',
                     kde=False,
                     stat='count',
                     bins=bins)
    
    sns.histplot(filtered_df['HeavyAtomCount'], 
                 label='Size-Matched Database', 
                 color='#7570b3',
                 kde=False,
                 stat='count',
                 bins=bins)
    
    plt.title('Distribution of Molecule Sizes (Heavy Atom Count) - Count')
    plt.xlabel('Heavy Atom Count')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/molecule_size_comparison_count.png', dpi=300)
    plt.close()
    
    # 3. Save the original combined plot for backward compatibility
    plt.figure(figsize=(12, 8))
    sns.histplot(refrigerants_df['HeavyAtomCount'], 
                 label='Refrigerants', 
                 color='#1b9e77',
                 kde=True, 
                 stat='density',
                 bins=bins)
    
    if database_df is not None:
        sns.histplot(database_df['HeavyAtomCount'], 
                     label='Database (Random Sample)', 
                     color='#d95f02',
                     kde=True, 
                     stat='density',
                     bins=bins)
    
    sns.histplot(filtered_df['HeavyAtomCount'], 
                 label='Size-Matched Database', 
                 color='#7570b3',
                 kde=True, 
                 stat='density',
                 bins=bins)
    
    plt.title('Distribution of Molecule Sizes (Heavy Atom Count)')
    plt.xlabel('Heavy Atom Count')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/molecule_size_comparison.png', dpi=300)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compare refrigerants with molecular database.')
    parser.add_argument('--n_samples', type=int, default=10000, 
                        help='Number of molecules to sample from the database')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--size_match', action='store_true',
                        help='Whether to match database molecules by size')
    parser.add_argument('--compare_all', action='store_true',
                        help='Whether to compare with both size-matched and full database')
    parser.add_argument('--size_buffer', type=float, default=1.0,
                        help='Buffer factor for size matching (default: 1.0 = no buffer)')
    parser.add_argument('--atom_range', nargs=2, type=int, metavar=('MIN_ATOMS', 'MAX_ATOMS'),
                        help='Restrict analysis to molecules with specified heavy atom count range (e.g., --atom_range 5 10)')
    parser.add_argument('--database_file', type=str, default='molecular_formulas.csv',
                        help='Path to the CoCoGraph training/reference SMILES database (default: molecular_formulas.csv)')
    parser.add_argument('--refrigerants_file', type=str, default='refrigerant_finding/refrigerants.csv',
                        help='Path to the known working-fluids file')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.random_seed)
    
    # Load refrigerants
    logger.info("Loading refrigerants...")
    refrigerants_df = pd.read_csv(args.refrigerants_file)
    
    # Load molecular database
    logger.info("Loading molecular database...")
    molecules_df = pd.read_csv(args.database_file)
    
    # Process refrigerants
    logger.info("Computing descriptors for refrigerants...")
    refrigerants_descriptors = compute_all_descriptors(
        refrigerants_df['SMILES'].tolist(), 
        category_name='refrigerant'
    )
    
    # Apply atom range filter if specified
    if args.atom_range is not None:
        min_atoms, max_atoms = args.atom_range
        logger.info(f"Applying heavy atom range filter: {min_atoms}-{max_atoms} atoms")
        
        # Filter refrigerants by atom range
        original_count = len(refrigerants_descriptors)
        refrigerants_descriptors = refrigerants_descriptors[
            (refrigerants_descriptors['HeavyAtomCount'] >= min_atoms) & 
            (refrigerants_descriptors['HeavyAtomCount'] <= max_atoms)
        ]
        filtered_count = len(refrigerants_descriptors)
        logger.info(f"Filtered refrigerants from {original_count} to {filtered_count} molecules within atom range")
        
        if filtered_count == 0:
            logger.error("No refrigerants remain after applying atom range filter. Exiting.")
            return
    
    # Initialize variables for database descriptors
    database_descriptors = None
    filtered_descriptors = None
    
    # Sample and process database molecules
    if args.compare_all or not args.size_match:
        logger.info(f"Computing descriptors for random sample of {args.n_samples} database molecules...")
        database_sample = molecules_df.sample(n=args.n_samples, random_state=args.random_seed)
        database_descriptors = compute_all_descriptors(
            database_sample['smiles'].tolist(), 
            category_name='database_full'
        )
        
        # Apply atom range filter to database molecules if specified
        if args.atom_range is not None:
            min_atoms, max_atoms = args.atom_range
            original_count = len(database_descriptors)
            database_descriptors = database_descriptors[
                (database_descriptors['HeavyAtomCount'] >= min_atoms) & 
                (database_descriptors['HeavyAtomCount'] <= max_atoms)
            ]
            filtered_count = len(database_descriptors)
            logger.info(f"Filtered full database from {original_count} to {filtered_count} molecules within atom range")
    
    # Always compute size-matched descriptors
    logger.info("Computing descriptors for size-matched database molecules...")
    if args.compare_all or args.size_match:
        # First, we need descriptors for all molecules to get size information
        if database_descriptors is None:
            logger.info("Sampling from database for size filtering...")
            # Use a larger sample to ensure we have enough molecules after filtering
            initial_sample_size = min(args.n_samples, len(molecules_df))
            initial_sample = molecules_df.sample(n=initial_sample_size, random_state=args.random_seed)
            temp_descriptors = compute_all_descriptors(
                initial_sample['smiles'].tolist(), 
                category_name='temp'
            )
            
            # Apply atom range filter if specified before size filtering
            if args.atom_range is not None:
                min_atoms, max_atoms = args.atom_range
                original_count = len(temp_descriptors)
                temp_descriptors = temp_descriptors[
                    (temp_descriptors['HeavyAtomCount'] >= min_atoms) & 
                    (temp_descriptors['HeavyAtomCount'] <= max_atoms)
                ]
                filtered_count = len(temp_descriptors)
                logger.info(f"Filtered temp database from {original_count} to {filtered_count} molecules within atom range")
            
            # Filter by size and sample
            filtered_molecules = get_size_filtered_molecules(
                temp_descriptors, 
                refrigerants_descriptors,
                args.n_samples,
                buffer_factor=args.size_buffer
            )
            filtered_descriptors = filtered_molecules.copy()
            filtered_descriptors['category'] = 'database_size_matched'
        else:
            # Apply atom range filter if specified before size filtering
            db_for_filtering = database_descriptors
            if args.atom_range is not None:
                min_atoms, max_atoms = args.atom_range
                original_count = len(db_for_filtering)
                db_for_filtering = db_for_filtering[
                    (db_for_filtering['HeavyAtomCount'] >= min_atoms) & 
                    (db_for_filtering['HeavyAtomCount'] <= max_atoms)
                ]
                filtered_count = len(db_for_filtering)
                logger.info(f"Filtered database for size matching from {original_count} to {filtered_count} molecules within atom range")
            
            # If we already calculated database descriptors, filter from those
            filtered_molecules = get_size_filtered_molecules(
                db_for_filtering, 
                refrigerants_descriptors,
                args.n_samples,
                buffer_factor=args.size_buffer
            )
            filtered_descriptors = filtered_molecules.copy()
            filtered_descriptors['category'] = 'database_size_matched'
    
    # Plot size distributions
    plot_size_distributions(
        refrigerants_descriptors,
        database_descriptors,
        filtered_descriptors,
        OUTPUT_DIR
    )
    
    # Combine datasets based on user options
    logger.info("Combining datasets...")
    if args.compare_all:
        # Compare with both database samples
        combined_df = pd.concat(
            [refrigerants_descriptors, database_descriptors, filtered_descriptors], 
            ignore_index=True
        )
        categories = ['refrigerant', 'database_full', 'database_size_matched']
    elif args.size_match:
        # Compare only with size-matched sample
        combined_df = pd.concat(
            [refrigerants_descriptors, filtered_descriptors], 
            ignore_index=True
        )
        categories = ['refrigerant', 'database_size_matched']
    else:
        # Compare only with full random sample
        combined_df = pd.concat(
            [refrigerants_descriptors, database_descriptors], 
            ignore_index=True
        )
        categories = ['refrigerant', 'database_full']
    
    # Save the combined dataset
    combined_df.to_csv(f"{OUTPUT_DIR}/combined_descriptors.csv", index=False)
    logger.info(f"Saved combined descriptors to {OUTPUT_DIR}/combined_descriptors.csv")
    
    # Get available descriptors (columns that exist and have data)
    available_descriptors = [
        col for col in ALL_DESCRIPTORS 
        if col in combined_df.columns and combined_df[col].notna().sum() > 0
    ]
    logger.info(f"Found {len(available_descriptors)} available descriptors with data")
    
    # Move important descriptors to the beginning for analysis
    key_descriptors = ['HeavyAtomCount', 'MolWt', 'MolLogP', 'TPSA', 'NumHAcceptors', 
                      'NumHDonors', 'NumRotatableBonds', 'NumAromaticRings']
    
    # Reorder descriptors list to prioritize key descriptors
    for desc in reversed(key_descriptors):
        if desc in available_descriptors:
            available_descriptors.remove(desc)
            available_descriptors.insert(0, desc)
    
    # Calculate Jensen-Shannon distances
    logger.info("Calculating Jensen-Shannon distances...")
    js_distances = calculate_js_distances(
        combined_df, 
        categories=categories, 
        features=available_descriptors
    )
    
    # Save JS distances
    js_distances.to_csv(f"{OUTPUT_DIR}/js_distances.csv", index=False)
    logger.info(f"Saved JS distances to {OUTPUT_DIR}/js_distances.csv")
    
    # Create summary tables with JS distances for each comparison
    for category in categories:
        if category == 'refrigerant':
            continue
        
        # Filter JS distances for this comparison
        category_js = js_distances[
            ((js_distances['Category1'] == 'refrigerant') & (js_distances['Category2'] == category)) |
            ((js_distances['Category1'] == category) & (js_distances['Category2'] == 'refrigerant'))
        ].sort_values('JS_Distance', ascending=False)
        
        # Save ranked JS distances for this comparison
        category_js.to_csv(f"{OUTPUT_DIR}/js_distances_ranked_{category}.csv", index=False)
        logger.info(f"Saved ranked JS distances for refrigerant vs {category}")
    
    # Statistical comparison using TableOne
    logger.info("Generating statistical comparison tables...")
    # Group descriptors into smaller batches to avoid TableOne limitations
    batch_size = 50
    for i in range(0, len(available_descriptors), batch_size):
        batch_descriptors = available_descriptors[i:i+batch_size]
        if not batch_descriptors:
            continue
            
        mytable = TableOne(
            combined_df,
            columns=batch_descriptors,
            categorical=[],
            groupby='category',
            pval=True
        )
        
        batch_name = f"batch_{i//batch_size + 1}"
        mytable.to_csv(f"{OUTPUT_DIR}/tableone_{batch_name}.csv")
        logger.info(f"Saved statistical comparison table {batch_name}")
    
    # Generate individual plots for each descriptor
    logger.info("Generating individual plots for each descriptor...")
    for feature in tqdm(available_descriptors, desc="Generating plots"):
        # Get JS distances for this feature
        feature_js = js_distances[js_distances['Feature'] == feature]
        
        # Create a separate plot for each comparison
        for cat1, cat2 in [(cat1, cat2) for cat1 in categories for cat2 in categories if cat1 < cat2]:
            # Skip comparisons that don't involve refrigerants
            if 'refrigerant' not in [cat1, cat2]:
                continue
                
            # Filter data for this comparison
            comparison_df = combined_df[combined_df['category'].isin([cat1, cat2])]
            
            # Get JS distance for this feature and comparison
            js_row = feature_js[
                ((feature_js['Category1'] == cat1) & (feature_js['Category2'] == cat2)) |
                ((feature_js['Category1'] == cat2) & (feature_js['Category2'] == cat1))
            ]
            js_value = None if js_row.empty else js_row.iloc[0]['JS_Distance']
            
            # Create plot directory for this comparison
            comparison_dir = f"{OUTPUT_DIR}/plots_{cat1}_vs_{cat2}"
            os.makedirs(comparison_dir, exist_ok=True)
            
            # Create the plot
            plot_feature_distribution(
                comparison_df, 
                feature=feature, 
                output_dir=comparison_dir,
                js_distance=js_value
            )
    
    logger.info("Analysis complete!")

if __name__ == "__main__":
    main() 