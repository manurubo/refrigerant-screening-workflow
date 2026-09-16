#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import argparse
from rdkit import Chem

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract properties of molecules in the canonicalized grid visualization.')
    parser.add_argument('--csv_file', type=str, default='refrigerant_finding/candidates_type_i_training_db/top_100_molecules.csv',
                        help='Path to the CSV file containing molecules (default: refrigerant_finding/candidates_type_i_training_db/top_100_molecules.csv)')
    parser.add_argument('--output_dir', type=str, default='refrigerant_finding/candidates_type_i_training_db',
                        help='Directory to save output files (default: refrigerant_finding/candidates_type_i_training_db)')
    parser.add_argument('--output_file', type=str, default='canonicalized_molecules_properties.csv',
                        help='Name of the output CSV file (default: canonicalized_molecules_properties.csv)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output will be saved to: {args.output_dir}")

    # Read the CSV file
    df = pd.read_csv(args.csv_file)
    
    # Extract SMILES strings from the DataFrame
    smiles_list = df['smiles'].tolist()

    # Convert SMILES to RDKit molecules
    molecules = []
    valid_indices = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                molecules.append(mol)
                valid_indices.append(i)
            else:
                print(f"Warning: Could not parse SMILES {i}: {smiles}")
        except Exception as e:
            print(f"Error processing SMILES {i}: {smiles} - {str(e)}")

    # Ensure we have molecules to display
    if not molecules:
        print("Error: No valid molecules found")
        exit(1)

    # Get canonicalized SMILES for each molecule
    canon_smiles_list = []
    canon_mols = []
    canon_indices = []
    refrigerant_canon_smiles = set()

    # First, identify which canonicalized SMILES correspond to refrigerants
    for i, (mol, idx) in enumerate(zip(molecules, valid_indices)):
        # Canonicalize the SMILES
        canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        
        # Check if it's a refrigerant
        is_ref = False
        if 'category' in df.columns and df.iloc[idx]['category'] == 'refrigerant':
            is_ref = True
        if not is_ref and 'molecule_type' in df.columns and df.iloc[idx]['molecule_type'] == 'refrigerant':
            is_ref = True
        
        # If it's a refrigerant, add it to the set of refrigerant canonical SMILES
        if is_ref:
            refrigerant_canon_smiles.add(canon_smiles)
        
        # Store the canonicalized information
        canon_smiles_list.append(canon_smiles)
        canon_mols.append(mol)
        canon_indices.append(idx)

    # Now filter out molecules that match a refrigerant's canonical SMILES but aren't refrigerants
    filtered_mols = []
    filtered_indices = []
    seen_smiles = set()  # To avoid any duplicates in general

    for i, (mol, idx, canon_smiles) in enumerate(zip(canon_mols, canon_indices, canon_smiles_list)):
        # Check if it's a refrigerant
        is_ref = False
        if 'category' in df.columns and df.iloc[idx]['category'] == 'refrigerant':
            is_ref = True
        if not is_ref and 'molecule_type' in df.columns and df.iloc[idx]['molecule_type'] == 'refrigerant':
            is_ref = True
        
        # If it's a refrigerant, always include it
        # If it's not a refrigerant, only include if its canonicalized SMILES isn't in the refrigerant set
        # Also check if we've already seen this SMILES to avoid duplicates
        if (is_ref or canon_smiles not in refrigerant_canon_smiles) and canon_smiles not in seen_smiles:
            filtered_mols.append(mol)
            filtered_indices.append(idx)
            seen_smiles.add(canon_smiles)

    # Extract properties for the filtered molecules
    if filtered_indices:
        # Get the list of non-normalized property columns
        normalized_columns = [col for col in df.columns if '_normalized' in col]
        non_normalized_columns = [col for col in df.columns if col not in normalized_columns and col not in ['smiles', 'category', 'MolName', 'molecule_type']]
        
        # Include only the columns requested
        desired_columns = ['MolName', 'category', 'smiles'] + non_normalized_columns
        
        # Create a new dataframe with only the desired rows and columns
        filtered_df = df.iloc[filtered_indices][desired_columns].copy()
        
        # Add molecule_type where category is missing
        if 'molecule_type' in df.columns:
            for i, idx in enumerate(filtered_indices):
                if pd.isna(filtered_df.iloc[i]['category']) and not pd.isna(df.iloc[idx]['molecule_type']):
                    filtered_df.loc[filtered_df.index[i], 'category'] = df.iloc[idx]['molecule_type']
        
        # Save to CSV
        output_path = os.path.join(args.output_dir, args.output_file)
        filtered_df.to_csv(output_path, index=False)
        
        print(f"Properties for {len(filtered_indices)} canonicalized molecules saved to {output_path}")
        print(f"Columns included: {', '.join(desired_columns)}")
    else:
        print("No molecules remained after filtering!")

if __name__ == "__main__":
    main() 