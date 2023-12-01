import pandas as pd
import cyvcf2
import os
from skfeature.function.information_theoretical_based import MRMR

def get_ad_associated_genes():
    # Replace with actual code to fetch or read the list of AD-associated genes
    return ['A2M', 'A2MP', 'ABCA1', 'ABCA12']
    #return ['']

def get_labels(filepaths):
    # Replace with actual code for labels
    return [1, 0, 1, 0, 0, 1, 1, 1, 0, 0]

def is_numeric(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def process_vcf(file_path):
    # Determine the line number where the data starts (i.e., the line after ## lines)
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith("#CHROM"):
                skip_rows = i
                break
    
    # Read VCF file into DataFrame from the data start line
    df = pd.read_csv(file_path, sep='\t', skiprows=skip_rows, header=0)
    
    # Get list of AD-associated genes
    ad_genes = get_ad_associated_genes()
    
    # Further filter to retain SNPs located on known AD-associated genes
    df_ad_genes = df[df['INFO'].str.contains('|'.join(ad_genes), na=False)]

    # Dynamically determine the genotype column
    genotype_col = next((col for col in df_ad_genes.columns if col.endswith('_MAXGT')), None)
    
    if genotype_col is None:
        raise ValueError("No genotype column found")
    
    # Extract numerical genotype information
    df_ad_genes['GENOTYPE_NUM'] = df_ad_genes[genotype_col].apply(lambda x: sum([int(allele) for allele in x.split(':')[0].split('/') if allele.isdigit()]))
    
    return df_ad_genes

    # Prepare data for mRMR
    X = df_ad_genes['GENOTYPE_NUM'].values.reshape(-1, 1)  # Each SNP is a sample, genotype value is the feature
    y = label  # Label for the individual

    # Perform mRMR feature selection to select 500 SNP features
    selected_feature_indices = MRMR.mrmr(X, y, n_selected_features=500)
    
    # Get selected features
    selected_features = df_ad_genes.iloc[selected_feature_indices]
    
    # Transpose the selected features DataFrame to get the desired shape (1, 500)
    selected_features_reshaped = selected_features.transpose()
    
    return selected_features_reshaped

if __name__ == "__main__":
    # Specify the directory containing your VCF files
    directory_path = '.'
    
    # Get a list of all VCF files in the specified directory
    vcf_files = [f for f in os.listdir(directory_path) if f.endswith('.vcf')]
    
    # Sort the list of files to ensure consistent order, then select the first three
    vcf_files = sorted(vcf_files)[:10]
    
    # Assume labels is a list of labels (1 for Alzheimer's, 0 for non-Alzheimer's) corresponding to the first three VCF files
    labels = get_labels(vcf_files)  # You will need to provide the correct labels here

    # Process each VCF file and stack the results into a matrix X
    X = pd.concat([process_vcf(os.path.join(directory_path, file_path)) for file_path in vcf_files])
    y = labels  # Labels for each individual
    
    # Perform mRMR feature selection to select 500 SNP features
    selected_feature_indices = MRMR.mrmr(X.values, y, n_selected_features=500)
    
    # Get selected features
    selected_features = X.iloc[:, selected_feature_indices]
    import pdb; pdb.set_trace()

    #Note: each file has some different columns. When they're combined, this means there will be lots of NaNs (but this is expected behavior)

    #TODO: add code to extract features from each row!


