import pandas as pd
import re

# Load citation pairs (assuming they're saved from the notebook)
df_pairs = pd.read_csv('drug_docs_with_refs.csv')

# Create citation pairs like in the notebook
citation_pairs = []
for _, row in df_pairs.iterrows():
    if pd.notna(row['references']) and row['references']:
        ref_dois = row['references'].split()
        for ref_doi in ref_dois:
            hit_row = df_pairs[df_pairs['doi'] == ref_doi]
            if not hit_row.empty:
                citation_pairs.append({
                    'hit_molregno': hit_row.iloc[0]['molregno'],
                    'hit_paper': ref_doi,
                    'hit_title': hit_row.iloc[0]['title'],
                    'drug_paper': row['doi'],
                    'drug_title': row['title']
                })

pairs_df = pd.DataFrame(citation_pairs)
print(f"Found {len(pairs_df)} citation pairs")

# Load patterns
with open('hit_synonyms.txt', 'r') as f:
    patterns = [line.strip() for line in f if line.strip()]

# Count matches for each pattern
pattern_counts = {}
for i, pattern in enumerate(patterns, 1):
    count = 0
    for title in pairs_df['hit_title']:
        if pd.notna(title) and re.search(pattern, str(title), re.IGNORECASE):
            count += 1
    pattern_counts[f"Pattern {i}: {pattern}"] = count

# Display results
for pattern, count in pattern_counts.items():
    print(f"{pattern}: {count}")