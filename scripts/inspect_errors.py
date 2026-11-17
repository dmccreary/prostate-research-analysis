import sys
import pandas as pd

if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('results/stage9_1/arm3_error_analysis.csv')

print('================================================================================')
print('STAGE 9.1 ARM 3: DETAILED FALSE POSITIVES (N = 13)')
print('================================================================================')
fps = df[df['error_type'] == 'False Positive'].reset_index(drop=True)
for i, r in fps.iterrows():
    print(f"[{i+1}] PMID: {r['pmid']} | Predicted Probability: {r['predicted_prob']:.4f} (Ground Truth: 0)")
    print(f"    Title: {r['title']}")
    print(f"    Top Positive Decision Drivers: {r['top_drivers']}")
    print(f"    Abstract Snippet: {r['abstract_snippet']}")
    print('-' * 80)

print('\n================================================================================')
print('STAGE 9.1 ARM 3: DETAILED FALSE NEGATIVES (N = 8)')
print('================================================================================')
fns = df[df['error_type'] == 'False Negative'].reset_index(drop=True)
for i, r in fns.iterrows():
    print(f"[{i+1}] PMID: {r['pmid']} | Predicted Probability: {r['predicted_prob']:.4f} (Ground Truth: 1)")
    print(f"    Title: {r['title']}")
    print(f"    Active Decision Drivers: {r['top_drivers']}")
    print(f"    Abstract Snippet: {r['abstract_snippet']}")
    print('-' * 80)
