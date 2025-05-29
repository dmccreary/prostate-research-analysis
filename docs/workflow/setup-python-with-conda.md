# Setup Python Environment with Conda

```sh
conda deactivate
conda create -n "doc-classifier" python=3
conda activate doc-classifier
pip install pandas tqdm Bio chardet
```

The prompt should now have the string "doc-classifier" in it.

```
(doc-classifier) prostate-research-analysis/src $
```

## Invoking the Python program

Copy this command to the command line to start processing the input data to get the abstracts.

```sh
python extractor3.py --input_file ../data/data.csv --output_file output.csv --email dan.mccreary@gmail.com
```

Sample working run of the 4th version:

``
$  python extractor4.py --input_file ../data/data.csv --output_file output.csv --email dan.mccreary@gmail.com
2025-05-29 14:13:51,874 - INFO - Starting PubMed abstract extraction
2025-05-29 14:13:51,874 - INFO - Input file: ../data/data.csv
2025-05-29 14:13:51,874 - INFO - Output file: output.csv
2025-05-29 14:13:51,874 - INFO - Batch size: 100
2025-05-29 14:13:51,874 - INFO - Max workers: 1
2025-05-29 14:13:51,874 - INFO - Specified encoding: Auto-detect
2025-05-29 14:13:51,914 - INFO - Detected encoding: utf-8 (confidence: 0.99)
2025-05-29 14:13:51,914 - INFO - Trying to read file with encoding: utf-8
2025-05-29 14:13:51,933 - INFO - Successfully read file with encoding: utf-8
2025-05-29 14:13:51,933 - INFO - Processing 7338 articles
Fetching abstracts: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 74/74 [01:15<00:00,  1.01s/it]
2025-05-29 14:15:34,578 - INFO - File saved with UTF-8 encoding
2025-05-29 14:15:34,578 - INFO - Successfully saved 7338 articles with abstracts to output.csv
2025-05-29 14:15:34,580 - INFO - Articles with abstracts: 7289/7338 (99.33%)
2025-05-29 14:15:34,581 - INFO - Extraction completed
```
