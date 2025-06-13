# Score Prompt Version 2

!!! prompt
    Please create Prostate Cancer Treatment Study Paper analysis program in Python.

    The program will analyze one row at a time from an input CSV file.  The sample file you will use is output100.csv in the project knowledge area.
    
    Each row corresponds to a specific paper published in Medline and has keywords to indicate the subject of the paper concerns prostate cancer treatment.  There will be around 10,000 papers each year to analyze. 

    I would like you to create a Python program that does a series of filters on these papers.
    
    ## FILTER ONE
    The first filter must check that one of the following keywords is included in the abstract:

    1. "prostate cancer"
    2. "prostate neoplasm"
    3. "prostate neoplasia"

    If one of these keywords appears then mark a new column called "prostate_keyword" "TRUE"
    If none of these keywords appears then mark a new column "prostate_keyword" "FALSE"

    ## FILTER TWO
    This filter is to discard any papers where we certain that the number of high-risk patients is under 50 OR the number of intermediate risk papers are less than 100.  This will be stored in a new column of the spreadsheet called "sample_size_indicator" which is "TRUE" if the size large enough, and "FALSE" if the size is too small. If the sample size is unknown, use a string of "UNKNOWN".

    Only pass sample_size_indicator values of TRUE and UNKNOWN to the next level two filter.

    ## FILTER THREE
    This filter criteria is that the median followup time is five years or greater in a new column "followup_indicator".
    If the median followup time is five years or greater then place "TRUE" in that column.
    If the median followup time is under or less than five years place a "FALSE" in that column.
    If the followup time is unknown, use a string of "UNKNOWN" in that column.

    ## FILTER FOUR
    This filter should remove all papers that have pathologic staging without clinical staging.  The new column name will be "pathologic_staging_filter".
    Any papers that have pathologic staging without clinical staging should be marked "FALSE".
    Papers that contain the keywords "clinical staging" should be marked "TRUE".
    If no indication of pathologic staging or clinical staging then the column should be marked "UNKNOWN".

    ## KEYWORD QUALITY SCORE
    Next, we would like you to use the summary-criteria.md which contains rules to create a score for each line in the output100.csv file.  The score should range on a scale of 0 to 100 and be placed in a new column called "keyword_score".
    
    Please use these score weights when calculating the keyword score:

    ```python
    # Positive quality indicators (based on summary criteria)
        self.positive_terms = {
            'randomized': 15,
            'prospective': 10,
            'multicenter': 8,
            'long-term': 5,
            'outcomes': 5,
            'survival': 8,
            'biochemical recurrence': 8,
            'disease-free survival': 10,
            'overall survival': 10,
            'cancer-specific survival': 8,
            'metastasis-free survival': 8,
            'radical prostatectomy': 5,
            'radiation therapy': 5,
            'brachytherapy': 5,
            'clinical staging': 8,
            'd\'amico': 10,
            'nccn': 10,
            'risk stratification': 8,
            'intermediate risk': 5,
            'high risk': 5,
            'low risk': 5
        }

    # Negative quality indicators (rejection criteria)
    negative_terms = {
        'abstract only': -50,
        'conference abstract': -50,
        'meeting abstract': -50,
        'case report': -10,
        'case series': -8,
        'review': 0,
        'editorial': -50,
        'commentary': -50,
        'letter': -50,
        'pathologic staging only': -30,
        'no stratification': -6
    }
    ```

    A score of 0 means that none of the criteria match.  
    A score of 100 implies that all of the summary-criteria are matching.
    Work hard to make sure that there is a good distribution of scores for the papers.

    Please have the Python program create and populate the following new columns to 
    the CSV with the following column name:
    
    1. prostate_keyword
    2. sample_size_indicator
    3. followup_indicator
    4. pathologic_staging_filter
    5. keyword_score

    Generate the code and then test the code on the output500.csv file to make sure 
    that the keyword_score values are evenly dstributed in the range 0 to 100.
