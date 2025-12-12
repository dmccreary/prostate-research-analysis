# Year Distribution Chart

Interactive visualization comparing publication years between positive (accepted) and negative (rejected) paper datasets.

<div style="max-width: 800px; margin: 0 auto;">
    <canvas id="yearDistributionChart"></canvas>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<script>
const ctx = document.getElementById('yearDistributionChart').getContext('2d');

const yearData = {
    labels: [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
    datasets: [
        {
            label: 'Positive (Accepted)',
            data: [1, 1, 1, 3, 3, 5, 8, 8, 9, 7, 9, 14, 7, 21, 23, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            backgroundColor: 'rgba(34, 197, 94, 0.8)',
            borderColor: 'rgba(34, 197, 94, 1)',
            pointRadius: 8,
            pointHoverRadius: 10,
            showLine: true,
            tension: 0.1
        },
        {
            label: 'Negative (Rejected)',
            data: [0, 0, 0, 0, 0, 0, 52, 4, 0, 0, 0, 41, 13, 5, 1, 0, 43, 17, 0, 0, 0, 62, 5, 1],
            backgroundColor: 'rgba(239, 68, 68, 0.8)',
            borderColor: 'rgba(239, 68, 68, 1)',
            pointRadius: 8,
            pointHoverRadius: 10,
            showLine: true,
            tension: 0.1
        }
    ]
};

const config = {
    type: 'scatter',
    data: {
        datasets: [
            {
                label: 'Positive (Accepted)',
                data: [
                    {x: 1999, y: 1}, {x: 2000, y: 1}, {x: 2001, y: 1}, {x: 2002, y: 3},
                    {x: 2003, y: 3}, {x: 2004, y: 5}, {x: 2005, y: 8}, {x: 2006, y: 8},
                    {x: 2007, y: 9}, {x: 2008, y: 7}, {x: 2009, y: 9}, {x: 2010, y: 14},
                    {x: 2011, y: 7}, {x: 2012, y: 21}, {x: 2013, y: 23}
                ],
                backgroundColor: 'rgba(34, 197, 94, 0.8)',
                borderColor: 'rgba(34, 197, 94, 1)',
                pointRadius: 10,
                pointHoverRadius: 14,
                showLine: true,
                borderWidth: 2,
                tension: 0.3
            },
            {
                label: 'Negative (Rejected)',
                data: [
                    {x: 2005, y: 52}, {x: 2006, y: 4}, {x: 2010, y: 41}, {x: 2011, y: 13},
                    {x: 2012, y: 5}, {x: 2013, y: 1}, {x: 2015, y: 43}, {x: 2016, y: 17},
                    {x: 2020, y: 62}, {x: 2021, y: 5}, {x: 2022, y: 1}
                ],
                backgroundColor: 'rgba(239, 68, 68, 0.8)',
                borderColor: 'rgba(239, 68, 68, 1)',
                pointRadius: 10,
                pointHoverRadius: 14,
                showLine: true,
                borderWidth: 2,
                tension: 0.3
            }
        ]
    },
    options: {
        responsive: true,
        plugins: {
            title: {
                display: true,
                text: 'Publication Year Distribution: Positive vs Negative Papers',
                font: { size: 18 }
            },
            legend: {
                position: 'top',
                labels: { font: { size: 14 } }
            },
            tooltip: {
                callbacks: {
                    label: function(context) {
                        return `${context.dataset.label}: ${context.parsed.y} papers in ${context.parsed.x}`;
                    }
                }
            }
        },
        scales: {
            x: {
                type: 'linear',
                position: 'bottom',
                title: {
                    display: true,
                    text: 'Publication Year',
                    font: { size: 14 }
                },
                min: 1998,
                max: 2024,
                ticks: {
                    stepSize: 2,
                    callback: function(value) {
                        return value;
                    }
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'Number of Papers',
                    font: { size: 14 }
                },
                min: 0,
                max: 70
            }
        }
    }
};

new Chart(ctx, config);
</script>

## Key Observations

1. **Temporal Gap**: Positive papers (green) end at 2013, while negative papers (red) continue through 2022
2. **No Overlap Post-2013**: Zero positive papers from 2014-2024, but 128 negative papers in this range
3. **Peak Years**:
   - Positive: 2012-2013 (44 papers)
   - Negative: 2005, 2010, 2015, 2020 (corresponding to the Excel sheet tabs)

## Implications for Classification

This temporal mismatch could cause a machine learning classifier to learn spurious correlations:

- Papers from 2015+ would likely be classified as negative regardless of content
- The classifier might use publication date as a proxy rather than actual quality criteria

**Recommendation**: Balance the datasets by adding recent positive papers or restricting analysis to overlapping years (2005-2013).
