# Ad Performance Case Study

## The Question

A company ran a Facebook ad campaign for a free iOS app in Brazil (Oct 7 – Dec 31, 2014). To what extent was it profitable? What should we do next?

## Key Findings

**The campaign was highly profitable:**

- Cost per install (avg): \$0.039
- Value per install (30-day): \$0.092
- Payback: day 0 on most days
- ROI: ~138% profit per dollar spent

**Three insights:**

1. **The campaign created far more installs than attribution shows.** Non-attributed downloads jumped 60× when the campaign started. These are real users the campaign brought in.
2. **Value takes time to realize.** Day-0 profit numbers understate true value. Users generate most revenue in the first week, but it extends to 30+ days.
3. **Tuesday is 15% more efficient than the average, Thursday/Friday are 8%--10% less efficient than the average.** We can increase profit by shifting daily budgets toward better-performing weekdays.

**Next steps:** validate with a geographical holdout test, shift budgets by weekday, and refresh creatives to address December decline.

## Structure

```
data/raw/          # Original CSVs
data/processed/    # Cleaned CSVs
notebooks/         # Main analysis notebook
reports/           # Exported HTML
```

## The Data

Two CSV files with daily metrics:

**Campaign data** (`ad_data.csv`): spend, CPM, CTR, paid downloads, CPI, ARPD, APPD
**App-wide data** (`app_data.csv`): total daily downloads, total daily revenue

Cleaned versions are in `data/processed/`.

## How to Run

```bash
# 1. Install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. View interactively
jupyter lab notebooks/01_ad_performance_case.ipynb

# 3. Or generate HTML
make report
```
