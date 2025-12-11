# Continuous Selenium Scraper

This Python project uses Selenium and pandas to scrape data from a website and save it to a CSV file daily.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Download ChromeDriver and ensure it is in your PATH.

## Usage

Run the script:
```bash
python main.py
```

The script will run continuously, scraping data once per day and updating `daily_report.csv`.

## Configuration
- Update `url` and `.your-selector` in `main.py` to target your desired website and element.
