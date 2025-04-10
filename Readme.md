# 📊 JNTUH Results Scraper & Analyzer

This Python script automates the process of fetching student results from the official JNTUH results portal for different program types — `REG`, `IDP`, and `IDDMP`. It scrapes data like student name, credits secured, SGPA, and result status, and generates a neat, summarized **PDF report**.

---

## 🚀 Features

- Select program type: REG, IDP, or IDDMP
- Roll number range input or pre-filled batches
- Scrapes result data using Selenium
- Automatically calculates:
  - 🎓 Average SGPA
  - 🥇 Highest SGPA
  - 🥉 Lowest SGPA
  - ❌ Failure statistics
- Outputs data into a **PDF file**
- Department name is extracted and printed once at the top of the report

---

## 🛠️ Requirements

Install dependencies with:

```bash
pip install selenium fpdf webdriver-manager
