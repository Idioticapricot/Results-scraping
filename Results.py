from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from fpdf import FPDF
import time

# --------------------------------------
# Step 1: Program Selection & Roll Input
# --------------------------------------

print("Select Program Type:")
print("1. IDDMP\n2. IDP\n3. REG")
choice = input("Enter your choice (1/2/3): ")

program_map = {"1": "IDDMP", "2": "IDP", "3": "REG"}
program = program_map.get(choice)

if not program:
    print("❌ Invalid choice. Exiting.")
    exit()

# Set URL for each program
urls = {
    "IDDMP": "https://results.jntuhceh.ac.in/result/94f2480fc5516bc3ff4278a904d5094c",
    "IDP":   "https://results.jntuhceh.ac.in/result/381b2884d6a0acb4b5025ca3ded6d159",
    "REG":   "https://results.jntuhceh.ac.in/result/4088e2c620dacb40fa5899869b2cdc12"
}

url = urls[program]

# --------------------------------------
# Step 2: Roll Number Generation
# --------------------------------------

roll_numbers = []

if program == "IDDMP":
    roll_numbers += [f"23011M21{str(i).zfill(2)}" for i in range(1, 21)]
    roll_numbers += [f"23011M22{str(i).zfill(2)}" for i in range(1, 21)]
    roll_numbers += [f"23011MB5{str(i).zfill(2)}" for i in range(1, 21)]

elif program in ["REG", "IDP"]:
    prefix = "23011A6" if program == "REG" else "23011P0"
    try:
        start = int(input("Enter starting number (e.g., 301): "))
        end = int(input("Enter ending number (e.g., 360): "))
        for i in range(start, end + 1):
            roll_numbers.append(prefix + str(i))
    except ValueError:
        print("❌ Invalid input. Please enter numbers only.")
        exit()

# --------------------------------------
# Step 3: PDF & Scraping Setup
# --------------------------------------

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)
pdf.cell(0, 10, f"JNTUH Results - {program} | Name | Credits | SGPA | Status", ln=True)
pdf.ln(5)

department_name = None  # to store department only once

sgpa_list = []
fail = 0
total = 0

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))

# --------------------------------------
# Step 4: Scraping Each Roll Number
# --------------------------------------

for roll in roll_numbers:
    print(f"Checking: {roll}")
    driver.get(url)

    try:
        wait = WebDriverWait(driver, 10)
        input_box = wait.until(EC.element_to_be_clickable((By.CLASS_NAME, "hallticket")))
        input_box.clear()
        input_box.send_keys(roll)
        time.sleep(0.5)

        name = driver.find_element(By.XPATH, '//div[@data-title="Full Name"]').text.strip()
        credits = driver.find_element(By.XPATH, '//div[@data-title="Credits Secured"]').text.strip()
        sgpa = driver.find_element(By.XPATH, '//div[@data-title="SGPA"]').text.strip()
        status = driver.find_element(By.XPATH, '//div[@data-title="Result Status"]').text.strip()

        if not department_name:
            try:
                department_name = driver.find_element(By.XPATH, '//div[@data-title="Branch"]').text.strip()
                pdf.set_font("Arial", 'B', size=13)
                pdf.cell(0, 10, f"Department: {department_name}", ln=True)
                pdf.ln(5)
                pdf.set_font("Arial", size=12)
            except:
                department_name = "Not Found"

        total += 1
        if status.lower() == "fail":
            fail += 1

        try:
            sgpa_value = float(sgpa)
            sgpa_list.append((sgpa_value, roll, name))
        except:
            sgpa_value = None

        pdf.cell(0, 8, f"{roll}: {name} | Credits: {credits} | SGPA: {sgpa} | {status}", ln=True)

    except Exception as e:
        pdf.cell(0, 8, f"{roll}: Result not found or error.", ln=True)

# --------------------------------------
# Step 5: Summary
# --------------------------------------

pdf.ln(5)
pdf.set_font("Arial", 'B', size=12)
pdf.cell(0, 10, "GPA Summary", ln=True)

if sgpa_list:
    s_values = [val[0] for val in sgpa_list]
    avg = round(sum(s_values) / len(s_values), 2)
    highest = max(sgpa_list)
    lowest = min(sgpa_list)

    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"Average SGPA: {avg}", ln=True)
    pdf.cell(0, 8, f"Highest SGPA: {highest[0]} - {highest[1]} ({highest[2]})", ln=True)
    pdf.cell(0, 8, f"Lowest SGPA: {lowest[0]} - {lowest[1]} ({lowest[2]})", ln=True)
else:
    pdf.cell(0, 8, "No SGPA data available.", ln=True)

pdf.ln(5)
pdf.set_font("Arial", 'B', size=12)
pdf.cell(0, 10, "Failure Statistics", ln=True)
pdf.set_font("Arial", size=12)
fail_percent = (fail / total) * 100 if total > 0 else 0
pdf.cell(0, 8, f"Total Students: {total}", ln=True)
pdf.cell(0, 8, f"Fail: {fail}", ln=True)
pdf.cell(0, 8, f"Fail Percentage: {fail_percent:.2f}%", ln=True)

# --------------------------------------
# Step 6: Save PDF & Exit
# --------------------------------------

driver.quit()
filename = f"{department_name}_{program}_Cyber_Results.pdf"
pdf.output(filename)
print(f"✅ PDF saved as '{department_name}'")
