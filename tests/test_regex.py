import re

content_str = "{'input': 'I want to apply for a commercial loan of ₹60,000,000'}".lower()
amounts = re.findall(r'(?:rs\.?|inr|₹|\$)?\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', content_str)
print("Regex amounts:", amounts)

high_value_found = False
for amt in amounts:
    try:
        val = float(amt.replace(',', ''))
        if val >= 5000000:
            high_value_found = True
            break
    except ValueError:
        pass
        
print("High value found:", high_value_found)
