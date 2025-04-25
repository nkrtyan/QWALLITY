# Windows Calculator Automation with Pywinauto
This is a sample automation project that demonstrates how to automate a simple test case for the Windows Calculator using pywinauto.
The test case performs the sum of two numbers and verifies the result.


Installation
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt


# Running the Test
pytest Tests/test_case.py
The test will:
- Launch Windows Calculator
- Randomly generate two numbers 
- Click them in the calculator
- Click the "+" and "=" buttons
- Validate the result against the expected sum


# NOTE
We use Accessibility Insights for Windows to inspect and locate UI elements such as button auto_ids and control names.
🔗 Download Accessibility Insights - https://accessibilityinsights.io/downloads/


Steps:
- Launch Calculator
- Open Accessibility Insights
- Hover over elements to get AutomationId (e.g., num1Button, plusButton, etc.)


