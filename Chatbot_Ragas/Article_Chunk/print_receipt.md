# User Story: Download and Print Receipt after Refilling Balance

## As a non-admin user, I want to have the opportunity to download and print a receipt with payment data after refilling my balance.

### Acceptance Criteria

- Add a "Print Receipt" link in the User Action screen. The "Print Receipt" link should always be clickable.
- If a user clicks on the "Print Receipt" link before checkout, the page is reloaded, and no template is downloaded.
- If a user checks out with Cash/Credit Card and clicks on the "Print Receipt" link, a "receipt.docx" file is downloaded.
- The receipt template should include the following required fields:
  - Username: Logged-in user's username.
  - Amount: Amount of refill.
  - Date: Current date and time at the moment of replenishment.
  - Payment Method: Cash/Credit Card.
  - Account Balance: Account balance after refill.
- If a user checks out several times with different amounts and clicks on the "Print Receipt" link, the system should print a receipt with the last checkout data.
- The link should be available only for non-admin users.