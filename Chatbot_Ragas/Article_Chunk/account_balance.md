# User Story: Viewing and Refilling Account Balance

## As a user, I want to see my Account Balance and refill it by cash or credit card.

### Acceptance Criteria

- When a logged-in user clicks on a button with their username, they are redirected to the Account Balance screen.

- The Account Balance screen contains the following fields:
  - Account Balance: System shows the user's current balance in float format.
  - Amount: System allows the user to enter only numbers.
  - Payment Method: Dropdown with options for Cash and Credit Card. If Credit Card is selected, additional fields are displayed:
    - Card Number: Only numbers.
    - Expiry Date: Date format.
    - Card CVV: 3 numbers.
    
- After clicking on the "Add Amount" button, a message 'Your changes are done' is displayed, and the Account Balance field shows the calculated balance.

### Database Tracking

- Data are added in Users and User_Payments tables.
- The Card Number, Expiry Date, and Card CVV fields are Null in case of selecting the Cash option.