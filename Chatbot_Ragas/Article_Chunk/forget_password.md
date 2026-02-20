# User Story: Resetting Password using 'Forget Password' Link

## As a user, I can reset my password using the 'Forget Password' link.

### Acceptance Criteria

- After clicking on 'Forget Password', the user is redirected to the Change Password screen.

- The Change Password screen contains the following fields:
  - Information about the process.
  - Email field for the user to enter their valid email.
  
- Once the user adds a valid email and sends it, a message 'Secure code sent to your email' is displayed, and a code is sent to the mentioned email.

- When the user clicks on the 'Change Password' link in the email, they are navigated to the Change Password screen with the secure code.

- The Change Password screen contains:
  - Code text field: Minimum value of at least 6 characters.
  - Password text field: Must be between 6 and 14 characters.
  - Submit button.