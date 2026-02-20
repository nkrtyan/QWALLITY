# User Story: Login to the system
## Login screen
Login screen contains the following fields:
- Username
- Password

It also includes:
- 2 buttons: Login, Cancel
- Link: Forget your password
- Toggle password visibility

## After successful login
After successful login, the user redirects to the Home page with the following message:
Welcome {registered name}

## After login with not existing user or not valid pass
system shows following message “Username or password is incorrect”. 
Clicking on Toggle, you can see input password.

System should block account after user 6 attempts.
Cancel button should reset all entered  data - both fields.