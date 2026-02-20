# User Story: Registering to the System

## As a user, I want to be able to register to the system, so that I can view information about online courses.

### Acceptance Criteria

- The following fields should be available on the Registration screen:
  - Name: A field up to 25 characters, accepting only letters.
  - Email: Format - xxx@domain. The system should display the following validation message if the format is invalid: "Please use valid email format." It should also be a unique entry.
  - Username: An alphanumeric field up to 50 characters. It should be a unique entry. If a duplicate username is entered, the system should display the message: "Username already exists."
  - Password: 8-14 alphanumeric characters.
  - Confirm Password: The value should be the same as the password.
  - Submit button: After submission, the user should see the following message: "Your account has been successfully created."

**Note:** In case of invalid data, no need to mention in red fields. All fields are required. After incorrect fields and submission, all fields should have validation messages.