# User Story: User Login and Profile Management

## As a user (admin/non-admin), I want to log in to the system.

### Acceptance Criteria

- Users can navigate to the Profile page and read/update the following data:
  - Country: Maximum 50 characters (letters).
  - City: Maximum 50 characters (letters).
  - Address: Maximum 50 characters (any).
  - Phone number: Only numbers.
  - Gender: Dropdown list with "Male" and "Female" options.
  - Marital status: Dropdown list with "single", "divorced", and "married" statuses.

- After clicking on the Save button, the data is saved in the Users table (MySQL) or Users collection (MongoDB).

- Database configuration: This user story should be implemented for both databases (MySQL and MongoDB) and should be configurable at the code level.