# User Story: Tracking User Login and Logout

## As a user (admin/non-admin), I can log in to the system.

### Acceptance Criteria

- Each login/logout datetime should be tracked in the database by user id.

- A "user_login_history" table is added with the following columns:
  - id: Not Null, Primary Key, INT
  - login_date: datetime
  - logout_date: datetime
  - user_id: INT, Foreign Key

- Any time a user logs in to the system, a new row is added to the "user_login_history" table with the datetime in the login_date column and a Null value in the logout_date column.

- After session timeout (configurable time) or after logout, the logout_date is populated with the server date.