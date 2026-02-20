# User Story: Course Actions

## As an admin, I want to be able to add new courses, edit, or delete them. Both admin and non-admin users can view all courses.

### Acceptance Criteria

1. **Add Course Screen:**
   - The Add Course screen should have 4 fields: Title, Price, Type, Description, and a Submit button. (Users can add courses from the Home screen)
     - Title: System should allow alphabetical characters, up to 50 characters.
     - Price: System should allow only numbers (int/float/decimal).
     - Type: Drop-down should contain 2 types: Fundamental and Advanced.
     - Description: Any characters, up to 250.
   - The Submit button should be in orange color.
   - After adding a new course, the user is redirected to the Home page where all courses are visible.

2. **Course Viewing:**
   - Both admin and non-admin users can view all courses.
   - Users can see separated courses on the Course page: Advanced and Fundamental.
   - Selecting any of Advanced and Fundamental, the user is redirected to the corresponding course section. Author, Title, and Buy columns should be visible on the screen.
   - After clicking on Buy, the user is redirected to a separate screen where they can buy the course. The screen contains the following fields:
     - Course title-price
     - Course description
     - Buy button (orange color)

3. **Non-Admin User Behavior:**
   - After clicking on the Buy button:
     - If the account balance is insufficient, the user is redirected to the My Courses screen.
     - In case of enough account balance, the user can see their bought courses on the My Courses screen.

4. **Admin User Behavior:**
   - The Buy button is not visible.
   - The Title is a link.
   - After clicking on the Title, the admin is redirected to the Edit/delete screen.
   - After clicking on the Edit button, the user is redirected to the Edit Course page. The page contains Title and Description fields, and a Submit button.
   - After submission, the user gets a 'Your changes are done' message and redirects to the Courses page.
   - After clicking on the Delete button, the admin is redirected to the Courses page, and a "Course is deleted" message displays.