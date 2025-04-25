from Pages.main import Main
from faker import Faker
faker = Faker()

def test_sum(app):
    main_obj = Main(app)
    num_1 = faker.random_number(digits=1)
    num_2 = faker.random_number(digits=1)
    main_obj.sum_numbers(num_1, num_2)
    result = main_obj.get_result_text()
    expected_rusult = str(num_1+num_2)
    assert result == expected_rusult
