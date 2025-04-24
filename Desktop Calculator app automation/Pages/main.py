from Helpers.general_helpers import Helper
import logging


class Main(Helper):

    # locators
    btn_plus = "plusButton"
    btn_equal = "equalButton"
    result = "CalculatorResults"

    def click_number(self, number: int):
        btn_id = f"num{number}Button"
        self.window_app.child_window(auto_id=btn_id).click()
        logging.info(f"{number} is clicked")

    def click_arithmetic(self, btn_name: str):
        self.window_app.child_window(auto_id=btn_name).click()
        logging.info(f"{btn_name} is clicked")

    def get_result_text(self):
        res = self.window_app.child_window(auto_id=self.result).window_text()
        res_text = str(res).split()[2]
        logging.info(f"The text result is {res_text}")
        return res_text

    def sum_numbers(self, num1: int, num2: int):
        self.click_number(num1)
        self.click_arithmetic(self.btn_plus)
        self.click_number(num2)
        self.click_arithmetic(self.btn_equal)
