import pytest

from utils import extract_boxed_answers
from utils import check_equivalence


def test_extract_boxed_answers():
    solutions_batch = [
        ["This is a solution with \\boxed{\frac{4}{2}}", "Another solution with \\boxed{\frac12}"],
        ["No boxed answer here", "Yet another \\boxed{100}"],
        ["These are two possible solutions, \\boxed{12} and \\boxed{23}", 'None']
    ]
    
    expected_output = ([
        ["\frac{4}{2}", "\frac12"],
        [None, "100"],
        [None, None]
    ],3)
    
    assert extract_boxed_answers(solutions_batch) == expected_output

def test_check_equivalence():
    # Test cases
    test_cases = [
        ("42", "42", True),
        ("42", "24", False),
        ("-10\\frac{1}{2}", "-5", True),
        ("100", "100.0", True),
        ("0.1", "0.10", True),
        ("1/2", "0.5", True),
        ("1/3", "0.333", False),
        (None, "1", False),
        ("x+1", "x+2", False),
        ("x+1", "x+2-1", True),
        ("10,\\!080","10080", True),
        ("\\dfrac{1}{10}","1/10", True)
    ]
    
    for answer1, answer2, expected in test_cases:
        assert check_equivalence([answer1], answer2) == expected


if __name__ == "__main__":
    pytest.main()