import re

def calculate(expression: str) -> str:
    """
    Evaluates a simple mathematical expression.
    Supported operators: +, -, *, /
    """
    # Basic security check to only allow math operations
    if not re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', expression):
         return "Error: Invalid characters in expression."
    
    try:
         # Safely evaluate the math expression
         result = eval(expression, {"__builtins__": None}, {})
         return str(result)
    except Exception as e:
         return f"Error evaluating expression: {e}"
