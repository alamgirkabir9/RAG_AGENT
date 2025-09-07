import re
import math
from typing import Union

class Calculator:
    """Mathematical calculator tool"""
    
    def calculate(self, expression: str) -> str:
        """Calculate mathematical expressions safely"""
        
        try:
            # Convert written numbers to digits first
            expression = self._convert_written_numbers(expression)
            
            # Extract mathematical expression from natural language
            math_expr = self._extract_math_expression(expression)
            
            if not math_expr:
                return "I couldn't find a mathematical expression to calculate."
            
            # Safe evaluation
            result = self._safe_eval(math_expr)
            
            #return f"🧮 The result is: {result}"
            return f"🧮 The result is {result}\n (from calculator)"

        except Exception as e:
            return f"❌ Calculation error: {str(e)}"
    
    def _convert_written_numbers(self, text: str) -> str:
        """Convert written numbers to digits"""
        number_map = {
            'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
            'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9',
            'ten': '10', 'eleven': '11', 'twelve': '12', 'thirteen': '13',
            'fourteen': '14', 'fifteen': '15', 'sixteen': '16', 'seventeen': '17',
            'eighteen': '18', 'nineteen': '19', 'twenty': '20'
        }
        
        text_lower = text.lower()
        for word, digit in number_map.items():
            text_lower = text_lower.replace(word, digit)
            
        return text_lower
    
    def _extract_math_expression(self, text: str) -> str:
        """Extract mathematical expression from text"""
        
        # Replace common words with operators
        text = text.lower()
        text = re.sub(r'\bplus\b|\badd\b|\band\b', '+', text)
        text = re.sub(r'\bminus\b|\bsubtract\b', '-', text)
        text = re.sub(r'\btimes\b|\bmultiplied by\b|\bmultiply\b', '*', text)
        text = re.sub(r'\bdivided by\b|\bdivide\b|\bover\b', '/', text)
        text = re.sub(r'\bsquared\b', '**2', text)
        text = re.sub(r'\bcubed\b', '**3', text)
        
        # Remove extra words
        text = re.sub(r'\bis\b|\bwhat\b|\bequals\b|\bthe\b|\bresult\b|\bof\b', '', text)
        
        # Extract mathematical expressions - more flexible pattern
        math_pattern = r'[\d\+\-\*/\(\)\.\s\^]+'
        matches = re.findall(math_pattern, text)
        
        if matches:
            # Take the longest match and clean it up
            longest_match = max(matches, key=len).strip()
            # Remove multiple spaces
            longest_match = re.sub(r'\s+', ' ', longest_match)
            # Remove spaces around operators for cleaner evaluation
            longest_match = re.sub(r'\s*([+\-*/])\s*', r'\1', longest_match)
            return longest_match.replace('^', '**')
        
        return text.strip()
    
    def _safe_eval(self, expression: str) -> Union[int, float]:
        """Safely evaluate mathematical expressions"""
        
        # Replace ^ with ** for Python exponentiation
        expression = expression.replace('^', '**')
        # Remove any remaining spaces
        expression = expression.replace(' ', '')
        
        # Only allow safe characters
        allowed_chars = set('0123456789+-*/.()^')
        if not all(c in allowed_chars for c in expression):
            raise ValueError("Invalid characters in expression")
        
        # Add math functions support
        safe_dict = {
            "__builtins__": {},
            "sqrt": math.sqrt,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "pi": math.pi,
            "e": math.e
        }
        
        result = eval(expression, safe_dict)
        
        # Format result nicely
        if isinstance(result, float) and result.is_integer():
            return int(result)
        elif isinstance(result, float):
            return round(result, 6)
        return result