import unittest
import os
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import re
import pandas as pd
import numpy as np

class TestClimateEDA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """
        Set up the test class by loading and executing the notebook.
        Extract code and markdown cells for testing.
        """
        # Load the notebook
        notebook_path = 'climate_eda.ipynb'
        if not os.path.exists(notebook_path):
            raise FileNotFoundError(f"Notebook file '{notebook_path}' not found.")
        
        with open(notebook_path, 'r', encoding='utf-8') as f:
            cls.notebook = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(cls.notebook, {'metadata': {'path': '.'}})
        except Exception as e:
            raise RuntimeError(f"Failed to execute notebook: {e}")
        
        # Extract code and markdown cells
        cls.code_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'code']
        cls.markdown_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'markdown']
        
        # Combine all code and markdown content
        cls.all_code = '\n'.join([cell['source'] for cell in cls.code_cells]) if cls.code_cells else ''
        cls.all_markdown = '\n'.join([cell['source'] for cell in cls.markdown_cells]) if cls.markdown_cells else ''
        
        # Debug print to ensure all_code is properly assigned
        print(f"Extracted Code (first 500 chars): {cls.all_code[:500]}")
    
    def test_bivariate_analysis(self):
        """Test for bivariate analysis"""
        bivariate_vis_patterns = [
            r"scatter(plot)?\(",
            r"reg(plot)?\(",
            r"lineplot\(",
            r"barplot\(",
            r"violinplot\(",
            r"heatmap\(",
            r"corr\("
        ]
        found_bivariate_vis = any(re.search(pattern, type(self).all_code) for pattern in bivariate_vis_patterns)
        self.assertTrue(found_bivariate_vis, "No evidence of bivariate visualizations")
    
    def calculate_grade(self):
        """Calculate the grade based on passing tests"""
        test_methods = [method for method in dir(self) if method.startswith('test_')]
        total_tests = len(test_methods)
        
        passed_tests = 0
        for test in test_methods:
            try:
                getattr(self, test)()
                passed_tests += 1
            except AssertionError:
                continue
        
        grade = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
        return round(grade)

if __name__ == '__main__':
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestClimateEDA)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)
    
    # Calculate and print grade
    grade = TestClimateEDA().calculate_grade()
    print(f"\nFinal Grade: {grade}/100")