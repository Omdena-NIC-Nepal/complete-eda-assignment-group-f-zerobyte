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
        """Load and execute the Jupyter Notebook before running tests."""
        with open('climate_eda.ipynb', 'r', encoding='utf-8') as f:
            cls.notebook = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
        try:
            ep.preprocess(cls.notebook, {'metadata': {'path': '.'}})
        except Exception as e:
            raise RuntimeError(f"Notebook execution failed: {e}")

        # Extract code and markdown cells
        cls.code_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'code']
        cls.markdown_cells = [cell for cell in cls.notebook.cells if cell['cell_type'] == 'markdown']
        cls.all_code = '\n'.join([cell['source'] for cell in cls.code_cells])
        cls.all_markdown = '\n'.join([cell['source'] for cell in cls.markdown_cells])
        
    def test_required_libraries(self):
        """Test that all required libraries are imported."""
        required_libs = ['pandas', 'numpy', 'matplotlib', 'seaborn']
        for lib in required_libs:
            self.assertIn(f"import {lib}", self.all_code, f"Missing required import: {lib}")
            
    def test_data_loading(self):
        """Test that climate data is loaded correctly."""
        self.assertRegex(self.all_code, r"read_csv\(['\"]data/Climate_Change_Indicators.csv['\"]\)", "Dataset not loaded properly")

    def test_yearly_aggregation(self):
        """Test that data is aggregated by year."""
        patterns = [r"groupby\(\s*['\"]Year['\"]\s*\)", r"resample\(\s*['\"]Y['\"]\s*\)"]
        found = any(re.search(pattern, self.all_code) for pattern in patterns)
        self.assertTrue(found, "No yearly data aggregation detected")

    def test_univariate_analysis(self):
        """Test for univariate visualizations and statistics."""
        vis_patterns = [r"hist\(", r"boxplot\(", r"displot\(", r"kdeplot\("]
        stats_patterns = [r"describe\(", r"mean\(", r"median\(", r"std\("]
        
        found_vis = any(re.search(pattern, self.all_code) for pattern in vis_patterns)
        found_stats = any(re.search(pattern, self.all_code) for pattern in stats_patterns)

        self.assertTrue(found_vis, "No univariate visualizations found")
        self.assertTrue(found_stats, "No descriptive statistics found")

    def test_bivariate_analysis(self):
        """Test for bivariate analysis using scatter plots or correlation analysis."""
        bivariate_patterns = [r"scatter\(", r"regplot\(", r"heatmap\("]
        corr_patterns = [r"corr\("]

        found_bivariate = any(re.search(pattern, self.all_code) for pattern in bivariate_patterns)
        found_corr = any(re.search(pattern, self.all_code) for pattern in corr_patterns)

        self.assertTrue(found_bivariate, "No bivariate visualizations found")
        self.assertTrue(found_corr, "No correlation analysis found")

    def test_multivariate_analysis(self):
        """Test for multivariate visualizations."""
        multivariate_patterns = [r"pairplot\(", r"PCA\(", r"heatmap\(", r"3d scatter"]
        found_multivariate = any(re.search(pattern, self.all_code) for pattern in multivariate_patterns)
        self.assertTrue(found_multivariate, "No multivariate visualizations found")

    def test_conclusions_present(self):
        """Test if conclusions or insights are written in markdown cells."""
        conclusion_patterns = [r"[Cc]onclusion", r"[Ff]inding", r"[Ss]ummary", r"[Ii]nsight"]
        found_conclusion = any(re.search(pattern, self.all_markdown) for pattern in conclusion_patterns)
        self.assertTrue(found_conclusion, "No conclusion or insights found in markdown cells")

    def test_min_number_of_visualizations(self):
        """Test that there are at least 5 visualizations."""
        vis_patterns = [r"plt\.\w+\(", r"sns\.\w+\(", r"df\.\w+\.plot\("]
        num_vis = sum(len(re.findall(pattern, self.all_code)) for pattern in vis_patterns)
        self.assertGreaterEqual(num_vis, 5, "Insufficient number of visualizations (minimum 5 required)")

    def test_climate_variables_analyzed(self):
        """Test that key climate variables are analyzed."""
        climate_vars = ['Global Average Temperature', 'CO2 Concentration', 'Sea Level Rise', 'Arctic Ice Area']
        found_vars = any(var in self.all_code for var in climate_vars)
        self.assertTrue(found_vars, "Not all required climate variables were analyzed")

    @classmethod
    def calculate_grade(cls, test_result):
        """Calculate the grade based on test results."""
        total_tests = test_result.testsRun
        failed_tests = len(test_result.failures) + len(test_result.errors)
        passed_tests = total_tests - failed_tests
        return (passed_tests / total_tests) * 100 if total_tests > 0 else 0

if __name__ == '__main__':
    # Run all tests and capture results
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestClimateEDA)
    test_runner = unittest.TextTestRunner(verbosity=2)
    test_result = test_runner.run(test_suite)

    # Calculate the final grade
    grade = TestClimateEDA.calculate_grade(test_result)

    print(f"\nFinal Grade: {round(grade)}/100")
