def test_vat_analysis():
    """
    Test the VAT analysis functionality with sample data
    """
    import tempfile
    import pandas as pd
    import numpy as np
    import os
    
    # Create a sample Excel file for testing
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        # Create a DataFrame with sample data
        data = {
            'Column1': ['', '', 'Total Sales for Dec 2024', ''] + [''] * 10 + ['Total'] + [''],
            'Column2': ['', '', '', 'Zero Rated', 'Exempt', 'Vatable 16%', '', '', 'Total Exempt Sales ----------------->', "Total '0' Rated, 8% & 16% vatable Sales", '', 'vat 16 Output taxwise - Registered', '100% Zero rated Vat', 'non- claimable Amount', 'proratable vat', 'Total', ' CLAIMABLE AMOUNT', ' TOTAL CLAIMABLE AMOUNT'],
            'Column3': ['', '', 1000000, 300000, 200000, 500000, '', '', 200000, 800000, '', 80000, 20000, 10000, 50000, 80000, 70000, 60000]
        }
        df = pd.DataFrame(data)
        
        # Save to Excel
        df.to_excel(tmp.name, index=False)
        
        # Run the analysis
        analysis_results = run_comprehensive_vat_analysis(tmp.name)
        
        # Check if results are generated correctly
        assert analysis_results is not None
        assert 'data' in analysis_results
        assert 'analyses' in analysis_results
        assert 'visualizations' in analysis_results
        assert 'report' in analysis_results
        
        # Check data extraction
        assert len(analysis_results['data']['sales_df']) > 0
        assert len(analysis_results['data']['vat_df']) > 0
        assert len(analysis_results['data']['total_df']) > 0
        
        # Check analyses
        assert analysis_results['analyses']['descriptive'] is not None
        assert analysis_results['analyses']['predictive'] is not None
        assert analysis_results['analyses']['prescriptive'] is not None
        assert analysis_results['analyses']['unsupervised'] is not None
        
        # Check visualizations
        assert analysis_results['visualizations']['descriptive'] is not None
        assert analysis_results['visualizations']['predictive'] is not None
        assert analysis_results['visualizations']['prescriptive'] is not None
        assert analysis_results['visualizations']['unsupervised'] is not None
        
        print("All tests passed!")
        
        # Clean up
        os.remove(tmp.name)

if __name__ == '__main__':
    # Run tests
    test_vat_analysis()
    
    # Start the Flask application
    app.run(debug=True, port=5000)
