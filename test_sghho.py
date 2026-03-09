"""
Quick test script for SG-HHO algorithm.

This script performs a quick validation of the SG-HHO implementation
with reduced parameters for fast testing.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.registry import get_algorithm, discover_plugins
from core.logging_setup import setup_logger
import feature_selection  # Trigger plugin discovery
import logging

def test_sghho_basic():
    """Test basic SG-HHO functionality."""
    print("="*70)
    print("SG-HHO Quick Test")
    print("="*70)
    
    # Setup logger
    test_output_dir = project_root / 'log' / 'test_sghho'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(test_output_dir, module_name='sghho_test')
    
    # Discover plugins
    discover_plugins(feature_selection)
    
    # Check if SGHHO is registered
    try:
        SGHHOSelector = get_algorithm('SGHHO')
        print("✓ SGHHO algorithm successfully registered")
    except KeyError as e:
        print(f"✗ Failed to find SGHHO: {e}")
        return False
    
    # Create selector with reduced parameters for quick test
    print("\nInitializing SG-HHO with test parameters...")
    selector = SGHHOSelector(
        target_col='target',
        band_range=(0, 150),
        logger=logger,
        epoch=10,           # Reduced for quick test
        pop_size=10,        # Reduced for quick test
        window_size=8,
        alpha_stability=0.2,
        beta_sparsity=0.1,
        n_cv_runs=2         # Reduced for quick test
    )
    print("✓ Selector initialized successfully")
    
    # Check if data file exists
    data_path = project_root / 'resource' / 'dataSet.csv'
    if not data_path.exists():
        print(f"\n✗ Data file not found: {data_path}")
        print("  Please ensure resource/dataSet.csv exists")
        return False
    
    print(f"✓ Data file found: {data_path}")
    
    # Run feature selection
    print("\n" + "="*70)
    print("Running SG-HHO optimization (this may take 1-2 minutes)...")
    print("="*70)
    
    try:
        output_path = project_root / 'test_sghho_output.csv'
        result = selector.run_selection(
            input_path=str(data_path),
            output_path=str(output_path)
        )
        
        print("\n" + "="*70)
        print("Test Results")
        print("="*70)
        print(f"✓ Feature selection completed successfully")
        print(f"  Selected features: {len(result.selected_features)}")
        print(f"  Selected indices: {result.selected_indices[:10]}..." if len(result.selected_indices) > 10 else f"  Selected indices: {result.selected_indices}")
        print(f"  Output saved to: {output_path}")
        
        # Analyze results
        if len(result.selected_indices) > 1:
            import numpy as np
            indices = sorted(result.selected_indices)
            gaps = np.diff(indices)
            print(f"\nSpectral Continuity Analysis:")
            print(f"  Max gap: {np.max(gaps)}")
            print(f"  Avg gap: {np.mean(gaps):.2f}")
            print(f"  Min gap: {np.min(gaps)}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during feature selection: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sghho_import():
    """Test if SG-HHO modules can be imported."""
    print("\n" + "="*70)
    print("Testing Module Imports")
    print("="*70)
    
    try:
        from improve.SGHHO import SpectralGroupHHO
        print("✓ SpectralGroupHHO imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SpectralGroupHHO: {e}")
        return False
    
    try:
        from feature_selection.sghho import SGHHOSelector
        print("✓ SGHHOSelector imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import SGHHOSelector: {e}")
        return False
    
    return True

if __name__ == '__main__':
    print("\n" + "="*70)
    print("SG-HHO Implementation Test Suite")
    print("="*70)
    
    # Test 1: Module imports
    import_success = test_sghho_import()
    
    if not import_success:
        print("\n✗ Import test failed. Please check the implementation.")
        sys.exit(1)
    
    # Test 2: Basic functionality
    print("\n")
    basic_success = test_sghho_basic()
    
    if basic_success:
        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)
        print("\nSG-HHO is ready to use. You can now:")
        print("1. Update config.yaml to use algorithm: SGHHO")
        print("2. Run: python main.py")
        print("3. Check improve/README_SGHHO.md for detailed usage")
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("✗ Some tests failed")
        print("="*70)
        sys.exit(1)
