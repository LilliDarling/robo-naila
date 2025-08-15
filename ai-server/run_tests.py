#!/usr/bin/env python3
"""Comprehensive test runner for AI server test suite"""

import sys
import os
import subprocess
import argparse
from pathlib import Path


def main():
    """Run all tests with comprehensive configuration"""
    
    # Add ai-server to Python path
    ai_server_path = Path(__file__).parent
    sys.path.insert(0, str(ai_server_path))
    
    parser = argparse.ArgumentParser(description='Run AI server test suite')
    parser.add_argument('--coverage', action='store_true', help='Run with coverage reporting')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--fast', action='store_true', help='Skip slow performance tests')
    parser.add_argument('--markers', '-m', help='Run specific test markers (e.g. unit, integration)')
    
    args = parser.parse_args()
    
    # Base uv command
    cmd = ['uv', 'run', 'pytest']
    
    # Test discovery paths
    cmd.extend([
        'tests/unit/',
        'tests/integration/', 
        'tests/performance/',
        'tests/e2e/'
    ])
    
    # Output formatting
    cmd.extend([
        '-v' if args.verbose else '-q',
        '--tb=short',
        '--show-capture=no',
        '--durations=10'
    ])
    
    # Markers
    if args.fast:
        cmd.extend(['-m', 'not performance'])
    elif args.markers:
        cmd.extend(['-m', args.markers])
    
    # Coverage
    if args.coverage:
        cmd.extend([
            '--cov=agents',
            '--cov=graphs', 
            '--cov=memory',
            '--cov=config',
            '--cov-report=html:htmlcov',
            '--cov-report=term-missing'
        ])
    
    # Parallel execution
    cmd.extend(['-n', 'auto'])
    
    # Color output
    cmd.append('--color=yes')
    
    print(f"Running command: {' '.join(cmd)}")
    print("=" * 80)
    
    # Execute tests
    result = subprocess.run(cmd, cwd=ai_server_path)
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED!")
        if args.coverage:
            print("Coverage report generated in htmlcov/")
    else:
        print(f"\nTests failed with exit code {result.returncode}")
        sys.exit(result.returncode)


if __name__ == '__main__':
    main()