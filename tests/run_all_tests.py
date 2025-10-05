# run_all_tests.py
import subprocess
import sys

tests = [
    "test_config.py",
    "test_env_creation.py",
    "test_reset.py",
    "test_episode.py",
    "test_soc_dynamics.py",
    "test_rewards.py",
    "test_full_episode.py",
    "test_multiple_episodes.py",
    "test_gym_api.py"
]

print("="*60)
print("RUNNING ALL VERIFICATION TESTS")
print("="*60)

failed = []

for test in tests:
    print(f"\n{'='*60}")
    print(f"Running {test}...")
    print("="*60)
    
    result = subprocess.run([sys.executable, test], capture_output=True, text=True)
    
    if result.returncode != 0:
        failed.append(test)
        print(f"\n✗ {test} FAILED")
        print(result.stderr)
    else:
        print(f"\n✓ {test} PASSED")

print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Total tests: {len(tests)}")
print(f"Passed: {len(tests) - len(failed)}")
print(f"Failed: {len(failed)}")

if failed:
    print("\nFailed tests:")
    for test in failed:
        print(f"  - {test}")
    sys.exit(1)
else:
    print("\n✓ ALL TESTS PASSED - Environment is ready for training!")