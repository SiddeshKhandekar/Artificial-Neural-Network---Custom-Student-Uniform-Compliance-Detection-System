"""Fix unicode characters in print statements for Windows cp1252 compatibility."""
import glob

files = glob.glob('backend/app/**/*.py', recursive=True)

replacements = {
    'print("    \\u2713': 'print("    [OK]',
    'print("  \\u2713':   'print("  [OK]',
    'print(f"    \\u2713': 'print(f"    [OK]',
    'print(f"    \\u2717': 'print(f"    [FAIL]',
    'print("    \\u2717':  'print("    [FAIL]',
    'print("    \\u2298':  'print("    [SKIP]',
    'print(f"    \\u2298': 'print(f"    [SKIP]',
    'print("    \\u26a0':  'print("    [WARN]',
    'print(f"    \\u26a0': 'print(f"    [WARN]',
    'print("  \\u2192 ':   'print("  -> ',
}

for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original = content
    for old, new in replacements.items():
        # Use actual unicode chars, not escape sequences
        pass
    
    # Direct replacements using actual chars
    content = content.replace('\u2713', '[OK]')  # ✓
    content = content.replace('\u2717', '[FAIL]')  # ✗
    content = content.replace('\u2298', '[SKIP]')  # ⊘
    content = content.replace('\u26a0', '[WARN]')  # ⚠
    content = content.replace('\u2192', '->')  # →
    content = content.replace('\u2014', '--')  # —
    
    if content != original:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Fixed: {fpath}')

print('Done!')
