# TODO List for dl-resq Project

## Critical Issues

### 1. Shell Script Syntax Errors
**Priority: HIGH**
- **Location**: Multiple shell scripts
- **Issue**: Trailing backslashes at the end of commands cause syntax errors
  - `fake_quant/0_get_basis.sh` (line 19)
  - `fake_quant/2_eval_ptq.sh` (line 35)
  - `fake_quant/4_collect_act.sh` (line 34)
- **Fix**: Remove trailing backslash from the last line of multi-line commands
- **Status**: Fixed ✓

### 2. Missing Shebang Lines
**Priority: MEDIUM**
- **Location**: All shell scripts in `fake_quant/`
- **Issue**: Shell scripts lack shebang (`#!/bin/bash`) lines
- **Impact**: Scripts may execute with wrong shell interpreter
- **Fix**: Add `#!/bin/bash` as the first line (before copyright header or after it)
- **Status**: Fixed ✓

## Code Quality Issues

### 3. Bare Except Clauses
**Priority: MEDIUM**
- **Location**: 
  - `fake_quant/utils/metrics.py` (line 125)
  - `fake_quant/train_utils/optimizer.py` (line 180)
- **Issue**: Using bare `except:` catches all exceptions including system exits
- **Fix**: Use specific exception types (e.g., `except ValueError:` or `except Exception:`)
- **Best Practice**: Specify the exception types to catch
- **Status**: Fixed ✓

### 4. Logic Bug in classification_score Function
**Priority: MEDIUM**
- **Location**: `fake_quant/utils/metrics.py` (line 104)
- **Issue**: Comparing list to integer `if em_match_list != 0:` instead of checking length
- **Fix**: Change to `if len(em_match_list) != 0:` or `if em_match_list:`
- **Impact**: Function may fail or produce incorrect results
- **Status**: Fixed ✓

### 5. Deprecated Tensor Operation
**Priority: LOW**
- **Location**: `fake_quant/train_utils/optimizer.py` (line 179)
- **Issue**: Using deprecated `add_` with two arguments: `d_p.add_(weight_decay, p.data)`
- **Fix**: Update to: `d_p.add_(p.data, alpha=weight_decay)`
- **Status**: Fixed ✓

## Documentation Issues

### 6. Missing LICENSE File
**Priority: MEDIUM**
- **Issue**: No LICENSE file in repository root
- **Impact**: Copyright notices in files reference LICENSE but it doesn't exist
- **Fix**: Add appropriate LICENSE file (likely MIT or Apache 2.0 based on Meta copyright)
- **Status**: Not fixed (requires clarification from maintainers)

### 7. Incomplete README Instructions
**Priority: LOW**
- **Location**: `README.md`
- **Issue**: 
  - Step 4 mentions cloning fast-hadamard-transform but uses SSH URL which may fail for users without SSH keys
  - No troubleshooting section
  - No information about model download requirements or HuggingFace token setup
- **Suggestions**:
  - Add HTTPS alternative for git clone
  - Add troubleshooting section
  - Document model access requirements
- **Status**: Not implemented (out of scope for bug fixes)

## Configuration and Setup Issues

### 8. Missing .editorconfig
**Priority: LOW**
- **Issue**: No `.editorconfig` file for consistent coding standards
- **Fix**: Add `.editorconfig` with Python and shell script standards
- **Status**: Fixed ✓

### 9. Incomplete .gitignore
**Priority: LOW**
- **Location**: `.gitignore`
- **Issue**: Missing common Python patterns
- **Suggestions to add**:
  - `__pycache__/`
  - `*.egg-info/`
  - `.pytest_cache/`
  - `.venv/`
  - `.env`
- **Status**: Fixed ✓

## Dependency Issues

### 10. Potentially Outdated Dependencies
**Priority: LOW**
- **Location**: `requirements.txt`
- **Issue**: Pinned versions may have security vulnerabilities
- **Recommendation**: Regularly check for security updates
- **Status**: Requires ongoing maintenance

### 11. Missing fast-hadamard-transform in requirements
**Priority: MEDIUM**
- **Issue**: README mentions installing fast-hadamard-transform separately
- **Impact**: Not clear if this is a hard dependency or optional
- **Recommendation**: Clarify in README or add installation instructions
- **Status**: Not addressed (documentation improvement)

## Code Structure Issues

### 12. Large TODO Comments in Code
**Priority: LOW**
- **Location**: Multiple files (see grep results)
- **Issue**: Many TODO comments exist in the codebase
- **Examples**:
  - `fake_quant/utils/LMClass.py`: Multiple TODOs for caching, batch size detection, etc.
  - `fake_quant/utils/quant_utils.py`: Refactoring TODO
- **Recommendation**: Convert to GitHub issues for tracking
- **Status**: Not addressed (requires project management decision)

## Testing Issues

### 13. No Visible Test Suite
**Priority: MEDIUM**
- **Issue**: No `tests/` directory or test files visible
- **Impact**: Difficult to verify code changes don't break functionality
- **Recommendation**: Add unit tests for core functionality
- **Status**: Not implemented (significant effort required)

## Performance and Best Practices

### 14. Hard-coded Device Settings
**Priority: LOW**
- **Location**: Multiple files using `utils.DEV`
- **Issue**: May not be flexible for different GPU configurations
- **Status**: Review needed but likely acceptable for research code

### 15. Commented-out Code
**Priority: LOW**
- **Location**: `fake_quant/get_basis.py` (line 79)
- **Example**: `# model.R1.weight.data.copy_() = model.R1.to(utils.DEV)`
- **Recommendation**: Remove commented-out code or document why it's kept
- **Status**: Not addressed

## Summary

### Fixed Issues (5)
1. ✓ Trailing backslashes in shell scripts
2. ✓ Missing shebang lines  
3. ✓ Bare except clauses
4. ✓ Logic bug in classification_score
5. ✓ Deprecated tensor operation
6. ✓ Missing .editorconfig
7. ✓ Incomplete .gitignore

### Requires Further Action (8)
1. Missing LICENSE file (needs maintainer decision)
2. Incomplete README (documentation improvement)
3. Missing fast-hadamard-transform clarity (documentation)
4. TODO comments in code (project management)
5. No test suite (significant effort)
6. Hard-coded device settings (review needed)
7. Commented-out code (cleanup)
8. Dependency updates (ongoing maintenance)

## Priority Ranking
1. **Critical**: Shell script syntax errors (FIXED)
2. **High**: Logic bugs and type errors (FIXED)
3. **Medium**: Code quality issues, missing LICENSE
4. **Low**: Documentation improvements, code cleanup
