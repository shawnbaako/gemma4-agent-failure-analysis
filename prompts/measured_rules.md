# Prompt rules measured in the writeup

Full candidate directories will be added after the competition's final submission deadline (2026-12-02).

## Mechanics rules added between v1 and v3 (Section 3)

- Go straight to them: `run_command` with `grep -rn "<lead>" --include=*.py <package_dir> | head -30`, then `read_file` with explicit `start_line` and `end_line` (about 40 lines either side of the grep hit). `read_file` without a range returns only the first 150 lines, so re-reading a file without a range never shows you more. The workspace layout in the task message tells you the package directory.
- Write `old_string` and `new_string` as plain literal text with real line breaks, exactly as the file shows them. Never wrap a value in backticks, never escape quotes as \" and never write \n for a newline: the tool parser drops the whole call and replies "mandatory input parameters are not present".
- If you get that "mandatory input parameters" error, do not resend the same call. Either anchor on neighbouring lines that contain no double quotes or backticks and keep the quoted line inside `new_string` only, or apply the change with `run_command` using a `python3 - <<'EOF'` script that reads the file, does `str.replace`, and writes it back.
- Make your first edit by roughly the 15th tool call or 3 minutes, whichever comes first, even if you are not certain; a wrong edit can be corrected, an uncommitted investigation scores zero.

## Reproduce-before-edit rule (v4, Section 6)

### 1b. If the statement does not describe the failure, observe it before you edit
- Many statements are only a PR title plus an issue link you cannot open ("fix for padding width", "preserve newlines"). The title tells you where the bug is, not what it does. Do not guess a fix from two words: 3 of 4 such guesses fail the hidden tests.
- Instead, once you have found the named function, spend 2-3 tool calls making the bug observable: write a tiny script to `/tmp` that calls that function with the edge case the title implies (empty input, zero width, trailing newline, unicode, None, a very long or nested value), run it with `timeout 20 python3 /tmp/repro.py`, and compare the output with what the title says should happen. Look at the nearest existing test for that function to see how it is normally called.
- Fix the behavior you observed, then rerun the same script to confirm. If you cannot provoke the failure within 3 calls, make your best edit anyway and move on; never let reproduction push your first edit past the 15th call.
- Skip this section when the statement already states the wrong behavior, a traceback, or the expected result.


## Read-the-tests rule (v5, Section 6)

### 1b. If the statement does not describe the failure, read the tests that pin down the expected behavior
- Many statements are only a PR title plus an issue link you cannot open ("fix for padding width", "preserve newlines"). The title tells you where the bug is, not what the correct behavior is. Do not guess the expected behavior; look it up in the repository's own tests.
- Once you have found the named function or class, call `get_code_neighbors("<pkg.module.Class.method>")` with its dotted path (for example `get_code_neighbors("rich.segment.Segment.split_cells")`). Neighbors whose path starts with `tests.` are the existing tests that exercise it. If there are none, use `run_command` with `grep -rn "<function_name>" tests | head -20`.
- Read one or two of those tests with a ranged `read_file` (find the line with `grep -n "def <test_name>" <test_file>`). They show how the function is called and the exact output expected for ordinary inputs. The bug is an input close to those cases (the edge case the title names) where the code does not follow the same rule.
- Spend at most 4 tool calls here. Then write the fix so the title's edge case behaves consistently with the tested cases. Never edit the tests themselves.
- Skip this section when the statement already states the wrong behavior, a traceback, or the expected result.

