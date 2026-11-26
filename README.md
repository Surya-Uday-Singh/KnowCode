# KnowCode

KnowCode summarizes Python code for a specific reader: a **Developer**, a **QA Engineer** or a
**Product Manager**. It splits the code into parts (functions, classes, methods, script sections),
asks an LLM (DeepSeek) to summarize each part for that reader, has a second LLM call (the "critic")
score the summary, retries once if the score is too low, and checks the result with ordinary code.
You get one summary per part, or one document for the whole file, as Markdown or JSON.

COSC 4426 – LLMs in Software Development.

---

## 1. What you need

| Thing | How to check / get it |
|---|---|
| **Python 3.12 or newer** | Run `python3 --version` (Mac) or `py --version` (Windows). If it's older than 3.12, install it from [python.org/downloads](https://www.python.org/downloads/). |
| **Git** | Run `git --version`. If missing: [git-scm.com/downloads](https://git-scm.com/downloads). |
| **A DeepSeek API key** | Only needed to actually summarize code (tests run without it). See step 4 below. |
| **A code editor** (optional) | VS Code is what we use. |

All commands below are typed into a terminal. In VS Code: **View → Terminal**.

---

## 2. First-time setup (do this once)

### Step 1: Download the project
```bash
git clone https://github.com/Surya-Uday-Singh/KnowCode.git
cd KnowCode
```

### Step 2: Create a virtual environment
A virtual environment is a private folder (`.venv`) for this project's Python packages, so they
don't mix with anything else on your computer.

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows (PowerShell or Command Prompt):**
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```
Your terminal prompt should now start with **`(.venv)`**. That means it's active.

> If `python3 -m venv` uses an older Python, name the version directly: `python3.12 -m venv .venv`.

### Step 3: Install KnowCode
```bash
pip install -e ".[dev]"
```
This installs KnowCode and the tools it needs (and the `knowcode` command). It takes a minute.
The `-e` means "editable": when you change the code, the command uses your changes right away.

### Step 4: Add your DeepSeek API key
1. Get a key:
   - Sign up at [platform.deepseek.com](https://platform.deepseek.com).
   - Add a little credit on the top-up page (US$5 is plenty for the whole project).
   - Create a key on the [API keys page](https://platform.deepseek.com/api_keys). It starts with `sk-`.
2. Make your own settings file by copying the example:
   - Mac / Linux: `cp .env.example .env`
   - Windows: `copy .env.example .env`
3. Open `.env` in your editor and paste your key after the `=` (no quotes, no spaces):
   ```
   DEEPSEEK_API_KEY=sk-your-key-here
   ```

> ⚠️ **Never share your key or commit `.env`.** It's already in `.gitignore`, so git ignores it,
> but don't paste the key into code, chats or documents either. Anyone with the key can spend your credit.

### Step 5: Check that everything works
```bash
pytest -q
```
You should see something like `326 passed`. This is free: the tests use a fake LLM and never call DeepSeek.

Then check your key with one real call (costs about ¥0.01, a fraction of a cent):
```bash
python -m knowcode.llm
```
It prints a summary of a small example function and the critic's verdict. If you see an error,
check the **Troubleshooting** section below.

---

## 3. Every time you open a new terminal

The virtual environment switches off when you close the terminal. Turn it back on with:

**Mac / Linux:**
```bash
cd path/to/KnowCode
source .venv/bin/activate
```
**Windows:**
```bash
cd path\to\KnowCode
.venv\Scripts\activate
```

---

## 4. Running KnowCode

### The command
```bash
knowcode summarize FILE --role ROLE --granularity LEVEL [options]
```

| Part | Required? | Values | Meaning |
|---|---|---|---|
| `FILE` | yes | a `.py` file, or `-` | The code to summarize. `-` means you paste the code in (see below). |
| `--role` | **yes** | `developer`, `qa`, `pm` | Who the summary is for. |
| `--granularity` | **yes** | `function`, `file` | `function`: one summary per part (function, class, method or script section). `file`: those summaries combined into one document for the whole file. |
| `--format` | no | `md` (default), `json` | Markdown for reading, JSON for other programs. |
| `--output` | no | a file name | Save the result to a file instead of printing it. |
| `--chunk` | no | a part's name | Only summarize that one part (e.g. `round_price` or `Cart.add`). Good for quick, cheap tries. |
| `--log-dir` / `--no-log` | no | a folder / – | Where to save the run log (default `runs/`), or don't save one. |

`knowcode summarize --help` shows all of this in the terminal.

### Try it on the example file
The repo includes a small example, `tests/fixtures/sample_cart.py` (a shopping cart).
```bash
# One function, for a QA engineer (about 30-60 seconds)
knowcode summarize tests/fixtures/sample_cart.py --role qa --granularity function --chunk round_price

# The same function for a product manager: compare the wording
knowcode summarize tests/fixtures/sample_cart.py --role pm --granularity function --chunk round_price

# The whole file as one document for a developer, saved to a file (a few minutes)
knowcode summarize tests/fixtures/sample_cart.py --role developer --granularity file --output cart_dev.md

# JSON instead of Markdown
knowcode summarize tests/fixtures/sample_cart.py --role pm --granularity function --format json --chunk Cart
```
To read a saved `.md` file nicely in VS Code, open it and press **Cmd+Shift+V** (Mac) or **Ctrl+Shift+V** (Windows).

### Summarize a pasted snippet
```bash
knowcode summarize - --role developer --granularity function
```
Paste your code, then press **Ctrl-D** (Mac/Linux) or **Ctrl-Z then Enter** (Windows) on a new line.

### What you'll see
While it runs, a progress line appears for each part:
```
[1/4] round_price ... ✅ passed (5/5) · 3 calls · 31 s
[2/4] CartItem ... ⚠️ below threshold (3/5) · 6 calls · 99 s
...
Done: 4 parts (3 passed, 1 below threshold) · 18 LLM calls · about ¥0.143 (deepseek-flash off-peak estimate)
Run log: runs/20261123-153012-a1b2
```
Then the summary itself (or "Written to …" if you used `--output`). Each part has a status:

| Status | Meaning |
|---|---|
| ✅ passed | The critic scored it 4 or 5 out of 5 and the automatic checks found no problems. |
| ⚠️ passed, rule problem | Good score, but it broke a style rule (e.g. programming words in a PM summary). Kept and labelled. |
| ⚠️ below threshold | Still scored under 4 after one retry. The best attempt is kept. |
| ⚠️ not reviewed | The critic couldn't produce a score. The summary is kept but unchecked. |
| ⛔ not summarized | No usable summary (e.g. the part was too long, or the API failed). The reason is shown. |

Press **Ctrl-C** to stop a run early. Parts that already finished are kept in the run log.

### Run logs
Every run saves a folder under `runs/` (ignored by git, so it stays on your computer):

| File | What's in it |
|---|---|
| `run.json` | The inputs, every setting (never the API key), which version of the code ran, the status and totals (calls, tokens, time, cost). |
| `parts.jsonl` | Everything about each part, one long line per part: the exact prompts sent, every reply, the critic's reasoning, the automatic check results. |
| `source.py` | The exact code that was summarized. |
| `output.md` / `output.json` | Exactly what was printed or saved. |

To read one part of `parts.jsonl` comfortably:
```bash
head -1 runs/<run-folder>/parts.jsonl | python -m json.tool > part1.json
```
Then open `part1.json` in your editor.

---

## 5. Cost and time

- Each part takes about **15 seconds to 2 minutes** (the critic "thinks" before answering, which is the slow part).
- Each part costs about **¥0.015–0.1** (well under 2 US cents). The `Done:` line shows an estimate for every run.
- Prices **double** on weekdays during Beijing business hours (9:00–12:00 and 14:00–18:00 Beijing time).
- ⚠️ **Don't run big files or whole projects at file level by accident.** A file with 50 functions is
  50 parts: tens of minutes and a few yuan. Use `--chunk` to try one part first.

---

## 6. Settings

Settings live in your `.env` file. `.env.example` lists them all with their default values. To change
one, copy its line into `.env`, remove the `#` and change the value. Common ones:

| Setting | Default | What it does |
|---|---|---|
| `DEEPSEEK_MODEL` | `deepseek-flash` | The model for writing summaries (and for the critic, unless `DEEPSEEK_CRITIC_MODEL` is set). |
| `KNOWCODE_CRITIC_RUNS` | `2` | How many times the critic reviews each summary (1-3). With 2, the lower score counts. |
| `KNOWCODE_CRITIC_THINKING` | `true` | Whether the critic reasons step by step (slower, but catches more errors). |
| `KNOWCODE_PASS_THRESHOLD` | `4` | The critic score (out of 5) a summary needs to pass. |
| `KNOWCODE_MAX_QUALITY_RETRIES` | `1` | How many times a failing summary is rewritten. |

---

## 7. Tests and code checks

```bash
pytest -q          # all tests, free, no API key needed (about 2 seconds)
pytest -m live     # the few tests that call the real API (costs about ¥0.1)
ruff check .       # finds code problems
ruff format .      # tidies code formatting
```
Please run `pytest -q` and `ruff check .` before committing.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `knowcode: command not found` / `pytest: command not found` | The virtual environment isn't active. Run the activate command from section 3. |
| `DEEPSEEK_API_KEY is not set` | You haven't created `.env` or added the key (step 4). Make sure you run commands from the KnowCode folder, where `.env` is. |
| `HTTP 401: the API key was rejected` | The key is wrong or was deleted. Copy it again from the DeepSeek API keys page. |
| `HTTP 402: … no balance left` | Your DeepSeek account is out of credit. Top it up. |
| `Syntax error (line …)` | The file isn't valid Python 3 code. KnowCode only summarizes code that Python can read. |
| `no part named '…'` with `--chunk` | Check the spelling. Methods are named `Class.method`, e.g. `Cart.add`. |
| `requires a different Python` during install | Your Python is older than 3.12. Install 3.12+ and create the virtual environment again. |
| `UnicodeEncodeError` on Windows | Windows had trouble printing the ✅/⚠️ symbols. Save with `--output file.md` instead, or run `set PYTHONUTF8=1` (Command Prompt) / `$env:PYTHONUTF8=1` (PowerShell) first. |

Exit codes (for scripts): `0` finished, `1` bad input, `2` run stopped (key or credit problem), `130` stopped with Ctrl-C.

---

## 9. Experiment tools (`experiment/` folder)

The `experiment/` folder is **not part of the KnowCode tool**. It holds code for the research
experiment, which compares KnowCode's role-specific summaries against a generic baseline.

### Generic baseline summaries
A generic summary is written for **no particular reader**. It uses the same code splitting, file
context, model, settings, prompt template and automatic checks as `knowcode summarize`, but:
- there is no target role (one neutral list, "Key details", instead of a role's sections);
- there is **no critic review**: write → automatic checks → one retry only if the checks found a problem.

```bash
python -m experiment.baseline tests/fixtures/sample_cart.py --granularity function --chunk round_price
python -m experiment.baseline tests/fixtures/sample_cart.py --granularity file --format json --output cart_generic.json
```
It takes the same options as `knowcode summarize` except `--role`. Run it from the KnowCode folder
with the virtual environment active. Each part costs about one generator call (≈ ¥0.003), and
usually takes a few seconds.

### Telling runs apart in `runs/`
| | Folder name ends with | `run.json` → `condition` |
|---|---|---|
| `knowcode summarize --role qa …` | `-qa` (or `-developer`, `-pm`) | `role_specific` |
| `python -m experiment.baseline …` | `-generic` | `generic_baseline` |

`run.json` also has a one-sentence `condition_description`, and `inputs.role` is `null` for generic runs.

### First drafts (before the critic) are in the logs
For role-specific runs, `parts.jsonl` → `drafts[0]` is the summary as first written, **before** any
critic feedback, and `drafts[1]` (if present) is the rewrite. Comparing generic summaries with
`drafts[0]` isolates the effect of the role instructions alone; comparing with the final summary
(`best_draft`) measures the whole system, critic included.

---

## 10. Where things are in the code

| File | What it does |
|---|---|
| `src/knowcode/preprocessor.py` | Checks the code is valid Python and splits it into parts, plus the file's imports, constants and an outline of its other definitions. |
| `src/knowcode/prompts.py` | Builds the instructions sent to the model for each role (the summary writer, the critic and the retry). |
| `src/knowcode/llm.py` | Talks to DeepSeek, retrying network problems; also a fake client used by the tests. |
| `src/knowcode/validator.py` | Checks each reply with ordinary code: valid JSON, right fields, no invented names, no jargon for product managers. |
| `src/knowcode/pipeline.py` | The main loop: write → check → critic review → one retry → keep the best version. |
| `src/knowcode/output.py` | Turns the results into Markdown or JSON, and builds the whole-file document. |
| `src/knowcode/runlog.py` | Saves the run log. |
| `src/knowcode/cli.py` | The `knowcode` command. |
| `src/knowcode/config.py` | Settings, read from `.env`. |
| `tests/` | The tests (`pytest`). `tests/fixtures/sample_cart.py` is the example file. |
| `experiment/` | Experiment-only code, not part of the tool: `generic.py` (the generic reader's instructions) and `baseline.py` (generic baseline summaries). |
