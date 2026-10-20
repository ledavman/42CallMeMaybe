*This project has been created as part of the 42 curriculum by drenck.*

# call me maybe

Introduction to function calling in LLMs, using constrained decoding on a small language model (Qwen3-0.6B).

## Description

Given a natural-language request such as `"What is the sum of 40 and 2?"`, the program does **not** answer the question. It outputs the function call that would solve it:

```json
{"prompt": "What is the sum of 40 and 2?", "name": "fn_add_numbers", "parameters": {"a": 40.0, "b": 2.0}}
```

A 0.6B-parameter model is unreliable at producing JSON when simply asked to. The goal of this project is to guarantee 100% valid, schema-compliant output anyway, by **constraining the model token by token** instead of relying on prompting. The function to call is always chosen by the LLM, never by keyword heuristics.

**Overview of the program**

1. Read the available functions (`functions_definition.json`) and the prompts (`function_calling_tests.json`).
2. For each prompt, build a text for the model to continue, ending right where the function name begins.
3. Generate the function name under constraint, then each parameter value under constraint.
4. Build the result as a Python object and write `function_calls.json`.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- About 2 GB of free disk space: the model (about 1.5 GB) is downloaded from Hugging Face on the first run

### Installation

```bash
make install        # runs: uv sync
```

### Run

```bash
make run            # uses the default paths below
```

or directly:

```bash
uv run python -m src [--functions_definition <file>] [--input <file>] [--output <file>]
```

| Option | Default |
|---|---|
| `--functions_definition` | `data/input/functions_definition.json` |
| `--input` | `data/input/function_calling_tests.json` |
| `--output` | `data/output/function_calls.json` |

The output directory is created if it does not exist.

### Other Makefile rules

| Rule | What it does |
|---|---|
| `make debug` | runs the program under `pdb` |
| `make lint` | `flake8 .` and `mypy .` with the mandatory flags |
| `make lint-strict` | `flake8 .` and `mypy . --strict` |
| `make clean` | removes `__pycache__` and `.mypy_cache` |

**Note on disk space:** the Hugging Face cache can be moved with the `HF_HOME` environment variable, for example `export HF_HOME=/path/with/space/hf_cache`.

## Example usage

Input (`function_calling_tests.json`):

```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Output (`function_calls.json`):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

Errors (missing file, invalid JSON, wrong structure...) are reported on `stderr` with a clear message and a non-zero exit code, never with a traceback.

## Algorithm explanation

### Token-by-token generation

An LLM produces text one token at a time. At each step, `get_logits_from_input_ids` returns one score per token of the vocabulary (151,936 values for Qwen3-0.6B), for the **next** position only. The chosen token is appended to the input and the process repeats.

### Constrained decoding

Before choosing a token, the program computes the set of token ids that keep the output valid, then picks the highest-scoring token **among those only**. This is equivalent to setting the logits of every invalid token to `-inf` and taking the argmax, but it avoids building a 151,936-element array at each step: only the scores of the allowed ids are read.

The model therefore keeps the decision (which function? which value?) but is physically unable to leave the format.

### Mapping tokens to text

`vocab.json` maps token strings to ids. The program inverts it (id to text) to know what each token would write. The vocabulary uses a byte-level BPE: every byte has a printable stand-in character (for example `Ġ` stands for a space). The conversion is done by rebuilding that byte-to-character table and decoding the bytes as UTF-8.

Tokens are **excluded** when they have no entry in the vocabulary (293 ids of the logits vector) or when their bytes are only a fragment of a multi-byte character (invalid UTF-8 on its own).

### The prompt

The prompt lists each function with its description, gives the user's request, and ends with `Answer: {"name": "`. The model only continues text, so the most natural continuation of this prompt is a function name.

### Choosing the function name

At each step, a token is allowed if the text generated so far plus the text of the token is still the **beginning of at least one function name** (`str.startswith`). The closing quote is allowed only when the text generated so far is exactly a complete function name. Two consequences:

- the model can never invent a function that does not exist;
- when two names share a prefix (`fn_add` and `fn_add_numbers`), the model decides whether to stop or to continue.

Optimizations: the tokens whose text cannot appear in any function name are filtered out once at start-up, and when exactly one token is allowed the model is not called at all.

### Choosing the parameters

The JSON skeleton (`", "parameters": {`, the keys, the separators, the closing quote) is **written by the program** and encoded into the model's context. The model only generates the **values**, one at a time, in the order of the function definition, each under a constraint that depends on its type:

| Type | Constraint |
|---|---|
| `number` | only tokens made of digits, `.` and `-`; the text must always match a valid number prefix (regular expression); the stop token (`,` or `}`) is allowed only once the number is complete |
| `integer` | same as `number` without the decimal point |
| `string` | every token without a double quote, a backslash or a non-printable character; ends on any token starting with `"` |
| `boolean` | choice between `true` and `false`, using the same mechanism as the function name |

Every generator has a maximum number of tokens, so that a model that never decides to stop cannot block the program.

### Writing the output

The result is a Python object validated by pydantic and written with `json.dump`. The model never writes JSON syntax itself, so the output file is always valid JSON with the exact keys required (`prompt`, `name`, `parameters`), and values have the types declared in the function definition.

## Design decisions

- **The program writes the structure, the model writes the values.** Fixed parts of the JSON are encoded and appended to the context; this removes whole classes of errors and saves model calls.
- **One shared context computed once.** The inverted vocabulary and the filtered token sets depend only on the vocabulary and the function names, not on the prompt, so they are computed once before the loop over the prompts.
- **Decoders take a function instead of the model.** The generators receive a `logits_fn` and an `encode` function. This keeps them independent of `llm_sdk` and testable with a fake model.
- **Partial-character tokens are excluded** from string generation. The model cannot produce a character that exists only as byte fragments (some rare characters or emojis). Common accented characters have their own token and are not affected.
- **Double quotes and backslashes are not allowed inside generated strings.** A quote is the end signal, and escape sequences are avoided. The trade-off is that a value containing these characters cannot be generated.
- **Any token starting with `"` ends a string** (see the challenges below). The text after the quote in that token is ignored because the program writes the following fixed text itself.
- **Pydantic for every class**, with `Literal` types for the allowed parameter types, so an unsupported type in a definition file is rejected with a clear message at loading time instead of failing later.
- **Fail fast on input errors.** The first invalid file stops the program with a clear message; the model is loaded only after both input files have been validated.
- **A common base class for errors**, so the entry point handles every expected error in one place.
- **Duplicate function names** are not rejected. **TODO: decide and document what happens.**
- **A prompt that fails to decode** is **TODO: describe the chosen behavior (skipped with a warning on stderr, or other)**.

## Performance analysis

Measured on **TODO: describe the machine**.

| Metric | Result |
|---|---|
| Valid JSON | 100% by construction (the file is written from Python objects) |
| Accuracy on the 11 example prompts | 11/11 correct function and arguments (checked by hand). **TODO: update after the final run and the larger tests** |
| Time per model call (short prompt) | about 0.2 s |
| Time for the 11 example prompts | about 60 s (2 to 5 s for simple prompts, 10 to 15 s for the regex-substitution ones) |
| Number tokens kept | 54 |
| String tokens kept | 145,415 out of 151,643 vocabulary entries |

Each call to the model recomputes the whole sequence, so long prompts (many functions, long descriptions) and long generated strings are the main cost. **TODO: add what you measured on a larger input file and how it compares to the 5-minute limit.**

## Challenges faced

- **Not enough disk space for the model.** The first load failed with `No space left on device` while downloading about 1.5 GB. Solved by moving the Hugging Face cache with `HF_HOME`.
- **Linting the virtual environment.** `flake8 .` crashed with a recursion error inside `.venv`. Solved with a `.flake8` file excluding `.venv` and the provided `llm_sdk` directory.
- **Understanding the SDK output.** `encode` returns a tensor of shape `[1, N]` while `get_logits_from_input_ids` expects a flat list of ints, so the call is `encode(text).tolist()[0]`. The logits vector (151,936) is also larger than the vocabulary file (151,643 entries), so some ids have no text and must always be forbidden.
- **Strings that never ended, and wrong regular expressions.** On the regex-substitution prompts, two generated strings looped until the token limit (about 150 s each), and another produced `cat.*cat.*` instead of `cat`. Printing the top-scoring tokens at each step showed that the model wanted to close the string with combined tokens such as `",`, `"}` or `"}}\n`, which were forbidden because they contain a quote. Only the bare `"` was allowed, which the model rarely picks, so continuing always won. Allowing every token that starts with a quote as an end-of-string marker fixed both the loops and the wrong results.
- **Choosing a function when names share a prefix.** The closing quote has to be allowed exactly when the text is a complete name, and not before.
- **Spurious leading space in generated strings.** **TODO: describe the rule you kept (or not) for the first token of a string, and why.**

## Testing strategy

- **Input errors, by hand:** missing file, a directory instead of a file, invalid JSON, a JSON value that is not a list, an item with a missing field, an empty list, invalid UTF-8 bytes, and unknown command-line options. Each one must give a clear message, a non-zero exit code and no traceback.
- **Constrained decoding, with a fake model:** unit tests (pytest) call the decoding functions with a fake `logits_fn` whose favorite token is always a forbidden one, and check that the result is still valid.
- **End to end:** the program is run on the example files and the output is checked with `python3 -m json.tool`, then compared with the expected function and arguments.
- **Edge cases:** **TODO: list what you tested (empty strings, large numbers, special characters, ambiguous prompts, functions with several parameters, a function without parameters, other function sets).**

## Project structure

```
src/
├── __main__.py   entry point: arguments, orchestration, error handling
├── parser.py     command-line arguments
├── errors.py     program exceptions
├── models.py     pydantic models
├── loaders.py    reading and validating the input files
├── vocab.py      token id to text mapping
├── prompt.py     prompt construction
├── decoder.py    generic decoding tools and function-name generation
├── values.py     generation of numbers, strings and booleans
├── pipeline.py   shared context and one-prompt-to-call logic
└── writer.py     writing the output file
llm_sdk/          provided package (not modified)
data/input/       example input files
```

**TODO: adjust this tree to match the final repository.**

## Resources

**Documentation**

- Python standard library: [argparse](https://docs.python.org/3/library/argparse.html), [re](https://docs.python.org/3/library/re.html), [json](https://docs.python.org/3/library/json.html)
- [pydantic](https://docs.pydantic.dev/)
- [uv](https://docs.astral.sh/uv/)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)

**Articles**

- Willard and Louf, *Efficient Guided Generation for Large Language Models* (arXiv:2307.09702). Read to understand the idea only; the library built on it is not used here, as the subject forbids it.
- Sennrich, Haddow and Birch, *Neural Machine Translation of Rare Words with Subword Units* (arXiv:1508.07909), the origin of BPE.
- Radford et al., *Language Models are Unsupervised Multitask Learners* (2019), which introduced byte-level BPE.

### How AI was used

An AI assistant (Claude) was used as a tutor and reviewer during the project:

- **Explaining concepts:** constrained decoding, logits, byte-level BPE and the `Ġ` convention, how a tokenizer splits numbers and quotes.
- **Planning:** a step-by-step roadmap and the split of the code into modules.
- **Reviewing my code and finding bugs**, which I then fixed myself: an `argmax` returning a value instead of a position, a substring test used where an equality test was needed, a generator expression that was always true, uninitialized variables, and a falsy check on token id `0`.
- **Code written with the assistant** and then read, adapted and tested by me: the byte-to-character table in `vocab.py`, the number, string and boolean generators, the shared context and the one-prompt pipeline, and the skeletons of `pyproject.toml`, `.flake8` and the Makefile.
- **Throwaway experiment scripts** to inspect the SDK and the model's predictions (not part of the repository).
- **A first draft of this README**, reviewed and completed by me.

**TODO: edit this list so that it matches exactly what you did, and make sure you can explain every part of the code listed above.**
