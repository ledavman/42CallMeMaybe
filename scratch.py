import os
import time

os.environ["HF_HOME"] = "/goinfre/drenck"

from llm_sdk import Small_LLM_Model  # noqa: E402
from src.errors import DecodingError, InputError  # noqa: E402
from src.loaders import load_functions, load_prompts  # noqa: E402
from src.pipeline import build_call, make_context  # noqa: E402

functions = load_functions("data/input/functions_definition.json")
prompts = load_prompts("data/input/function_calling_tests.json")

model = Small_LLM_Model()
ctx = make_context(model, functions)
print("tokens nombre :", len(ctx.num_tokens),
      "| tokens chaîne :", len(ctx.str_tokens))
print("ids :", ctx.quote_id, ctx.comma_id, ctx.brace_id)

# Commence par UN seul prompt, puis enlève [:1]
for item in prompts:
    start = time.perf_counter()
    try:
        call = build_call(item, ctx)
        print(call.model_dump_json())
    except (DecodingError, InputError) as e:
        print("ERREUR :", e)
    print(f"{time.perf_counter() - start:.1f}s")