

vocab = load_vocab(model.get_path_to_vocab_file())
names = [f.name for f in functions]
candidates = candidate_tokens(vocab, names)     # une seule fois
quote_id: int = {text: id for id, text in vocab.items()}['"']

for item in prompts:
    text = build_prompt(functions, item.prompt)
    prompt_ids = model.encode(text).tolist()[0]
    name = generate_function_name(
        model.get_logits_from_input_ids,
        prompt_ids, candidates, names, quote_id,
    )
    print(item.prompt, "->", name)