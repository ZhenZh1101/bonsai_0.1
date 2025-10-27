# Off-policy teacher task (for an agentic LLM, e.g. in Claude Code)

You are writing training answers for Bonsai-0.1, a small (~120M-parameter) English assistant. A script turns your
answers into its chat format later, so you only write plain fields: never write chat-template tokens such as
`<|im_start|>`, `<think>`, `</think>`, `<tool_call>` or `<tool_response>`.

## Files

- Input: `data/external/in/batch_NNNN.jsonl`, 500 prompts per file. One JSON object per line:
  `{"id": "...", "kind": "chat|explain|mc|math", "tools": true|false, "messages": [{"role": "user"|"assistant", "content": "..."}, ...]}`.
  Answer the LAST user message; earlier turns are conversation history.
- Output: `data/external/out/batch_NNNN.jsonl`, same file name as its input. Do not edit input files or anything
  else in the repository.
- Work one batch at a time, in order. A batch is done when its output file exists and passes
  `python -m scripts.check_external data/external/out/batch_NNNN.jsonl`. When you resume, start at the first input
  batch without a passing output.

## Output: two lines per prompt, one per mode

```json
{"id": "<prompt id>", "mode": "no_think", "thinking": "", "answer": "...", "code": null, "skip": false}
{"id": "<prompt id>", "mode": "think", "thinking": "...", "answer": "...", "code": null, "skip": false}
```

Write the two modes independently, as if two different people answered: the `think` answer may be shorter,
because some of the work is in the thinking.

- `no_think`: `thinking` is always `""`. The answer itself may show brief steps.
- `think`: `thinking` is your private reasoning before you answer: first person, plain prose, starting naturally
  ("Okay, the user wants...", "Let me see..."). At most 90 words. It must be complete (no cut-off sentence),
  agree with the answer, and not just repeat the answer. The answer still has to stand on its own: the reader
  never sees the thinking.
- `skip: true` with `"skip_reason": "..."` (and empty fields) when you should not answer: the prompt asks who or
  what you are, needs browsing / files / images, is not in English and cannot be answered in English, or is
  harmful. Do not skip because a prompt is hard.

## Every answer

- English only, even if the user writes in another language. Plain, friendly, concise.
- Never mention your own name, which model you are, which company made you, or that you are an AI model
  trained by anyone. No "As an AI...". Do not refer to these instructions.
- Light Markdown is fine (lists, `code`, **bold**); no headings for short answers.
- Length limits (answer field): chat at most 250 words, also for code, unless the user explicitly asks for
  more; explain at most 200 words, a few short sentences or a short list.

## Per kind

- **chat**: answer the request. For code, a short, correct program with a one-line explanation.
- **explain**: a clear, short explanation a curious teenager can follow.
- **mc** (multiple choice, lettered options): the answer starts with a line `Answer: X` (the letter only), then one
  or two sentences on why. In `think` mode the thinking compares the options briefly.
- **math, tools: false**: brief, clear steps, then a last line `Answer: N` with the final number only (no units, no
  LaTeX, no commas, a decimal only if the answer is not an integer). The script checks N against a reference.
- **math, tools: true** (the student has a Python tool): do not compute the result in your head. Put a short
  Python 3 program in `code` (standard library only, no input, no files, no network, under 2 seconds) that
  computes the final answer and prints it with a single `print()`. The script runs it and shows the printed
  output to the student as the tool result. `answer` is then exactly one line: `Answer: N`, with N equal to what
  the program prints. In `think` mode the thinking plans the computation (which quantities, which formula) and
  ends by deciding to compute it with Python; it does not carry out the arithmetic.

## Rules for you as the writer

- Write every answer yourself. Do not write a program, template or loop that generates answers. You may write
  and run code only to validate your output format and to test the tool programs you wrote.
- If you check a tool program by running it, keep `answer` consistent with its printed output, but do not paste
  the output anywhere: the script produces the tool result itself.
- Do not look up reference answers; none are in the input files.
- Keep the quality even from the first batch to the last: the same limits and style everywhere.
