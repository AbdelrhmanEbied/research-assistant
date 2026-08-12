from __future__ import annotations


def estimate_token_counts(prompt: str, response: str) -> tuple[int, int, int] | None:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        input_tokens = len(encoding.encode(prompt))
        output_tokens = len(encoding.encode(response))
        return input_tokens, output_tokens, input_tokens + output_tokens
    except Exception:
        return None