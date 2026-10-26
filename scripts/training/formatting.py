"""Prompt construction and loss boundaries, independent of torch/transformers."""


def assistant_labels(prompt_ids, full_ids):
    if len(full_ids) <= len(prompt_ids) or list(full_ids[:len(prompt_ids)]) != list(prompt_ids):
        raise ValueError('tokenized completion must strictly extend the exact prompt prefix')
    return [-100] * len(prompt_ids) + list(full_ids[len(prompt_ids):])


def make_messages(row, *, history_len=0):
    if type(history_len) is not int or not 0 <= history_len <= 32:
        raise ValueError('invalid history length')
    messages = [{'role': 'system', 'content': row['system_prompt']}]
    images = []

    def user(rgb):
        images.append(rgb['path'])
        messages.append({'role': 'user', 'content': [
            {'type': 'text', 'text': 'current game screen'}, {'type': 'image'}]})

    for past in row['history'][-history_len:] if history_len else []:
        user(past['rgb'])
        messages.append({'role': 'assistant', 'content': past['action']})
    user(row['rgb'])
    messages.append({'role': 'assistant', 'content': '<think>\n\n</think>'})
    return messages, images
