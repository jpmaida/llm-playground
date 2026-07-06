def format_res(res, return_thinking=False):
    res = res.strip()

    if return_thinking:
        res = res.replace("<think>", "[pensando...]")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    
    return res