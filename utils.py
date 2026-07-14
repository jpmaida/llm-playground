def format_res(res: str, return_thinking=False):
    res = res.strip()

    if return_thinking:
        res = res.replace("<think>", "[pensando...]")
        res = res.replace("</think>", "\n---\n")
    else:
        if "</think>" in res:
            res = res.split("</think>")[-1].strip()
    
    return res

def extract_thinking(res: str):
    res = res.strip()

    start_think_pos = res.find('<think>')
    end_think_pos = res.find('</think>')
    if start_think_pos != -1 and end_think_pos != -1:
        res = res[start_think_pos:end_think_pos]
        return res + "</think>"
    else:
        return ""