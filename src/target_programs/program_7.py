from target_programs.functions_to_approximate import dl_textbook_fn

def program_7(x: float):
    y:float = dl_textbook_fn(x)

    if round(y, 0) == 100:
        raise Exception("You found a hard-to-reach bug!")

    if y > 100:
        print("dl_textbook_fn(x) > 100!", y)
    
    if y < 100:
        print("dl_textbook_fn(x) < 100!", y)