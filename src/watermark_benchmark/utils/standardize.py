""" Standardization """

def standardize(model, sys, user):
    """ Return a standardized version of the prompt for a given model """

    if "llama" in model and sys:
        if sys:
            return f"[INST] <<SYS>> {sys} <</SYS>> {user} [/INST]"
        else:
            return f"[INST] {user} [/INST]"

    if 'vicuna' in model or 'koala' in model:
        if sys:
            return f"System: {sys}\nHuman: {user}\nAssistant:"
        else:
            return f"Human: {user}\nAssistant:"

    else:
        raise NotImplementedError(f"No known standardization for model {model}. \
                                  Please add it manually to utils/standardize.py")
