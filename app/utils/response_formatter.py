def format_response(response):
    if "Espero que essas informações sejam úteis!" in response:
        response = response.replace("Espero que essas informações sejam úteis!\n", "")

    if "Seu nome é qual?" in response:
        response = response.replace("Seu nome é qual?", "")

    if "Meu nome é Aurora" not in response:
        response = "Ah, sim! Eu sou a Aurora, prazer em te conhecer!"

    if response.lower().startswith("qual seu nome?"):
        response = "Ah, meu nome é Aurora! Como posso te ajudar hoje?"

    response = response.replace("\n\n", "\n").strip()

    return response