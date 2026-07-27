from ollama import chat
import json


def analyze_portfolio(portfolio_data, user_question):

    # Convert portfolio dictionary into readable text
    portfolio_summary = json.dumps(
        portfolio_data,
        indent=4
    )

    system_prompt = """
You are an educational investing assistant.

Your role:
- Explain portfolio metrics in simple language.
- Explain risk and return tradeoffs.
- Explain diversification concepts.
- Use only the provided portfolio information.

Rules:
- Do not give financial advice.
- Do not recommend buying or selling.
- Do not predict future returns.
- Explain that backtests represent historical performance only.
"""

    user_prompt = f"""
Here is the portfolio backtest data:

{portfolio_summary}


User question:

{user_question}


Provide an educational explanation.
"""


    response = chat(
        model="llama3.1:8b",

        messages=[
            {
                "role": "system",
                "content": system_prompt
            },

            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )


    return response["message"]["content"]