import chainlit as cl


class ChatUI:
    def __init__(self, openai_provider, router_llm):
        self.openai_provider = openai_provider
        self.router_llm = router_llm
        cl.on_chat_start(self.start)
        cl.on_message(self.on_message)

    async def start(self):
        cl.user_session.set(
            "history",
            [
                {
                    "role": "system",
                    "content": self.openai_provider.prompt_system,
                }
            ],
        )
        await cl.Message(content="Bonjour !").send()

    async def on_message(self, message: cl.Message):
        history = cl.user_session.get("history")
        history.append({"role": "user", "content": message.content})

        msg = cl.Message(content="")
        await msg.send()

        #### ROUTING #####
        model_routed = self.router_llm.route(message.content)
        print(f"Routed to model: {model_routed}")
        ##################

        response = self.openai_provider.completion_tools(
            messages=history, model=model_routed
        )
        choice = response.choices[0].message

        if choice.tool_calls:
            history.append(choice.model_dump())
            for tool_call in choice.tool_calls:
                async with cl.Step(name=tool_call.function.name, type="tool") as step:
                    step.input = tool_call.function.arguments
                    result = self.openai_provider.tools.execute(tool_call)
                    step.output = str(result)

                history.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )

        async for part in self.openai_provider.completion_stream(
            messages=history, model=model_routed
        ):
            if token := part.choices[0].delta.content or "":
                await msg.stream_token(token)

        history.append({"role": "assistant", "content": msg.content})
        cl.user_session.set("history", history)

        msg.content += f"\n\n*— Modèle`{model_routed}`*"
        await msg.update()
