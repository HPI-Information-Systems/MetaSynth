import asyncio

async def fetch_completion(chat, client, llm_name, temperature, top_p=None, top_k=None):
    # This line will be awaited when the coroutine is executed
    if temperature is None:
        completion = await client.chat.completions.create(
            model=llm_name,
            messages=chat,
            timeout=60*60,
            max_tokens=2048,
        )
    elif top_p is not None and top_k is not None:
        completion = await client.chat.completions.create(
            model=llm_name,
            messages=chat,
            temperature=temperature,
            timeout=60*60,
            top_p=top_p,
            presence_penalty=1.5,
            max_tokens=2048,
            extra_body={
                "top_k": top_k, 
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
    else:
        completion = await client.chat.completions.create(
            model=llm_name,
            messages=chat,
            max_tokens=2048,
            temperature=temperature,
            timeout=60*60
        )
    return completion.choices[0].message.content

async def get_responses(chats, llm=None, sampling_params=None, client=None, llm_name=None, temperature=None, top_p=None, top_k=None):
    if llm is not None and sampling_params is not None:
        responses = llm.chat(chats, sampling_params, chat_template_kwargs={"enable_thinking": False})
        return [r.outputs[0].text for r in responses]
    elif client is not None and llm_name is not None:
        # Create a list of coroutine objects (tasks) for each chat completion request
        tasks = [fetch_completion(chat, client, llm_name, temperature, top_p, top_k) for chat in chats]
        
        # Run all tasks concurrently and gather the results
        responses = await asyncio.gather(*tasks)

        return responses
    else:
        raise ValueError("Either llm or client must be provided.")