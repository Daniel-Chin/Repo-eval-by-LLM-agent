'''
用 UI 思路 实现 LLM Agent Interface (LAI) 的 top-level code 示例。  
可以作为真的骨架。  
top-level 指的是，你只需要提供 render() 和 systemPrinciples(), 然后执行 main() 就可以保持整个 LLM-in-the-loop application 运转。  
请阅读理解本代码，有问题问 Dan.  

Pipeline:  
1. LAI sends system principles to LLM agent.  
2. LAI sends screen to LLM agent.  
3. LLM agent calls "summarizeScreen(summary)".  
4. LLM agent uses CoT to reason and plan.  
5. LLM agent calls function(s) among available ones.
6. LAI handles each function call and returns `how_long_to_wait` for each.  
7. Wait for the *minimum* wait time among all `how_long_to_wait`s.
8. Go to step 2.  
'''

import typing as tp
import asyncio
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from functools import cached_property
import itertools
import json

import openai
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.shared.function_definition import FunctionDefinition
from openai.types.shared_params.function_definition import FunctionDefinition as FunctionDefinitionParam

UIEventHandler = tp.Callable[..., float]    # returns how long to wait (in sec) before querying the LLM agent again. Use math.inf to wait forever.  

class LAIAsUI(metaclass=ABCMeta):
    @abstractmethod
    def systemPrinciples(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> tp.Tuple[
        str, str, tp.List[tp.Tuple[
            FunctionDefinition, UIEventHandler, 
        ]], 
    ]:
        '''
        override this to return `(instructions, screen_as_text, functions)`
        `instructions` is not the system principle. It's the specific instructions at the current time.
        `screen_as_text` is a verbose text description of the current state.
        `functions` lists available functions at the current state.
        '''
        raise NotImplementedError()

@dataclass(frozen=True)
class LAIReply:
    '''
    The LAI's reply to the LLM agent.
    '''
    id_: int
    instructions: str
    screen_as_text: str
    screen_summary: str | None = None

    @cached_property
    def for_agent(self):
        if self.screen_summary is None:
            return f'''
{self.instructions}
<screen id={self.id_}>
{self.screen_as_text}
</screen>
            '''.strip()
        else:
            return f'''
{self.instructions}
<screen_summary id={self.id_}>
{self.screen_summary}
</screen_summary>
            '''.strip()

SUMMARIZE_SCREEN = FunctionDefinition(
    name = 'summarizeScreen',
    description = 'Look at the current screen. Summarize what you do not already know. If there is no notable info, return "Everything is normal".',
    parameters = dict(
        type = 'object',
        properties = {
            'screen_id': {
                'type': 'int',
                'description': 'ID of the screen.',
            },
            'summary': {
                'type': 'string',
                'description': 'Your summary.',
            },
        },
        required = ['screen_id', 'summary'],
    ),
)

async def main(
    laiAsUI: LAIAsUI, client: openai.OpenAI, 
    **kwargs_chat_completion_create, 
):
    chat_history: tp.List[ChatCompletionMessage | LAIReply] = []
    def Messages(last: LAIReply) -> tp.Generator[ChatCompletionMessageParam, None, None]:
        yield {
            'role': 'system',
            'content': laiAsUI.systemPrinciples(),
        }
        for m in itertools.chain(chat_history, [last]):
            if isinstance(m, ChatCompletionMessage):
                yield tp.cast(ChatCompletionMessageParam, m.model_dump())
            elif isinstance(m, LAIReply):
                yield {
                    'role': 'user',
                    'content': m.for_agent,
                }

    id_iter = itertools.count(0)
    while True:
        screen_id = next(id_iter)
        instructions, screen_as_text, functions = laiAsUI.render()
        
        async def summarize():
            to_summarize = LAIReply(
                id_=screen_id,
                instructions='Summarize the current screen.',
                screen_as_text=screen_as_text,
            )
            completion = client.chat.completions.create(
                messages=Messages(to_summarize),
                tool_choice='required',
                tools=[
                    {
                        'type': 'function',
                        'function': tp.cast(
                            FunctionDefinitionParam, 
                            SUMMARIZE_SCREEN.model_dump(), 
                        ),
                    },
                ],
                **kwargs_chat_completion_create,
            )
            assert isinstance(completion, ChatCompletion)
            tool_calls = completion.choices[0].message.tool_calls
            assert tool_calls is not None
            kwargs = json.loads(tool_calls[0].function.arguments)
            assert kwargs['screen_id'] == screen_id, (kwargs, screen_id)
            summary = kwargs['summary']
            assert isinstance(summary, str), type(summary)
            return summary

        async def react():
            to_react = LAIReply(
                id_=screen_id,
                instructions=instructions,
                screen_as_text=screen_as_text,
            )
            completion = client.chat.completions.create(
                messages=Messages(to_react),
                tools=[
                    {
                        'type': 'function',
                        'function': tp.cast(
                            FunctionDefinitionParam, 
                            functionDef.model_dump(), 
                        ),
                    } for functionDef, _ in functions
                ],
                **kwargs_chat_completion_create,
            )
            assert isinstance(completion, ChatCompletion)
            tool_calls = completion.choices[0].message.tool_calls
            if tool_calls is None:
                return 0.0
            time_to_wait: tp.List[float] = []
            for tool_call in tool_calls:
                fCall, = [
                    fCall for fDef, fCall in functions 
                    if fDef.name == tool_call.function.name
                ]
                kwargs = json.loads(tool_call.function.arguments)
                time_to_wait.append(fCall(**kwargs))
            return min(time_to_wait)
        
        summary, min_wait_time = await asyncio.gather(
            summarize(), react(),
        )

        chat_history.append(
            LAIReply(
                id_=screen_id,
                instructions=instructions,
                screen_as_text=screen_as_text,
                screen_summary=summary,
            ),
        )

        await asyncio.sleep(min_wait_time)
