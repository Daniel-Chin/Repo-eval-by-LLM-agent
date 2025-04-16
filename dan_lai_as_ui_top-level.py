'''
用 UI 思路 实现 LLM Agent Interface (LAI) 的 top-level code 示例。  
可以作为真的骨架。  
top-level 指的是，你只需要提供 render() 和 handle() 和 systemPrinciples(), 然后执行 main() 就可以保持整个 LLM-in-the-loop application 运转。  
请阅读理解本代码，有问题问 Dan.  

Pipeline:  
1. LAI sends system principles to LLM agent.  
2. LAI sends screen to LLM agent.  
3. LLM agent calls "summarizeScreen(summary)".  
4. LLM agent uses CoT to reason and plan.  
5. LLM agent calls a function among available ones.
6. LAI handles the function call and returns `how_long_to_wait`.  
7. After waiting, go to step 2.  
'''

import typing as tp
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod

from openai.types.chat.chat_completion_message import ChatCompletionMessage

class LAIAsUI(metaclass=ABCMeta):
    @abstractmethod
    def systemPrinciples(self) -> str:
        raise NotImplementedError()

    @abstractmethod
    def render(self) -> tp.Tuple[str, str]:
        '''
        override this to return `(instructions, screen_as_text)`
        `instructions` is not the system principle. It's the specific instructions at the current time.
        `screen_as_text` is a verbose text description of the current state.
        '''
        raise NotImplementedError()
    
    @abstractmethod
    def handle(self, function_call: str) -> tp.Tuple[str, float]:
        '''
        override this to handle the function call.  
        return how long to wait (in sec) before querying the LLM agent again.  
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

    def asText(self):
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
    
    def summarized(self, summary):
        return LAIReply(
            id_=self.id_,
            instructions=self.instructions,
            screen_as_text=self.screen_as_text,
            screen_summary=summary,
        )

def main(laiAsUI: LAIAsUI):
    chat_history: tp.List[ChatCompletionMessage | LAIReply] = []
    ...
