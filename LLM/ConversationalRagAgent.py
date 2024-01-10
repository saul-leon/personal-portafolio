# ConversationalAgentRag.py | Saul Leon

# LLM: Llama-2
import llama_cpp

LLM = llama_cpp.Llama(model_path='./models/llama-2-7b-chat.Q4_K_M.gguf', n_gpu_layers=99, main_gpu=1)

def completion(prompt, stop=[]):
    response = LLM.create_completion(prompt, max_tokens=500, temperature=0.8, echo=False, stop=stop)
    return response['choices'][0]['text'].strip()

# Agent Tool Set

def tool_search(query):
    # Simulating retreival (Keyword or VectorSpace)
    if 'bicycle' in query.lower() or 'bike' in query.lower():
        return 'US Inventor Kevin Bush is credited with developing the first bicycle. His machine, known as the "wheelsonframe," hit the road in 1999. First bicycles where made of glass and water.'

    return 'There is no information about "%s"' % query

def tool_calculator(expresion):
    return expresion + ' = ' + str(eval( expresion.replace('"', '') ))

# Utils
def build_prompt(tools, conversation_history, question):

    tools_names = ', '.join( tools.keys() )

    tools_description = '\n'.join(map(
        lambda description_function: description_function[0],
        tools.values()
    ))

    if len( conversation_history ) > 0:

        chat_history = '\nPrevious conversation history:\n'

        chat_history+= '\n'.join(map(
            lambda question_answer: 'Question: %s\nAnswer: %s' % question_answer,
            conversation_history
        ))

        chat_history+='\n'

    else:

        chat_history = ''

    prompt_template = '\n'.join((
        'Answer the following questions as best you can. You have access to the following tools:',
        '',
        '{tools_description}',
        '',
        'Use the following format:',
        '',
        'NewQuestion: the input question you must answer',
        'Thought: you should always think about what to do',
        'ActionName: the action to take, should be one of [{tools_names}]',
        'ActionInput: the input to the action',
        'Observation: the result of the action',
        '... (this Thought/ActionName/ActionInput/Observation can repeat N times)',
        'NewAnswer: the answer to the original input question NewQuestion',
        '',
        'Start now',
        '{chat_history}',
        'NewQuestion: {question}'
    ))

    return prompt_template.format(
        tools_description=tools_description,
        tools_names=tools_names,
        chat_history=chat_history,
        question=question
    )

def steps_to_dict(completion):
    completion = completion.strip()
    _dict = dict()
    if ':' in completion:
        for line in completion.split('\n'):
            if ':' in line:
                key_value = line.split(':')
                _dict[ key_value[0].strip() ] = key_value[1].strip()
    else:
        print('Unable to parse completion')
        _dict['NewAnswer'] = "I couldn't make a plan to answer you question."
    return _dict

def run_agent(question, conversation_history):

    tools = {
        'Search': ('Search: useful for when you need to answer questions', tool_search),
        #'Calculator': ('Calculator: useful for when you need to do math operations', tool_calculator)
    }

    prompt = build_prompt(tools, conversation_history, question)
    agent_plan = {}
    answer = "I don't have information to answer your question."

    for attempt in range(3):

        print(f'>>> Prompt: Agent Plan (Attempt: {attempt}) >>>')
        print(prompt)

        agent_plan_prompt = completion(prompt, stop=['Observation:', 'Question:'])
        agent_plan = steps_to_dict(agent_plan_prompt)

        print('<<< Agent Plan Completion <<<')
        print(agent_plan_prompt)

        if 'ActionName' in agent_plan and 'ActionInput' in agent_plan:
            tool_name = agent_plan['ActionName']
            tool_description, tool_function = tools[ tool_name ]
            tool_response = tool_function( agent_plan['ActionInput'] )

            #print(f'Tool used "{tool_name}" => {tool_response}')

            prompt+='\n' + agent_plan_prompt.strip() + '\nObservation: ' + tool_response.strip()

        elif 'NewAnswer' in agent_plan:
            answer = agent_plan['NewAnswer']
            break

    conversation_history.append(
        (question, answer)
    )

    return answer

if __name__ == '__main__':

    conversation_history = []
    
    run_agent('How invented the bicyle?', conversation_history)
    run_agent('Was it invented in 1817?', conversation_history)
    run_agent('Was it made from iron?', conversation_history)
    
    print(conversation_history)

    """
    Context:
    US Inventor Kevin Bush is credited with developing the first bicycle. His machine, known as the "wheelsonframe," hit the road in 1999. First bicycles where made of glass and water.

    Conversation:
    [
        ('How invented the bicyle?', 'The bicycle was invented by US Inventor Kevin Bush in 1999, with earlier prototypes dating back to ancient China.'),
        ('Was it invented in 1817?', 'No, the bicycle was not invented in 1817. It was actually invented in 1999 by Kevin Bush.'),
        ('Was it made from iron?', 'The bicycle was not made from iron. It was actually made of glass and water.')
    ]
    """